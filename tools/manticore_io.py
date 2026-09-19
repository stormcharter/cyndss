#!/usr/bin/env python3
"""
Спільне I/O-ядро проходів по індексу Manticore.

НАВІЩО. Проходів уже два — `purge_revoked_ttps.py` (відкликані/deprecated
техніки MITRE) і `purge_ungrounded_ttps.py` (мітки без опори в тексті), — і
кожен вів власну копію тієї самої чорної роботи: SQL у Manticore, /bulk
replace, читання тексту з Mongo. Копії розʼїхались, і виправлення, зроблені в
одній, у другій не зʼявились. Тепер розмова з Manticore й Mongo живе тут в
одному екземплярі.

ТРИ ГРАБЛІ, ЗАКРИТІ ТУТ (усі три однакові тим, що спрацьовують МОВЧКИ —
інструмент не падає, а бадьоро звітує, що робити нічого):

1. `SELECT` віддає щонайбільше **1000** рядків незалежно від `LIMIT`, поки не
   задано `OPTION max_matches`. Прохід «бачив увесь корпус», а бачив 1000 із
   1540. -> `build_scan_query` завжди ставить стелю в рівень з LIMIT.
2. id документа зберігається як **unsigned** 64-bit, а `SELECT` віддає його
   **signed**: усе, що більше за 2^63, приїжджає відʼємним, і `/bulk` на такий
   id відповідає 400 Bad Request. -> згортання живе в `build_bulk_line`, а не в
   місці виклику: інакше кожен новий прохід мусить памʼятати про це самотужки.
3. Mongo під `--auth`, а текст тягнули через `docker exec mongosh` без
   креденшелів — «текст недоступний» на КОЖНОМУ рядку. -> `fetch_texts` бере
   тексти одним запитом через pymongo (заодно без 379 підпроцесів і без пароля
   в argv). Запускати з venv, де pymongo є.

Модуль без сторонніх залежностей, крім pymongo у `fetch_texts` (імпорт
локальний, щоб решта функцій лишалась доступною host-пітоном).
"""

import json
import re
import urllib.parse
import urllib.request

#: Стеля вибірки за замовчуванням. Корпус ~1.5k документів, тож із запасом.
DEFAULT_SCAN_LIMIT = 100000

_OBJECT_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")


# =====================================================================
#  Чисті функції (покриті tests/test_manticore_io.py)
# =====================================================================
def build_scan_query(cols, table, where="", limit=DEFAULT_SCAN_LIMIT):
    """SQL для перегляду рядків індексу.

    `OPTION max_matches` обовʼязковий і завжди дорівнює LIMIT: без нього
    Manticore мовчки віддає 1000 рядків, а нижча за LIMIT стеля так само тихо
    обріже вибірку. `where` — уже готова умова без слова WHERE (порожня =
    дивимось усе)."""
    clause = f" WHERE {where}" if where else ""
    return (f"SELECT {cols} FROM {table}{clause} "
            f"LIMIT {limit} OPTION max_matches={limit}")


def manticore_doc_id(value):
    """id документа в тому вигляді, в якому його приймає /bulk.

    Manticore тримає id як unsigned 64-bit, а SELECT віддає signed."""
    doc_id = int(value)
    return doc_id + 2 ** 64 if doc_id < 0 else doc_id


def build_bulk_line(table, doc_id, doc):
    """Один NDJSON-рядок `replace` для /bulk.

    Згортання знакового id — саме тут, а не в місці виклику: так жоден новий
    прохід по індексу не зможе про нього забути."""
    return json.dumps(
        {"replace": {"table": table, "id": manticore_doc_id(doc_id), "doc": doc}},
        default=str,
    )


def is_valid_mongo_id(value):
    """Чи є value канонічним 24-символьним hex ObjectId."""
    return isinstance(value, str) and bool(_OBJECT_ID_RE.match(value))


def valid_object_ids(ids):
    """Відсіює нестандартні id і дублікати, зберігаючи порядок.

    Тексти тягнуться одним запитом по списку _id, тож мігровані/легасі-рядки,
    які в ObjectId не перетворяться, треба прибрати ДО запиту — падати через
    один такий рядок на всьому проході не можна."""
    seen, out = set(), []
    for value in ids or []:
        if not is_valid_mongo_id(value) or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def parse_embedding(raw):
    """SELECT віддає embedding рядком 'f1,f2,...' -> список float."""
    if not raw:
        return []
    return [float(x) for x in str(raw).split(",") if x.strip() != ""]


def env_value(key, path=".env"):
    """Значення ключа з .env (той самий файл, що читають колектори й shipper)."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


# =====================================================================
#  Мережа (виконується на сервері; юніт-тестами не покрите)
# =====================================================================
def mano_sql(base_url, query):
    """POST SQL у Manticore (/sql?mode=raw) -> перший результат-сет."""
    data = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        f"{base_url}/sql?mode=raw", data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.loads(r.read().decode())
    res = payload[0] if isinstance(payload, list) else payload
    if res.get("error"):
        raise RuntimeError(f"Manticore SQL error: {res['error']}")
    return res


def rows(res):
    """Result-set Manticore -> список dict-рядків."""
    return res.get("data", [])


def bulk_replace(base_url, table, doc_id, doc):
    """Один replace через /bulk (NDJSON)."""
    body = (build_bulk_line(table, doc_id, doc) + "\n").encode()
    req = urllib.request.Request(f"{base_url}/bulk", data=body,
                                 headers={"Content-Type": "application/x-ndjson"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def fetch_texts(mongo_uri, db_name, collection, mongo_ids):
    """Тексти документів ОДНИМ запитом -> {mongo_id: text}.

    Через pymongo, а не `docker exec mongosh`: Mongo на сервері під --auth, і
    кожен виклик mongosh або лишався б без креденшелів, або світив пароль в
    argv. Запускати з venv, де pymongo є."""
    from bson import ObjectId          # noqa: PLC0415 — залежність лише цього шляху
    from pymongo import MongoClient    # noqa: PLC0415

    ids = valid_object_ids(mongo_ids)
    if not ids:
        return {}
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=15000)
    try:
        cursor = client[db_name][collection].find(
            {"_id": {"$in": [ObjectId(i) for i in ids]}}, {"text": 1})
        return {str(d["_id"]): d.get("text", "") for d in cursor}
    finally:
        client.close()


def mongo_uri_from_env(path=".env"):
    """COLLECTOR_MONGO_URI з .env — той самий рядок, яким ходять колектори й UI."""
    return env_value("COLLECTOR_MONGO_URI", path)
