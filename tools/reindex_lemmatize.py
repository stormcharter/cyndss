"""
Дешева переіндексація Manticore під нову схему (морфологія) + лематизований
content — БЕЗ повторного ембеддингу/екстракції/верифікації.

ІДЕЯ. Важкі поля (embedding, title, summary, verified_*) уже пораховані й лежать
у старому індексі як stored — їх можна ПРОЧИТАТИ назад. Єдине, що змінюється, —
схема (morphology/min_infix) та поле content (тепер лематизоване). content не
stored, тож його відновлюємо з оригіналу в Mongo (по mongo_id) і лематизуємо.

Отже: SELECT старих рядків -> дотягуємо оригінали з Mongo -> DROP+CREATE нова
схема -> REPLACE усіх рядків із лематизованим content і збереженими embedding.
Жодного Ollama/LLM/NVD. Індекс перебудовний із Mongo, тож нічого не втрачається.

ЗАПУСК (на сервері, з UI-venv — там є pymongo + pymorphy3):
    python tools/reindex_lemmatize.py                 # DRY-RUN: лише рахує й звіряє
    python tools/reindex_lemmatize.py --yes           # реально DROP+перебудова

Manticore тягнемо по ХОСТ-URL (127.0.0.1:9998), не docker-internal:
    MANTICORE_HTTP_URL (деф. http://127.0.0.1:9998), MANTICORE_TABLE (events_enriched)
    MONGO_URI / MONGO_DB / MONGO_COLLECTION (як у ui.py)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemmatizer import lemmatize_text
from shipper.schema import build_create_table_sql

# Поля, що переносяться зі старого рядка як є (усе, крім content, який не stored,
# і embedding, який окремо парситься з рядка у список float).
_PRESERVED_FIELDS = (
    "mongo_id", "source_type", "channel_name", "url", "collected_at",
    "title", "language", "summary",
    "verified_cve_ids", "verified_ttp_ids",
    "cve_count_verified", "ttp_count_verified", "ioc_count",
)


def parse_embedding(value):
    """Manticore віддає FLOAT_VECTOR рядком «f1,f2,...». Для /bulk JSON треба
    список float. Приймає також готовий список / порожнє / None."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    s = str(value).strip()
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip() != ""]


def build_migrated_doc(old_row: dict, original_text) -> dict:
    """Чиста трансформація: старий рядок індексу + оригінальний текст із Mongo
    -> doc для нової схеми. content ЗАВЖДИ відновлюється з original_text і
    лематизується (старе значення content, якщо є, ігнорується). Збережені поля
    переносяться без перерахунку; embedding парситься у список float."""
    doc = {k: old_row[k] for k in _PRESERVED_FIELDS if k in old_row}
    doc["embedding"] = parse_embedding(old_row.get("embedding"))
    doc["content"] = lemmatize_text(original_text or "")
    return doc


# --------------------------------------------------------------------------- #
# Нижче — I/O-обвязка (main). Юніт-тестами не покривається: перевіряється
# dry-run'ом на сервері перед реальним запуском.
# --------------------------------------------------------------------------- #
def _manticore_sql(base_url, sql, expect_data=False):
    import json
    import urllib.request

    req = urllib.request.Request(
        base_url.rstrip("/") + "/sql?mode=raw",
        data=sql.encode("utf-8"),
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )
    resp = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
    obj = resp[0] if isinstance(resp, list) else resp
    if obj.get("error"):
        raise RuntimeError("Manticore SQL error: %s" % obj["error"])
    return obj.get("data", []) if expect_data else obj


def _manticore_bulk(base_url, ndjson_body):
    import urllib.request

    req = urllib.request.Request(
        base_url.rstrip("/") + "/bulk",
        data=ndjson_body.encode("utf-8"),
        headers={"Content-Type": "application/x-ndjson"},
    )
    urllib.request.urlopen(req).read()


def _fetch_original_texts(mongo_ids):
    """{mongo_id: text} з Mongo одним $in-запитом (дзеркалить ui.fetch_original_texts)."""
    import pymongo
    from bson import ObjectId

    uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27018/?directConnection=true")
    db = os.getenv("MONGO_DB", "threat_intel")
    coll = os.getenv("MONGO_COLLECTION", "raw_events")

    object_ids = []
    for mid in mongo_ids:
        if not mid:
            continue
        try:
            object_ids.append(ObjectId(mid))
        except Exception:
            continue
    if not object_ids:
        return {}
    col = pymongo.MongoClient(uri)[db][coll]
    cur = col.find({"_id": {"$in": object_ids}}, {"text": 1})
    return {str(d["_id"]): d.get("text", "") for d in cur}


def main(argv=None):
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Cheap Manticore reindex (lemmatize + new schema).")
    ap.add_argument("--yes", action="store_true", help="реально DROP+перебудувати (без цього — dry-run)")
    ap.add_argument("--batch", type=int, default=200, help="розмір мікро-батча REPLACE")
    ap.add_argument("--limit", type=int, default=0, help="обмежити к-сть рядків (для тесту), 0 = усі")
    args = ap.parse_args(argv)

    base = os.getenv("MANTICORE_HTTP_URL", "http://127.0.0.1:9998")
    table = os.getenv("MANTICORE_TABLE", "events_enriched")

    cols = "id, " + ", ".join(_PRESERVED_FIELDS) + ", embedding"
    limit = args.limit or 1_000_000
    sql = (f"SELECT {cols} FROM {table} "
           f"LIMIT {limit} OPTION max_matches={limit}")
    rows = _manticore_sql(base, sql, expect_data=True)
    print(f"[read] {len(rows)} рядків зі старого індексу '{table}'")

    texts = _fetch_original_texts([r.get("mongo_id") for r in rows])
    have_text = sum(1 for r in rows if texts.get(r.get("mongo_id")))
    print(f"[mongo] знайдено оригінали для {have_text}/{len(rows)} документів")

    # Скільки реально стане лематизованим (кирилиця) — груба діагностика.
    sample = next((texts.get(r.get("mongo_id")) for r in rows if texts.get(r.get("mongo_id"))), "")
    if sample:
        print(f"[sample] '{sample[:60]}...' -> '{lemmatize_text(sample)[:60]}...'")

    if not args.yes:
        print("\n=== DRY-RUN === (нічого не змінено). Для реальної перебудови додай --yes")
        return

    # СТРАХОВКА перед DROP. embedding існує ЛИШЕ в Manticore — у Mongo лежить
    # тільки сирий текст. Якщо процес помре між DROP і вставкою, вектори
    # доведеться рахувати заново через Ollama (години GPU). Тому спершу
    # скидаємо вичитані рядки на диск, і лише потім чіпаємо таблицю.
    import time

    backup_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", f"reindex_backup_{table}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
    )
    backup_path = os.path.abspath(backup_path)
    with open(backup_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[backup] {len(rows)} рядків (з embedding) збережено: {backup_path}")

    print(f"[schema] DROP + CREATE '{table}' з новою схемою (morphology + min_infix)")
    _manticore_sql(base, f"DROP TABLE IF EXISTS {table}")
    _manticore_sql(base, build_create_table_sql(table))

    inserted = 0
    lines = []
    for r in rows:
        doc = build_migrated_doc(r, texts.get(r.get("mongo_id"), ""))
        lines.append(json.dumps({"replace": {"table": table, "id": r["id"], "doc": doc}}, default=str))
        if len(lines) >= args.batch:
            _manticore_bulk(base, "\n".join(lines) + "\n")
            inserted += len(lines)
            print(f"[insert] {inserted}/{len(rows)}")
            lines = []
    if lines:
        _manticore_bulk(base, "\n".join(lines) + "\n")
        inserted += len(lines)

    # Звірка: мовчазна часткова вставка — найгірший результат (індекс виглядає
    # робочим, але частини корпусу нема). Порівнюємо з тим, що реально лежить.
    final = _manticore_sql(base, f"SELECT COUNT(*) AS c FROM {table}", expect_data=True)
    in_index = final[0].get("c", final[0].get("count(*)")) if final else 0
    print(f"[done] надіслано {inserted}, у індексі {in_index}, було {len(rows)}")
    if int(in_index) != len(rows):
        print(f"[WARN] РОЗБІЖНІСТЬ: очікували {len(rows)}, маємо {in_index}. "
              f"Дані для відновлення: {backup_path}")
        return 1
    print(f"[ok] усі {len(rows)} документів на місці. Бекап можна видалити: {backup_path}")
    return 0


if __name__ == "__main__":
    main()
