#!/usr/bin/env python3
"""
Переобчислення векторів ВСЬОГО індексу Manticore під нову схему ембедингів.

НАВІЩО. Шиппер раніше ембедив СИРИЙ оригінал посту. Виміряно (1486 подій,
корпус 832 uk / 174 ru / 3 en, англоцентричний nomic): 15 довгих постів
втрачено (Ollama 400), 462 відкинуто як хибні «семантичні дублікати», а KNN
для англійських запитів марний. Тепер вектор рахується з англійського
«title. summary» (`embedding_text.py`, одне джерело істини з шиппером і UI).
Вектори вже проіндексованих документів лишились старими, а змішувати два
простори в одному індексі не можна, тож шиппер на старті відмовляється писати
в індекс без маркера нової схеми (`shipper/embedding_scheme.py`). Цей
інструмент і є «лікуванням»: перераховує вектор кожного документа й, лише
якщо ВСЕ вдалось, пише маркер.

ЯК. Для кожного рядка індексу: title/summary беремо з Manticore (stored),
сирий текст — з Mongo (джерело правди; `content` у Manticore indexed-без-stored,
а REPLACE потребує документа ЦІЛКОМ, тож без тексту рядок не чіпаємо взагалі).
Новий вектор = Ollama `/api/embed` від `embedding_text(title, summary, текст)`,
L2-нормалізований рівно так, як це робить шиппер (`EmbeddingClient.embed`).
Документ збирається так само, як у `purge_ungrounded_ttps.py` (жодного
атрибута не загублено), але `content` — ЛЕМАТИЗОВАНИЙ, як його пише шиппер
(`manticore_sink._to_manticore_doc`): REPLACE із сирим текстом тихо зіпсував би
пошук. Потрібен venv із pymorphy3 (UI-venv) і pymongo.

БЕЗПЕКА.
  * Без `--apply` — DRY-RUN: нічого не пишемо, лише рахуємо й пробуємо Ollama.
  * `--apply` відмовляється працювати, поки шиппер тримає оренду
    (`shipper_state/main_pipeline`: є `lease_owner` і `lease_expires_at` у
    майбутньому) — інакше шиппер дописував би вектори іншої схеми в індекс,
    що саме переїжджає. Обійти — `--force` (на свій ризик).
  * Маркер пишемо тільки після повного успіху (жодного збою, жодного
    осиротілого рядка, кількість документів в індексі зійшлась). Збій = код
    виходу 1 і БЕЗ маркера. Збій REPLACE одного документа (HTTP/таймаут) не
    рве прохід: рахується як `failed`, решта переїжджає.
  * ОСИРОТІЛІ рядки (у Mongo немає тексту або mongo_id не ObjectId) переробити
    неможливо: REPLACE затер би `content`. Лишаючись зі старим вектором, вони
    блокують маркер, тож їх лічимо окремо (`orphans`, не `failed`) й друкуємо
    ПОВНИМ списком. `--drop-orphans` (разом з `--apply`) видаляє їх з Manticore:
    вони й так зламані (UI тягне текст із Mongo за mongo_id), а Mongo — джерело
    правди. Без прапорця вони й далі блокують маркер.
  * Модель за замовчуванням — OLLAMA_EMBED_MODEL із .env (як у шиппера), інакше
    маркер ніколи не збігся б. Друкуємо обидві схеми: яку запише інструмент і яку
    очікує шиппер. Якщо EMBED_PROVIDER шиппера — remote/auto, `--apply`
    відхиляється: інструмент ембедить лише через Ollama й такої схеми не дасть.
  * Ідемпотентно: повторний запуск перераховує й заміщає ті самі документи.

Запуск (з UI-venv):
    python tools/reembed_index.py                 # DRY-RUN
    python tools/reembed_index.py --apply         # реально перезаписує

Коди виходу: 0 — готово; 1 — були збої/осиротілі рядки (маркера немає); 2 — немає
доступу до Mongo; 3 — шиппер тримає оренду; 4 — шиппер ембедить не через Ollama
(remote/auto), `--apply` неможливий.
"""
import argparse
import json
import math
import os
import sys
import urllib.request
from types import SimpleNamespace
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_text import embedding_text  # noqa: E402
from lemmatizer import lemmatize_text  # noqa: E402
# Модуль на верхньому рівні — лише stdlib: pydantic/motor сюди не тягнуться.
from shipper.embedding_scheme import (  # noqa: E402
    MARKER_ID,
    current_scheme,
    marker_fields,
    scheme_for,
)
from tools.manticore_io import (  # noqa: E402
    build_delete_query,
    build_scan_query,
    bulk_replace,
    env_value,
    fetch_texts,
    mano_sql,
    mongo_uri_from_env,
    rows as _rows,
)

#: Колекція й документ оренди шиппера (дзеркало shipper/mongo_source.py; збіг
#: перевіряє tests/test_reembed_index.py — тут не імпортуємо, щоб не тягнути
#: motor/pydantic у host-python).
STATE_COLLECTION = "shipper_state"
LEASE_DOC_ID = "main_pipeline"

#: Розмірність KNN в індексі (shipper/schema.py: KNN_DIMS='768').
EMBED_DIMS = 768

#: Скільки рядків видаляти одним DELETE ... WHERE id IN (...).
DELETE_CHUNK = 500

#: Причина пропуску рядка, що робить його ОСИРОТІЛИМ (див. docstring модуля).
REASON_NO_TEXT = "немає тексту в Mongo"

#: Усі stored-поля, крім embedding: старий вектор не потрібен (ми його заміщуємо).
SCAN_COLS = ("id, mongo_id, source_type, channel_name, url, collected_at, title, "
             "language, summary, verified_cve_ids, cve_count_verified, ioc_count, "
             "verified_ttp_ids, ttp_count_verified")


# =====================================================================
#  Чисте ядро (покрите tests/test_reembed_index.py)
# =====================================================================
def embed_text_for(row, source_text):
    """Що ембедити для рядка індексу: «title. summary» із Manticore, а коли
    summary — плейсхолдер чи порожній, то сирий текст із Mongo (обрізаний).
    Правило — те саме `embedding_text`, що в шиппері."""
    return embedding_text(row.get("title"), row.get("summary"), source_text)


def normalize_vector(values):
    """L2-нормалізація, як у `EmbeddingClient.embed` (`if norm > 0`)."""
    values = [float(x) for x in values]
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm > 0 else values


def valid_vector(vec, dims=EMBED_DIMS):
    """REPLACE без валідного вектора зламає KNN: потрібні рівно `dims` скінченних
    чисел."""
    return len(vec) == dims and all(math.isfinite(x) for x in vec)


def chunked(seq, size):
    """Послідовність -> списки по `size` зі збереженням порядку."""
    size = max(1, int(size))
    return [list(seq[i:i + size]) for i in range(0, len(seq), size)]


def embed_in_batches(embed_fn, texts, batch):
    """Вектори для всіх `texts` (того ж порядку); `None` там, де не вдалось.

    Пакет ембедиться одним запитом. Якщо пакет упав або повернув не стільки
    векторів, скільки текстів (напр. ОДИН текст перевищив контекст і Ollama
    відповіла 400 на весь запит), пакет перераховується поштучно — тоді
    зіпсований текст стає одним `None`, а не 32."""
    out = []
    for part in chunked(texts, batch):
        try:
            vectors = embed_fn(part)
            if len(vectors) != len(part):
                raise ValueError(f"отримано {len(vectors)} векторів на {len(part)} текстів")
            out.extend(vectors)
            continue
        except Exception:
            pass
        for text in part:
            try:
                single = embed_fn([text])
                out.append(single[0] if len(single) == 1 else None)
            except Exception:
                out.append(None)
    return out


def build_reembedded_doc(row, source_text, vector):
    """Документ під /bulk replace: усі поля як були (той самий набір, що в
    `purge_ungrounded_ttps.build_replace_doc`), крім нового вектора й `content`.

    `content` — ЛЕМАТИЗОВАНИЙ оригінал, як його пише шиппер."""
    return {
        "mongo_id": row.get("mongo_id", ""),
        "source_type": row.get("source_type", "") or "",
        "channel_name": row.get("channel_name", "") or "",
        "url": row.get("url", "") or "",
        "collected_at": int(row.get("collected_at", 0) or 0),
        "title": row.get("title", "") or "",
        "language": row.get("language", "") or "",
        "content": lemmatize_text(source_text or ""),
        "summary": row.get("summary", "") or "",
        "embedding": vector,
        "verified_cve_ids": row.get("verified_cve_ids", "") or "",
        "verified_ttp_ids": row.get("verified_ttp_ids", "") or "",
        "cve_count_verified": int(row.get("cve_count_verified", 0) or 0),
        "ttp_count_verified": int(row.get("ttp_count_verified", 0) or 0),
        "ioc_count": int(row.get("ioc_count", 0) or 0),
    }


def plan_items(rows, texts):
    """Рядки -> (items, skipped).

    items — [(row, source_text, embed_text)] для рядків, які можна переробити;
    skipped — [(row, причина)]: без тексту з Mongo (REPLACE затер би `content`)
    або без чого ембедити."""
    items, skipped = [], []
    for row in rows:
        source_text = texts.get(row.get("mongo_id"))
        if not source_text or not str(source_text).strip():
            skipped.append((row, REASON_NO_TEXT))
            continue
        text = embed_text_for(row, source_text)
        if not text:
            skipped.append((row, "нема що ембедити"))
            continue
        items.append((row, source_text, text))
    return items, skipped


def lease_is_active(state_doc, now=None):
    """Чи тримає шиппер оренду: є `lease_owner` і `lease_expires_at` у майбутньому
    (поля й логіка — `MongoSource.acquire_lease`)."""
    if not state_doc or not state_doc.get("lease_owner"):
        return False
    expires = state_doc.get("lease_expires_at")
    if not isinstance(expires, datetime):
        return False
    if expires.tzinfo is None:   # pymongo без tz_aware віддає naive UTC
        expires = expires.replace(tzinfo=timezone.utc)
    return expires > (now or datetime.now(timezone.utc))


def lease_block_reason(state_doc, force, now=None):
    """Текст відмови, якщо `--apply` зараз небезпечний; `None` — можна."""
    if force or not lease_is_active(state_doc, now):
        return None
    return (f"шиппер живий: оренда {state_doc.get('lease_owner')} діє до "
            f"{state_doc.get('lease_expires_at')}. Зупиніть шиппер "
            "(docker compose stop shipper) і повторіть, або --force на свій ризик.")


def should_write_marker(failed, scanned, in_index, orphans=0, dropped=0):
    """Маркер — лише після ПОВНОГО успіху: жодного збою, жодного осиротілого рядка
    (`orphans` — ті, що ЛИШИЛИСЬ в індексі зі старим вектором), у індексі
    лишилась ненульова кількість рядків і рівно стільки, скільки мало лишитись
    (переглянуто мінус видалено осиротілих: REPLACE не мав ні розмножити, ні
    загубити рядки). Тоді кожен рядок, що лишився, — переембедений."""
    remaining = scanned - dropped
    return failed == 0 and orphans == 0 and remaining > 0 and in_index == remaining


def _lookup(key, env, path):
    """Значення налаштування шиппера: `env` (dict, у тестах) або процесне
    оточення, а потім .env — той самий пріоритет, що й у pydantic-settings."""
    if env is not None:
        return env.get(key, "") or ""
    return os.environ.get(key) or env_value(key, path)


def shipper_view(env=None, path=".env"):
    """Налаштування ембедингів шиппера, прочитані з оточення/.env, у вигляді, який
    приймає `current_scheme` (embed_provider / remote_embed_model /
    ollama_embed_model). Імпортувати `shipper.config` (pydantic) з інструмента
    важко, тож читаємо ті самі змінні напряму; відповідність
    `current_scheme(Settings)` перевіряє tests/test_reembed_index.py."""
    return SimpleNamespace(
        embed_provider=(_lookup("EMBED_PROVIDER", env, path) or "ollama").strip().lower(),
        remote_embed_model=(_lookup("REMOTE_EMBED_MODEL", env, path)
                            or "Qwen/Qwen3-Embedding-0.6B").strip(),
        ollama_embed_model=(_lookup("OLLAMA_EMBED_MODEL", env, path)
                            or "nomic-embed-text").strip(),
    )


def build_parser(env=None):
    ap = argparse.ArgumentParser(
        description="Переобчислення векторів індексу Manticore під схему «title. summary»")
    ap.add_argument("--manticore-url", default="http://127.0.0.1:9998")
    ap.add_argument("--table", default="events_enriched")
    ap.add_argument("--mongo-uri", default="",
                    help="за замовчуванням COLLECTOR_MONGO_URI із .env")
    ap.add_argument("--mongo-db", default="threat_intel")
    ap.add_argument("--mongo-collection", default="raw_events")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--embed-model", default=shipper_view(env).ollama_embed_model,
                    help="за замовчуванням OLLAMA_EMBED_MODEL із .env (як у шиппера, "
                         "інакше «nomic-embed-text»): за цим іменем пишеться маркер схеми")
    ap.add_argument("--batch", type=int, default=32,
                    help="скільки текстів ембедити одним запитом до Ollama")
    ap.add_argument("--apply", action="store_true",
                    help="реально перезаписати (без прапорця - DRY-RUN)")
    ap.add_argument("--drop-orphans", action="store_true",
                    help="разом з --apply: видалити з Manticore рядки без тексту в Mongo "
                         "(осиротілі; вони блокують маркер)")
    ap.add_argument("--force", action="store_true",
                    help="не зважати на оренду шиппера (на свій ризик)")
    return ap


# =====================================================================
#  Потік (I/O ін'єктується — тест ганяє його на фейку)
# =====================================================================
def report_schemes(args, shipper, out):
    """Друкує схему, яку запише інструмент, і схему, яку очікує шиппер; попереджає
    про розбіжність. Повертає False, якщо шиппер ембедить не через Ollama
    (remote/auto) — такої схеми інструмент дати не може."""
    tool_scheme = scheme_for(args.embed_model)
    shipper_scheme = current_scheme(shipper)
    out(f"[i] схема, яку запише інструмент: {tool_scheme}")
    out(f"[i] схема, яку очікує шиппер:     {shipper_scheme}")
    if shipper.embed_provider != "ollama":
        out(f"[WARN] EMBED_PROVIDER шиппера = {shipper.embed_provider}: інструмент ембедить "
            "лише через Ollama й такої схеми не дасть, тож --apply буде відхилено. "
            "Переведіть шиппер на EMBED_PROVIDER=ollama або переіндексуйте іншим шляхом.")
        return False
    if tool_scheme != shipper_scheme:
        out("[WARN] схема інструмента не збігається зі схемою шиппера: після --apply шиппер "
            f"усе одно відмовиться стартувати. Задайте --embed-model {shipper.ollama_embed_model}.")
    return True


def execute(args, io, out=print, shipper=None):
    """Весь прохід. `io` — об'єкт із scan_rows / fetch_texts / lease_state /
    embed / replace / delete / count / write_marker (див. LiveIO). `shipper` —
    налаштування ембедингів шиппера (`shipper_view`; за замовчуванням з
    оточення/.env). Повертає код виходу."""
    shipper = shipper if shipper is not None else shipper_view()
    ollama_only = report_schemes(args, shipper, out)
    if args.apply and not ollama_only:
        out(f"[!] відмова: EMBED_PROVIDER шиппера = {shipper.embed_provider}, а інструмент "
            "ембедить лише через Ollama — маркер потрібної схеми він не запише.")
        return 4

    rows = io.scan_rows()
    out(f"[i] рядків в індексі: {len(rows)}")
    if not rows:
        out("[OK] індекс порожній — переробляти нічого (шиппер сам поставить "
            "маркер на старті).")
        return 0

    texts = io.fetch_texts([r.get("mongo_id") for r in rows])
    items, skipped = plan_items(rows, texts)
    orphans = [row for row, reason in skipped if reason == REASON_NO_TEXT]
    out(f"[i] текстів із Mongo: {len(texts)}; до переобчислення: {len(items)}; "
        f"осиротілих (немає тексту в Mongo): {len(orphans)}; "
        f"інших пропущених: {len(skipped) - len(orphans)}")
    for row, reason in skipped:   # повний список: оператору треба знати, що саме зникне
        out(f"    [WARN] id={row.get('id')} mongo_id={row.get('mongo_id')}: {reason} — не чіпаю")
    if orphans:
        out(f"[i] осиротілі рядки лишаються зі СТАРИМИ векторами й блокують маркер. "
            f"`--apply --drop-orphans` видалить їх із Manticore ({len(orphans)} шт.; "
            "джерело правди — Mongo, а UI тягне звідти текст за mongo_id).")

    if not args.apply:
        for row, _src, text in items[:5]:
            out(f"    id={row.get('id')}: {text[:100]!r}")
        if items:
            try:
                dims = len(io.embed([items[0][2]])[0])
                out(f"[i] Ollama відповідає, розмірність вектора: {dims}")
            except Exception as exc:
                out(f"[WARN] Ollama недоступна ({exc}) — --apply не пройде.")
        out("\n[DRY-RUN] нічого не змінено. Повторіть із --apply, щоб перезаписати.")
        return 0

    reason = lease_block_reason(io.lease_state(), args.force)
    if reason:
        out(f"[!] відмова: {reason}")
        return 3

    out(f"\n[APPLY] ембедимо {len(items)} текстів (пакет {args.batch})...")
    vectors = embed_in_batches(io.embed, [text for _r, _s, text in items], args.batch)

    # Оренду звіряємо ще раз просто перед записом: ембединг міг тривати хвилини.
    reason = lease_block_reason(io.lease_state(), args.force)
    if reason:
        out(f"[!] відмова перед записом: {reason}")
        return 3

    reembedded = 0
    failed = len(skipped) - len(orphans)   # осиротілі — окремо, вони не «збій»
    for (row, source_text, _text), vector in zip(items, vectors):
        if vector is None:
            failed += 1
            out(f"    [FAIL] id={row.get('id')}: ембединг не вдався")
            continue
        vector = normalize_vector(vector)
        if not valid_vector(vector):
            failed += 1
            out(f"    [FAIL] id={row.get('id')}: вектор невалідний "
                f"({len(vector)} != {EMBED_DIMS} або нескінченні значення) — не пишу")
            continue
        try:
            replaced = io.replace(row["id"], build_reembedded_doc(row, source_text, vector))
        except Exception as exc:   # HTTPError/URLError/timeout одного документа не рве прохід
            failed += 1
            out(f"    [FAIL] id={row.get('id')}: replace впав: {exc!r}")
            continue
        if not replaced:
            failed += 1
            out(f"    [FAIL] id={row.get('id')}: replace не вдався")
            continue
        reembedded += 1

    # Осиротілих переробити неможливо (REPLACE затер би content), тож єдиний вихід
    # із блокування маркера — видалити їх. Лише за явним --drop-orphans.
    dropped = failed_deletes = 0
    if args.drop_orphans and orphans:
        for chunk in chunked([row["id"] for row in orphans], DELETE_CHUNK):
            try:
                io.delete(chunk)
            except Exception as exc:
                failed += len(chunk)
                failed_deletes += len(chunk)
                out(f"    [FAIL] видалення {len(chunk)} осиротілих рядків не вдалось: {exc!r}")
                continue
            dropped += len(chunk)
        out(f"[i] видалено осиротілих рядків: {dropped} із {len(orphans)}")
    orphans_left = len(orphans) - dropped - failed_deletes   # лишились зі старими векторами

    in_index = io.count()
    out(f"[i] scanned={len(rows)} re-embedded={reembedded} failed={failed} "
        f"orphans={orphans_left} dropped={dropped} | документів в індексі: {in_index}")
    if should_write_marker(failed, len(rows), in_index, orphans_left, dropped):
        scheme = scheme_for(args.embed_model)
        io.write_marker(scheme, in_index)
        out(f"[OK] готово. Маркер схеми записано: {scheme}")
        return 0
    if len(rows) - dropped == 0 and failed == 0:
        out("[OK] усі рядки були осиротілими й видалені: індекс порожній — маркер не "
            "потрібен, шиппер сам поставить його на старті.")
        return 0
    if orphans_left:
        command = "python tools/reembed_index.py --apply --drop-orphans"
        if args.embed_model != "nomic-embed-text":
            command += f" --embed-model {args.embed_model}"
        out(f"[!] маркер НЕ записано: в індексі лишилось {orphans_left} осиротілих рядків "
            "(немає тексту в Mongo / mongo_id не ObjectId) зі старими векторами. Щоб "
            f"видалити їх з Manticore й дописати маркер, повторіть: {command} "
            "(або поверніть їхні тексти в Mongo).")
    out("[!] маркер НЕ записано: були збої, осиротілі рядки або кількість документів не "
        "зійшлась. Виправте причину й повторіть запуск (він ідемпотентний).")
    return 1


# =====================================================================
#  I/O (виконується на сервері; юніт-тестами не покрите)
# =====================================================================
class LiveIO:
    """Справжні Manticore / Mongo / Ollama."""

    def __init__(self, args, mongo_uri):
        self.args = args
        self.mongo_uri = mongo_uri

    def scan_rows(self):
        return _rows(mano_sql(self.args.manticore_url,
                              build_scan_query(SCAN_COLS, self.args.table)))

    def fetch_texts(self, mongo_ids):
        return fetch_texts(self.mongo_uri, self.args.mongo_db,
                           self.args.mongo_collection, mongo_ids)

    def _state(self):
        from pymongo import MongoClient  # noqa: PLC0415 — залежність лише цього шляху
        client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=15000)
        return client, client[self.args.mongo_db][STATE_COLLECTION]

    def lease_state(self):
        client, state = self._state()
        try:
            return state.find_one({"_id": LEASE_DOC_ID})
        finally:
            client.close()

    def embed(self, texts):
        body = json.dumps({"model": self.args.embed_model, "input": texts}).encode()
        req = urllib.request.Request(
            self.args.ollama_url.rstrip("/") + "/api/embed", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode())["embeddings"]

    def replace(self, doc_id, doc):
        out = bulk_replace(self.args.manticore_url, self.args.table, doc_id, doc)
        return not (isinstance(out, dict) and out.get("errors"))

    def delete(self, ids):
        """DELETE за id (виняток при помилці Manticore — ловить `execute`)."""
        mano_sql(self.args.manticore_url, build_delete_query(self.args.table, ids))

    def count(self):
        res =mano_sql(self.args.manticore_url, f"SELECT count(*) c FROM {self.args.table}")
        return int(_rows(res)[0]["c"])

    def write_marker(self, scheme, documents):
        client, state = self._state()
        try:
            state.update_one({"_id": MARKER_ID},
                             {"$set": marker_fields(scheme, documents)}, upsert=True)
        finally:
            client.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    mongo_uri = args.mongo_uri or mongo_uri_from_env()
    if not mongo_uri:
        print("[!] немає COLLECTOR_MONGO_URI у .env і не задано --mongo-uri — "
              "Mongo під --auth, читати нічим.", file=sys.stderr)
        return 2
    try:
        lemmatize_text("перевірка")   # pymorphy3 має бути, інакше content зіпсується
    except Exception as exc:
        print(f"[!] лематизація недоступна ({exc}). Запускайте з UI-venv, де є pymorphy3.",
              file=sys.stderr)
        return 2
    return execute(args, LiveIO(args, mongo_uri))


if __name__ == "__main__":
    sys.exit(main())
