#!/usr/bin/env python3
"""
Ретроактивна чистка Manticore від відкликаних (revoked) / deprecated технік
MITRE ATT&CK у полі `verified_ttp_ids`.

НАВІЩО. Shipper (verifier.py, коміт a8eb177) уже НЕ інжестить revoked/deprecated
техніки — але записи, залиті ДО фікса, лишилися з застарілими тегами (напр.
deprecated T1494). Ще важливіше: MITRE періодично депрекейтить техніки й НАДАЛІ,
тож потреба «пройтись мітлою» по вже проіндексованому корпусу повторюватиметься.

ЧОМУ REPLACE, а не UPDATE. Таблиця `events_enriched` — `engine='columnar'`;
Manticore не дозволяє UPDATE колонкового атрибута (`unable to update columnar
attribute`). Тому кожен зачеплений документ перезаписуємо цілком через /bulk
replace. Поле `content` — indexed-без-stored (у Manticore його нема), тож текст
тягнемо з Mongo (джерело правди) по mongo_id; `embedding` дістається SELECT-ом
і кладеться назад БЕЗ переембеддингу (та сама модель — дрейфу нема).

ВАЛІДНИЙ НАБІР технік = ключі mitre_names.json (він уже без revoked/deprecated).

Запуск (на сервері, З VENV — потрібен pymongo, бо Mongo під --auth):
    source .venv/bin/activate
    python tools/purge_revoked_ttps.py                  # DRY-RUN (нічого не пише)
    python tools/purge_revoked_ttps.py --apply          # реально перезаписує

Опції: --manticore-url, --table, --mongo-uri, --mongo-db, --mongo-collection, --names.

Розмова з Manticore й Mongo винесена в tools/manticore_io.py — там же закриті
три мовчазні граблі (стеля вибірки 1000 рядків, знаковий id, читання з Mongo
під автентифікацією). Раніше вони жили тут у власній копії й не були закриті.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.manticore_io import (  # noqa: E402
    build_scan_query,
    bulk_replace,
    fetch_texts,
    is_valid_mongo_id,
    mano_sql,
    mongo_uri_from_env,
    parse_embedding,
    rows as _rows,
)


# =====================================================================
#  Чисте ядро (покрите tests/test_purge_revoked_ttps.py)
# =====================================================================
def invalid_ttps(ttp_csv, valid_ids):
    """Розбиває CSV `verified_ttp_ids` на (kept, removed) відносно valid_ids.

    Нормалізує регістр і пробіли, зберігає порядок. valid_ids — множина
    валідних технік (напр. ключі mitre_names.json). Порожній/None → ([], [])."""
    if not ttp_csv:
        return [], []
    kept, removed = [], []
    for raw in str(ttp_csv).split(","):
        tid = raw.strip().upper()
        if not tid:
            continue
        (kept if tid in valid_ids else removed).append(tid)
    return kept, removed


def build_replace_doc(row, content, embedding, kept_ttps):
    """Збирає doc під Manticore /bulk replace зі збереженням усіх полів, крім
    вичищених TTP. `content` — з Mongo, `embedding` — назад як є, kept_ttps —
    список технік, що лишаються (лічильник перераховується)."""
    return {
        "mongo_id": row.get("mongo_id", ""),
        "source_type": row.get("source_type", "") or "",
        "channel_name": row.get("channel_name", "") or "",
        "url": row.get("url", "") or "",
        "collected_at": int(row.get("collected_at", 0) or 0),
        "title": row.get("title", "") or "",
        "language": row.get("language", "") or "",
        "content": content or "",
        "summary": row.get("summary", "") or "",
        "embedding": embedding,
        "verified_cve_ids": row.get("verified_cve_ids", "") or "",
        "verified_ttp_ids": ",".join(kept_ttps),
        "cve_count_verified": int(row.get("cve_count_verified", 0) or 0),
        "ttp_count_verified": len(kept_ttps),
        "ioc_count": int(row.get("ioc_count", 0) or 0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Чистка revoked/deprecated MITRE-технік з Manticore")
    ap.add_argument("--manticore-url", default="http://127.0.0.1:9998")
    ap.add_argument("--table", default="events_enriched")
    ap.add_argument("--mongo-uri", default="",
                    help="за замовчуванням COLLECTOR_MONGO_URI із .env")
    ap.add_argument("--mongo-db", default="threat_intel")
    ap.add_argument("--mongo-collection", default="raw_events")
    ap.add_argument("--names", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mitre_names.json"))
    ap.add_argument("--apply", action="store_true", help="реально перезаписати (без прапорця — DRY-RUN)")
    args = ap.parse_args(argv)

    valid_ids = set(json.load(open(args.names, encoding="utf-8")).keys())
    print(f"[i] валідних технік у наборі: {len(valid_ids)} (джерело: {os.path.basename(args.names)})")

    cols = ("id, mongo_id, source_type, channel_name, url, collected_at, title, "
            "language, summary, verified_cve_ids, cve_count_verified, ioc_count, "
            "verified_ttp_ids, ttp_count_verified, embedding")
    res = mano_sql(args.manticore_url,
                   build_scan_query(cols, args.table, where="verified_ttp_ids!=''"))
    scanned = _rows(res)
    print(f"[i] рядків із TTP: {len(scanned)}")

    affected = []
    for row in scanned:
        kept, removed = invalid_ttps(row.get("verified_ttp_ids"), valid_ids)
        if removed:
            affected.append((row, kept, removed))

    if not affected:
        print("[✓] відкликаних/deprecated технік у корпусі не знайдено — чистити нічого.")
        return 0

    print(f"[!] зачеплено рядків: {len(affected)}")
    for row, kept, removed in affected:
        print(f"    id={row['id']} mongo_id={row['mongo_id']} "
              f"| викинути: {removed} | лишити: {kept or '—'}")

    if not args.apply:
        print("\n[DRY-RUN] нічого не змінено. Повторіть із --apply, щоб перезаписати.")
        return 0

    # Текст живе в Mongo (content у Manticore indexed-без-stored) — тягнемо
    # одним запитом на всі рядки, а не по документу на підпроцес mongosh.
    mongo_uri = args.mongo_uri or mongo_uri_from_env()
    if not mongo_uri:
        print("[!] немає COLLECTOR_MONGO_URI у .env і не задано --mongo-uri — "
              "Mongo під --auth, читати нічим.", file=sys.stderr)
        return 2
    texts = fetch_texts(mongo_uri, args.mongo_db, args.mongo_collection,
                        [r.get("mongo_id") for r, _k, _r in affected])
    print(f"[i] дістав текстів із Mongo: {len(texts)}")

    print("\n[APPLY] перезаписую...")
    for row, kept, removed in affected:
        content = texts.get(row.get("mongo_id"))
        if not content:
            # Без тексту REPLACE стер би `content` — рядок не чіпаємо взагалі.
            print(f"    [WARN] id={row['id']}: текст недоступний — пропускаю.")
            continue
        embedding = parse_embedding(row.get("embedding"))
        if len(embedding) != 768:
            print(f"    [WARN] id={row['id']}: embedding {len(embedding)} != 768 — пропускаю "
                  "(REPLACE без валідного вектора зламає KNN).")
            continue
        doc = build_replace_doc(row, content, embedding, kept)
        out = bulk_replace(args.manticore_url, args.table, row["id"], doc)
        errs = out.get("errors") if isinstance(out, dict) else None
        print(f"    id={row['id']}: replace ok (errors={errs})")

    # Верифікація
    res2 = mano_sql(args.manticore_url,
                    f"SELECT count(*) c FROM {args.table} WHERE verified_ttp_ids!=''")
    print(f"[i] рядків із TTP після чистки: {_rows(res2)[0]['c']}")
    print("[✓] готово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
