#!/usr/bin/env python3
"""
Ретроактивна чистка Manticore від НЕЗАЗЕМЛЕНИХ міток CVE/MITRE.

НАВІЩО. Shipper тепер має гейт заземлення (`shipper/grounding.py`): мітка
проходить, лише якщо ідентифікатор є в тексті дослівно або модель навела
цитату, яка справді міститься в документі. Але записи, залиті ДО гейта,
лишилися з вигаданими мітками. Виміряно на бойовому корпусі 07.09.2026:
379 із 1495 подій отримали техніку, унікальних технік лише 5, T1190 — 375
міток із 380, і ЖОДНА техніка не трапляється в тексті дослівно. Верифікація
NVD/MITRE це пропустила, бо підтверджує лише ІСНУВАННЯ ідентифікатора.

КРИТЕРІЙ ТУТ СУВОРІШИЙ, НІЖ У ПРОДІ. Для історичних рядків цитат моделі не
збережено — їх тоді навіть не просили. Тож заднім числом працює лише дослівне
правило (`Grounding.LITERAL`): ідентифікатор має бути в тексті документа.
Це не «перестрахування», а єдиний критерій, який взагалі можна перевірити.

ЧОМУ REPLACE, а не UPDATE. Таблиця `events_enriched` — `engine='columnar'`;
Manticore не дозволяє UPDATE колонкового атрибута. Тому кожен зачеплений
документ перезаписуємо цілком через /bulk replace. Поле `content` —
indexed-без-stored (у Manticore його нема), тож текст тягнемо з Mongo
(джерело правди) по mongo_id; `embedding` дістається SELECT-ом і кладеться
назад БЕЗ переембеддингу (та сама модель — дрейфу нема).

Запуск (на сервері, host-python, лише stdlib):
    python3 tools/purge_ungrounded_ttps.py            # DRY-RUN (нічого не пише)
    python3 tools/purge_ungrounded_ttps.py --apply    # реально перезаписує

Опції: --manticore-url, --table, --mongo-container, --mongo-db, --ttps-only.
Споріднений інструмент — tools/purge_revoked_ttps.py (чистить відкликані
MITRE-техніки; там критерій — існування, тут — доречність).
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shipper.grounding import harvest_literal_ids, id_is_literal  # noqa: E402
from tools.manticore_io import (  # noqa: E402
    build_scan_query,
    bulk_replace,
    env_value,
    fetch_texts,
    mano_sql,
    mongo_uri_from_env,
    parse_embedding,
    rows as _rows,
    valid_object_ids,
)


# =====================================================================
#  Чисте ядро (покрите tests/test_purge_ungrounded.py)
# =====================================================================
def split_by_grounding(id_csv, text):
    """Розбиває CSV ідентифікаторів на (kept, removed) за дослівною згадкою.

    Нормалізує регістр і пробіли, зберігає порядок. Порожній вхід -> ([], [])."""
    if not id_csv:
        return [], []
    kept, removed = [], []
    for raw in str(id_csv).split(","):
        ident = raw.strip().upper()
        if not ident:
            continue
        (kept if id_is_literal(ident, text or "") else removed).append(ident)
    return kept, removed


def merge_harvested(kept, harvested, confirms):
    """Дописує до `kept` дослівно знайдені ідентифікатори, існування яких
    підтверджує `confirms(ident) -> bool`.

    Порядок: спершу те, що вже було, потім дібране. Повертає (merged, added)."""
    merged = list(kept)
    added = []
    for ident in harvested or []:
        if ident in merged or not confirms(ident):
            continue
        merged.append(ident)
        added.append(ident)
    return merged, added


def nvd_id_exists(payload):
    """Чи підтвердив NVD існування запису. Нас цікавить лише факт наявності —
    опис і CVSS тягне бойовий верифікатор, тут вони не потрібні."""
    if not isinstance(payload, dict):
        return False
    try:
        return int(payload.get("totalResults") or 0) > 0 and bool(payload.get("vulnerabilities"))
    except (TypeError, ValueError):
        return False


def build_replace_doc(row, content, embedding, kept_cves, kept_ttps):
    """Збирає doc під Manticore /bulk replace: усі поля як були, крім
    вичищених списків CVE/TTP і перерахованих під них лічильників."""
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
        "verified_cve_ids": ",".join(kept_cves),
        "verified_ttp_ids": ",".join(kept_ttps),
        "cve_count_verified": len(kept_cves),
        "ttp_count_verified": len(kept_ttps),
        "ioc_count": int(row.get("ioc_count", 0) or 0),
    }


# =====================================================================
#  I/O (виконується на сервері; не покрите юніт-тестами)
# =====================================================================
def make_nvd_confirmer(api_key, base_url, pause=0.7):
    """Повертає confirms(cve_id) з памʼяттю на вже питане.

    Ключ NVD піднімає ліміт до 50 запитів/30 с; без ключа — 5, тож пауза між
    викликами обовʼязкова. Мережеву помилку трактуємо як «не підтверджено»:
    краще не дописати справжній CVE, ніж дописати неперевірений."""
    cache = {}

    def confirms(cve_id):
        if cve_id in cache:
            return cache[cve_id]
        url = base_url + "?" + urllib.parse.urlencode({"cveId": cve_id})
        req = urllib.request.Request(url, headers=
                                     {"apiKey": api_key} if api_key else {})
        ok = False
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                ok = nvd_id_exists(json.loads(r.read().decode()))
        except Exception as e:
            print(f"    [WARN] NVD {cve_id}: {e} — не підтверджую.")
        time.sleep(pause)
        cache[cve_id] = ok
        return ok

    return confirms


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Чистка незаземлених міток CVE/MITRE з Manticore")
    ap.add_argument("--manticore-url", default="http://127.0.0.1:9998")
    ap.add_argument("--table", default="events_enriched")
    ap.add_argument("--mongo-uri", default="",
                    help="за замовчуванням COLLECTOR_MONGO_URI із .env")
    ap.add_argument("--mongo-db", default="threat_intel")
    ap.add_argument("--mongo-collection", default="raw_events")
    ap.add_argument("--ttps-only", action="store_true",
                    help="чистити лише техніки, CVE не чіпати")
    ap.add_argument("--no-harvest", action="store_true",
                    help="лише чистити; не добирати дослівні ID з тексту")
    ap.add_argument("--names", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mitre_names.json"),
                    help="набір валідних технік для добору")
    ap.add_argument("--nvd-base-url",
                    default="https://services.nvd.nist.gov/rest/json/cves/2.0")
    ap.add_argument("--apply", action="store_true",
                    help="реально перезаписати (без прапорця - DRY-RUN)")
    args = ap.parse_args(argv)

    cols = ("id, mongo_id, source_type, channel_name, url, collected_at, title, "
            "language, summary, verified_cve_ids, cve_count_verified, ioc_count, "
            "verified_ttp_ids, ttp_count_verified, embedding")
    harvest = not args.no_harvest
    res = mano_sql(args.manticore_url,
                    build_scan_query(cols, args.table,
                                     where="" if harvest else
                                     "verified_ttp_ids!='' OR verified_cve_ids!=''"))
    rows = _rows(res)
    print(f"[i] рядків на перегляд: {len(rows)}"
          f"{'' if harvest else ' (лише з мітками)'}")

    mongo_uri = args.mongo_uri or mongo_uri_from_env()
    if not mongo_uri:
        print("[!] немає COLLECTOR_MONGO_URI у .env і не задано --mongo-uri — "
              "Mongo під --auth, читати нічим.", file=sys.stderr)
        return 2
    texts = fetch_texts(mongo_uri, args.mongo_db, args.mongo_collection,
                        [r.get("mongo_id") for r in rows])
    print(f"[i] дістав текстів із Mongo: {len(texts)}")

    confirm_ttp = confirm_cve = (lambda _ident: False)
    if harvest:
        valid_ttps = set(json.load(open(args.names, encoding="utf-8")).keys())
        confirm_ttp = valid_ttps.__contains__
        confirm_cve = make_nvd_confirmer(env_value("NVD_API_KEY"), args.nvd_base_url)
        print(f"[i] добір увімкнено (валідних технік у наборі: {len(valid_ttps)})")

    affected, unreadable = [], 0
    for row in rows:
        # Текст живе в Mongo (content у Manticore не stored) — без нього
        # заземлення перевірити нічим, тож рядок не чіпаємо взагалі.
        text = texts.get(row.get("mongo_id"))
        if not text:
            unreadable += 1
            continue

        kept_ttps, removed_ttps = split_by_grounding(row.get("verified_ttp_ids"), text)
        if args.ttps_only:
            kept_cves = [c.strip().upper() for c in
                         str(row.get("verified_cve_ids") or "").split(",") if c.strip()]
            removed_cves = []
        else:
            kept_cves, removed_cves = split_by_grounding(row.get("verified_cve_ids"), text)

        added_cves = added_ttps = []
        if harvest:
            h_cves, h_ttps = harvest_literal_ids(text)
            kept_ttps, added_ttps = merge_harvested(kept_ttps, h_ttps, confirm_ttp)
            if not args.ttps_only:
                kept_cves, added_cves = merge_harvested(kept_cves, h_cves, confirm_cve)

        if removed_ttps or removed_cves or added_ttps or added_cves:
            affected.append((row, text, kept_cves, kept_ttps,
                             removed_cves, removed_ttps, added_cves, added_ttps))

    if unreadable:
        print(f"[!] рядків без доступного тексту: {unreadable} (не змінюються)")

    if not affected:
        print("[OK] незаземлених міток не знайдено — чистити нічого.")
        return 0

    dropped_ttps = sum(len(a[5]) for a in affected)
    dropped_cves = sum(len(a[4]) for a in affected)
    added_ttps_n = sum(len(a[7]) for a in affected)
    added_cves_n = sum(len(a[6]) for a in affected)
    print(f"[!] зачеплено рядків: {len(affected)} "
          f"| викидаємо технік: {dropped_ttps}, CVE: {dropped_cves}"
          f" | добираємо технік: {added_ttps_n}, CVE: {added_cves_n}")
    for row, _t, kept_cves, kept_ttps, rm_cves, rm_ttps, ad_cves, ad_ttps in affected[:50]:
        print(f"    id={row['id']} mongo_id={row['mongo_id']}"
              f" | техніки -{rm_ttps or '-'} +{ad_ttps or '-'} = {kept_ttps or '-'}"
              f" | CVE -{rm_cves or '-'} +{ad_cves or '-'} = {kept_cves or '-'}")
    if len(affected) > 50:
        print(f"    ... і ще {len(affected) - 50} рядків")

    if not args.apply:
        print("\n[DRY-RUN] нічого не змінено. Повторіть із --apply, щоб перезаписати.")
        return 0

    print("\n[APPLY] перезаписую...")
    for row, text, kept_cves, kept_ttps, *_rest in affected:
        embedding = parse_embedding(row.get("embedding"))
        if len(embedding) != 768:
            print(f"    [WARN] id={row['id']}: embedding {len(embedding)} != 768 — пропускаю "
                  "(REPLACE без валідного вектора зламає KNN).")
            continue
        doc = build_replace_doc(row, text, embedding, kept_cves, kept_ttps)
        out = bulk_replace(args.manticore_url, args.table, row["id"], doc)
        errs = out.get("errors") if isinstance(out, dict) else None
        print(f"    id={row['id']}: replace ok (errors={errs})")

    res2 = mano_sql(
        args.manticore_url,
        f"SELECT count(*) c FROM {args.table} WHERE verified_ttp_ids!=''")
    print(f"[i] рядків із технікою після чистки: {_rows(res2)[0]['c']}")
    print("[OK] готово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
