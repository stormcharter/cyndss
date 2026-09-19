"""
Юніт-тести спільного I/O-ядра інструментів чистки (`tools/manticore_io.py`).

НАВІЩО ОКРЕМИЙ МОДУЛЬ. Два проходи по індексу — `purge_revoked_ttps.py`
(відкликані техніки MITRE) і `purge_ungrounded_ttps.py` (незаземлені мітки) —
робили ту саму чорну роботу двома копіями. У копіях розʼїхались виправлення:
незаземлений прохід уже вміє те, чого відкликаний не вміє, і мовчки на цьому
падає. Тому все, що стосується розмови з Manticore й Mongo, живе тут в одному
екземплярі, і кожна з трьох грабель закрита в ОДНОМУ місці:

  1. SELECT у Manticore мовчки віддає щонайбільше 1000 рядків, хоч би який був
     LIMIT, поки не задано `OPTION max_matches` -> `build_scan_query`.
  2. id документа зберігається як unsigned 64-bit, а SELECT віддає signed: усе,
     що більше за 2^63, приїжджає відʼємним, і /bulk на такий id відповідає
     400 Bad Request -> `build_bulk_line` (не call-site, щоб забути було ніде).
  3. Mongo під `--auth`, тож текст документа тягнеться через pymongo одним
     запитом, а не через `docker exec mongosh` без креденшелів -> `fetch_texts`
     (мережевої частини тут немає, тестується підготовка списку id).

Запуск:
    python tests/test_manticore_io.py
або pytest.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.manticore_io import (  # noqa: E402
    build_bulk_line,
    build_scan_query,
    is_valid_mongo_id,
    manticore_doc_id,
    valid_object_ids,
)


# --- 1. Стеля вибірки ------------------------------------------------------
def test_scan_query_lifts_the_default_match_ceiling():
    q = build_scan_query("id, mongo_id", "events_enriched", limit=100000)
    assert "LIMIT 100000" in q
    assert "OPTION max_matches=100000" in q


def test_scan_query_keeps_ceiling_in_step_with_limit():
    # Стеля нижча за LIMIT знову тихо обріже вибірку — вони мають збігатися.
    q = build_scan_query("id", "events_enriched", limit=50)
    assert "LIMIT 50 OPTION max_matches=50" in q


def test_scan_query_without_filter_scans_everything():
    q = build_scan_query("id", "events_enriched")
    assert "SELECT id FROM events_enriched" in q
    assert "WHERE" not in q


def test_scan_query_applies_the_given_filter():
    q = build_scan_query("id", "events_enriched", where="verified_ttp_ids!=''")
    assert "FROM events_enriched WHERE verified_ttp_ids!='' LIMIT" in q


# --- 2. Знаковий id --------------------------------------------------------
def test_positive_id_passes_through():
    assert manticore_doc_id(7) == 7
    assert manticore_doc_id("358095497603592183") == 358095497603592183


def test_negative_id_is_folded_back_into_unsigned():
    assert manticore_doc_id(-8871276304639429742) == 2**64 - 8871276304639429742


def test_folded_id_stays_in_range():
    assert 0 < manticore_doc_id(-1) < 2**64


def test_bulk_line_folds_the_id_itself():
    """Згортання живе в побудові рядка, а не в місці виклику.

    Інакше кожен новий прохід по індексу мусить памʼятати про це самотужки —
    саме так баг і пережив появу другого інструмента."""
    line = build_bulk_line("events_enriched", -8871276304639429742, {"mongo_id": "x"})
    payload = json.loads(line)
    assert payload["replace"]["id"] == 2**64 - 8871276304639429742


def test_bulk_line_carries_table_and_doc():
    payload = json.loads(build_bulk_line("events_enriched", 7, {"mongo_id": "abc"}))
    assert payload["replace"]["table"] == "events_enriched"
    assert payload["replace"]["doc"] == {"mongo_id": "abc"}


def test_bulk_line_is_a_single_ndjson_record():
    line = build_bulk_line("t", 1, {"a": 1})
    assert "\n" not in line


# --- 3. Підготовка списку id для одного запиту в Mongo ---------------------
def test_keeps_canonical_object_ids_in_order():
    ids = ["507f1f77bcf86cd799439011", "507f191e810c19729de860ea"]
    assert valid_object_ids(ids) == ids


def test_drops_non_canonical_ids():
    assert valid_object_ids(["507f1f77bcf86cd799439011", "legacy-string-id", "", None]) == [
        "507f1f77bcf86cd799439011"
    ]


def test_deduplicates_ids():
    assert valid_object_ids(["507f1f77bcf86cd799439011"] * 3) == ["507f1f77bcf86cd799439011"]


def test_empty_input_yields_empty_list():
    assert valid_object_ids([]) == []
    assert valid_object_ids(None) == []


def test_object_id_validator_accepts_canonical_hex():
    assert is_valid_mongo_id("507f1f77bcf86cd799439011")


def test_object_id_validator_rejects_everything_else():
    for bad in ("", "zzzf1f77bcf86cd799439011", "507f1f77bcf86cd79943901", 12345, None):
        assert not is_valid_mongo_id(bad), bad


# --- ручний запуск без pytest --------------------------------------------
if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
