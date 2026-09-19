"""
Юніт-тести для чистого ядра tools/purge_revoked_ttps.py — інструменту, що
вичищає з Manticore відкликані/deprecated MITRE-техніки, які потрапили в
`verified_ttp_ids` ДО того, як shipper почав їх фільтрувати (коміт a8eb177).

Запуск:
    python tests/test_purge_revoked_ttps.py
або pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.manticore_io import is_valid_mongo_id  # noqa: E402
from tools.purge_revoked_ttps import (  # noqa: E402
    build_replace_doc,
    invalid_ttps,
)


# --- invalid_ttps: розбити CSV на «лишити» / «викинути» --------------------
def test_drops_id_absent_from_valid_set():
    kept, removed = invalid_ttps("T1494", {"T1566", "T1059"})
    assert kept == []
    assert removed == ["T1494"]


def test_keeps_valid_drops_invalid_preserving_order():
    kept, removed = invalid_ttps("T1566, T1494, T1059", {"T1566", "T1059"})
    assert kept == ["T1566", "T1059"]
    assert removed == ["T1494"]


def test_normalizes_case_and_whitespace():
    kept, removed = invalid_ttps("  t1566 ,  t1494 ", {"T1566"})
    assert kept == ["T1566"]
    assert removed == ["T1494"]


def test_empty_input_yields_nothing():
    assert invalid_ttps("", {"T1566"}) == ([], [])
    assert invalid_ttps(None, {"T1566"}) == ([], [])


def test_all_valid_removes_none():
    kept, removed = invalid_ttps("T1566,T1059", {"T1566", "T1059"})
    assert kept == ["T1566", "T1059"]
    assert removed == []


# --- build_replace_doc: зібрати doc під Manticore /bulk replace ------------
def _row():
    return {
        "mongo_id": "6a3d18fa04f835f532542fb0",
        "source_type": "telegram",
        "channel_name": "КІБЕРБЕЗ",
        "url": "",
        "collected_at": 1680710508,
        "title": "Unexplained Anomalies in Videos",
        "language": "uk",
        "summary": "Two videos...",
        "verified_cve_ids": "",
        "cve_count_verified": 0,
        "ioc_count": 0,
    }


def test_build_replace_clears_ttp_when_none_kept():
    doc = build_replace_doc(_row(), content="повний текст", embedding=[0.1, 0.2],
                            kept_ttps=[])
    assert doc["verified_ttp_ids"] == ""
    assert doc["ttp_count_verified"] == 0


def test_build_replace_sets_kept_ttps_and_count():
    doc = build_replace_doc(_row(), content="t", embedding=[0.1],
                            kept_ttps=["T1566", "T1059"])
    assert doc["verified_ttp_ids"] == "T1566,T1059"
    assert doc["ttp_count_verified"] == 2


def test_build_replace_carries_over_other_fields_and_content_embedding():
    row = _row()
    emb = [0.1, 0.2, 0.3]
    doc = build_replace_doc(row, content="ПОВНИЙ ТЕКСТ", embedding=emb, kept_ttps=[])
    # незмінні поля переносяться
    assert doc["mongo_id"] == row["mongo_id"]
    assert doc["title"] == row["title"]
    assert doc["summary"] == row["summary"]
    assert doc["source_type"] == row["source_type"]
    assert doc["channel_name"] == row["channel_name"]
    assert doc["collected_at"] == row["collected_at"]
    assert doc["language"] == row["language"]
    assert doc["verified_cve_ids"] == ""
    assert doc["cve_count_verified"] == 0
    assert doc["ioc_count"] == 0
    # content (з Mongo) і embedding (з Manticore) кладуться назад
    assert doc["content"] == "ПОВНИЙ ТЕКСТ"
    assert doc["embedding"] == emb


# --- mongo_id -> JS: захист від ін'єкції в `mongosh --eval` -----------------
# Інструмент тягне текст із Mongo, підставляючи mongo_id у JS-рядок, який
# віддається mongosh через --eval. Раніше id клеївся форматуванням '%s' без
# перевірки — рядок на кшталт  abc') || (function(){...  виривався з літерала
# ObjectId(...) і виконувався як довільний JS у контексті БД (читання/видалення).
INJECTION_PAYLOADS = [
    "abc') || (function(){return db.dropDatabase()})(",
    "6a3d18fa04f835f532542fb0'); db.raw_events.drop(); ('",
    '6a3d18fa04f835f532542fb0"}); print(1); ({"a":"',
    "6a3d18fa04f835f532542fb" + chr(92) + "'",
    "'; while(true){}; '",
    "../../etc/passwd",
    "6a3d18fa04f835f532542fb0 ",     # хвостовий пробіл — уже не валідний ObjectId
    "6a3d18fa04f835f532542fb",       # 23 символи
    "6a3d18fa04f835f532542fb00",     # 25 символів
    "ZZZZ18fa04f835f532542fb0",      # не hex
    "",
    None,
    12345,
]


def test_valid_object_id_accepted():
    assert is_valid_mongo_id("6a3d18fa04f835f532542fb0") is True
    assert is_valid_mongo_id("6A3D18FA04F835F532542FB0") is True


def test_invalid_ids_rejected_by_validator():
    for bad in INJECTION_PAYLOADS:
        assert is_valid_mongo_id(bad) is False, repr(bad)


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
