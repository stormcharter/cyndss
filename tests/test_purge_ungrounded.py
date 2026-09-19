"""
Юніт-тести чистого ядра tools/purge_ungrounded_ttps.py — ретроактивної чистки
Manticore від міток, не заземлених у тексті документа.

КОНТЕКСТ. Записи, залиті ДО гейта заземлення, несуть мітки, яких у тексті
немає взагалі (на бойовому корпусі — 375 T1190 із 380, жодної дослівної).
Верифікація їх пропустила, бо підтверджує лише існування ідентифікатора.

ВАЖЛИВО ПРО КРИТЕРІЙ. Для історичних рядків доступний ЛИШЕ дослівний критерій:
цитат моделі (`claimed_context`) для них ніхто не зберігав — їх тоді навіть не
просили. Тому тут працює тільки `Grounding.LITERAL`, і це свідомо суворіше за
бойовий гейт: заднім числом мʼякший критерій просто нічим підтвердити.

Запуск:
    python tests/test_purge_ungrounded.py
або pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.purge_ungrounded_ttps import (  # noqa: E402
    build_replace_doc,
    merge_harvested,
    nvd_id_exists,
    split_by_grounding,
)

TEXT = ("Дослідники повідомили про активну експлуатацію CVE-2026-3055 "
        "у публічному вебінтерфейсі поштового шлюзу.")


# --- split_by_grounding ---------------------------------------------------
def test_keeps_identifier_present_in_text():
    kept, removed = split_by_grounding("CVE-2026-3055", TEXT)
    assert kept == ["CVE-2026-3055"]
    assert removed == []


def test_removes_identifier_absent_from_text():
    kept, removed = split_by_grounding("T1190", TEXT)
    assert kept == []
    assert removed == ["T1190"]


def test_splits_mixed_csv_preserving_order():
    kept, removed = split_by_grounding("T1190, CVE-2026-3055 ,T1566", TEXT)
    assert kept == ["CVE-2026-3055"]
    assert removed == ["T1190", "T1566"]


def test_normalizes_case_and_whitespace():
    kept, _ = split_by_grounding("  cve-2026-3055  ", TEXT)
    assert kept == ["CVE-2026-3055"]


def test_empty_csv_yields_nothing():
    assert split_by_grounding("", TEXT) == ([], [])
    assert split_by_grounding(None, TEXT) == ([], [])


def test_missing_text_removes_everything():
    # Немає тексту — немає й підстав лишати мітку.
    kept, removed = split_by_grounding("T1190", "")
    assert kept == []
    assert removed == ["T1190"]


# --- merge_harvested ------------------------------------------------------
# Прохід має не лише РІЗАТИ, а й ДОБИРАТИ: 16 статей із 45 містять CVE дослівно,
# але проіндексувались до того, як зʼявився символьний збір. Додаємо лише те,
# існування чого підтверджене (CVE — за NVD, техніка — за mitre_names.json).
def test_appends_confirmed_ids_after_the_existing_ones():
    merged, added = merge_harvested(["CVE-2026-3055"], ["CVE-2019-0708"],
                                    lambda i: True)
    assert merged == ["CVE-2026-3055", "CVE-2019-0708"]
    assert added == ["CVE-2019-0708"]


def test_does_not_duplicate_what_is_already_kept():
    merged, added = merge_harvested(["T1566"], ["T1566"], lambda i: True)
    assert merged == ["T1566"]
    assert added == []


def test_skips_ids_that_do_not_confirm():
    merged, added = merge_harvested([], ["CVE-2026-0001", "CVE-2026-0002"],
                                    lambda i: i.endswith("0002"))
    assert merged == ["CVE-2026-0002"]
    assert added == ["CVE-2026-0002"]


def test_nothing_harvested_changes_nothing():
    merged, added = merge_harvested(["T1566"], [], lambda i: True)
    assert merged == ["T1566"]
    assert added == []


# --- nvd_id_exists: розбір відповіді NVD ---------------------------------
def test_nvd_confirms_when_the_record_is_returned():
    assert nvd_id_exists({"totalResults": 1, "vulnerabilities": [{"cve": {}}]})


def test_nvd_denies_on_empty_result():
    assert not nvd_id_exists({"totalResults": 0, "vulnerabilities": []})


def test_nvd_denies_on_garbage():
    assert not nvd_id_exists(None)
    assert not nvd_id_exists({})


# --- build_replace_doc ----------------------------------------------------
ROW = {
    "id": 7,
    "mongo_id": "507f1f77bcf86cd799439011",
    "source_type": "telegram",
    "channel_name": "КІБЕРБЕЗ",
    "url": "https://t.me/x/1",
    "collected_at": 1750000000,
    "title": "Mail gateway exploited",
    "language": "uk",
    "summary": "s",
    "verified_cve_ids": "CVE-2026-3055",
    "verified_ttp_ids": "T1190,T1566",
    "cve_count_verified": 1,
    "ttp_count_verified": 2,
    "ioc_count": 3,
}


def test_replace_doc_rewrites_both_lists_and_counters():
    doc = build_replace_doc(ROW, TEXT, [0.1, 0.2], ["CVE-2026-3055"], [])
    assert doc["verified_cve_ids"] == "CVE-2026-3055"
    assert doc["cve_count_verified"] == 1
    assert doc["verified_ttp_ids"] == ""
    assert doc["ttp_count_verified"] == 0


def test_replace_doc_carries_text_from_mongo_and_embedding_as_is():
    # `content` — indexed-без-stored, у Manticore його нема: беремо з Mongo.
    # Ембеддинг кладемо назад без переобчислення — модель та сама, дрейфу нема.
    doc = build_replace_doc(ROW, TEXT, [0.1, 0.2], [], [])
    assert doc["content"] == TEXT
    assert doc["embedding"] == [0.1, 0.2]


def test_replace_doc_preserves_untouched_fields():
    doc = build_replace_doc(ROW, TEXT, [], [], [])
    assert doc["mongo_id"] == ROW["mongo_id"]
    assert doc["channel_name"] == "КІБЕРБЕЗ"
    assert doc["collected_at"] == 1750000000
    assert doc["title"] == "Mail gateway exploited"
    assert doc["ioc_count"] == 3


def test_replace_doc_does_not_leak_the_row_id_into_the_body():
    # id адресує документ у /bulk, всередині doc йому не місце.
    assert "id" not in build_replace_doc(ROW, TEXT, [], [], [])


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
