"""
Контракт екстрактора: модель зобовʼязана ЦИТУВАТИ, а не вгадувати.

ПЕРШОПРИЧИНА виродженого T1190 (375 із 380 міток на бойовому корпусі) — сам
промпт: у прикладі відповіді стояли конкретні значення
    "extracted_cve_ids": ["CVE-2026-3055"], "mitre_tactics_techniques": ["T1190"]
і модель просто копіювала приклад у кожен документ. Тому тут є регресійний
тест: у шаблоні промпта не має бути жодного конкретного ідентифікатора, який
можна списати.

Друга частина контракту — розбір відповіді: кожен ідентифікатор може прийти
з цитатою (`quote`), і ця цитата має доїхати до `ExtractedEntities`, бо саме
на ній стоїть заземлення (`shipper/grounding.py`).

Запуск:
    python tests/test_extractor_contract.py
або pytest.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shipper.extractor import (  # noqa: E402
    EXTRACTION_PROMPT_PREFIX,
    OllamaExtractor,
    build_extraction_prompt,
)

parse = OllamaExtractor._to_pydantic


# --- промпт не має чого списувати -----------------------------------------
def test_prompt_contains_no_concrete_technique_id():
    assert not re.search(r"\bT\d{4}\b", EXTRACTION_PROMPT_PREFIX), (
        "конкретний T-ідентифікатор у промпті модель копіює як дефолт"
    )


def test_prompt_contains_no_concrete_cve_id():
    assert not re.search(r"CVE-\d{4}-\d{4,}", EXTRACTION_PROMPT_PREFIX)


def test_prompt_demands_a_quote_from_the_text():
    assert "quote" in EXTRACTION_PROMPT_PREFIX.lower()


def test_prompt_allows_returning_nothing():
    low = EXTRACTION_PROMPT_PREFIX.lower()
    assert "empty" in low or "omit" in low


def test_untrusted_text_still_wrapped():
    prompt = build_extraction_prompt("шкідливий текст")
    assert "шкідливий текст" in prompt
    assert prompt.startswith(EXTRACTION_PROMPT_PREFIX)


# --- розбір відповіді: об'єктна форма з цитатами ---------------------------
def test_parses_objects_with_quotes():
    got = parse({
        "title": "Mail gateway exploited",
        "summary": "s",
        "detected_language": "uk",
        "extracted_cve_ids": [{"id": "cve-2026-3055", "quote": "діра в шлюзі"}],
        "mitre_tactics_techniques": [{"id": "t1190", "quote": "публічний вебінтерфейс"}],
    })
    assert got.extracted_cve_ids == ["CVE-2026-3055"]
    assert got.mitre_tactics_techniques == ["T1190"]
    assert got.cve_contexts == {"CVE-2026-3055": "діра в шлюзі"}
    assert got.ttp_contexts == {"T1190": "публічний вебінтерфейс"}


def test_parses_legacy_flat_strings_without_contexts():
    got = parse({
        "extracted_cve_ids": ["CVE-2026-3055"],
        "mitre_tactics_techniques": ["T1190"],
    })
    assert got.extracted_cve_ids == ["CVE-2026-3055"]
    assert got.mitre_tactics_techniques == ["T1190"]
    assert got.cve_contexts == {}
    assert got.ttp_contexts == {}


def test_parses_mixed_shapes():
    got = parse({"mitre_tactics_techniques": [
        "T1566",
        {"id": "T1190", "quote": "через вебінтерфейс"},
    ]})
    assert got.mitre_tactics_techniques == ["T1566", "T1190"]
    assert got.ttp_contexts == {"T1190": "через вебінтерфейс"}


def test_accepts_claimed_context_as_quote_alias():
    got = parse({"mitre_tactics_techniques": [
        {"technique_id": "T1190", "claimed_context": "через вебінтерфейс"},
    ]})
    assert got.mitre_tactics_techniques == ["T1190"]
    assert got.ttp_contexts == {"T1190": "через вебінтерфейс"}


def test_drops_items_without_identifier():
    got = parse({"mitre_tactics_techniques": [{"quote": "щось було"}, None, 42]})
    assert got.mitre_tactics_techniques == []
    assert got.ttp_contexts == {}


def test_missing_arrays_default_to_empty():
    got = parse({"title": "t"})
    assert got.extracted_cve_ids == []
    assert got.mitre_tactics_techniques == []
    assert got.cve_contexts == {}


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
