"""
Тести заземлення (grounding) міток CVE/MITRE.

ПРОБЛЕМА, яку це закриває. `shipper/extractor.py` віддає LLM лише сирий текст,
а модель сама вирішує, які ідентифікатори навісити. На бойовому корпусі це
виродилось у T1190 на 375 із 380 міток, причому ЖОДНА техніка не трапляється
в тексті поста дослівно. Верифікація (NVD/MITRE) підтверджує лише ІСНУВАННЯ
ідентифікатора, тож шум проходив її наскрізь.

ПРАВИЛО. Мітка приймається, лише якщо:
  * LITERAL — ідентифікатор є в тексті дослівно, або
  * QUOTED  — модель навела фрагмент (`claimed_context`), який реально
              міститься в документі.
Інакше UNSUPPORTED — відкидаємо ще ДО звернення до NVD/MITRE.

Запуск:
    python tests/test_grounding.py
або pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shipper.grounding import (  # noqa: E402
    ACCEPT_LITERAL,
    ACCEPT_QUOTED,
    MAX_CONTEXT_CHARS,
    MIN_CONTEXT_CHARS,
    Grounding,
    context_is_quoted,
    filter_grounded,
    id_is_literal,
    grounding_of,
)


# --- id_is_literal: CVE ---------------------------------------------------
def test_cve_present_verbatim_is_literal():
    assert id_is_literal("CVE-2026-3055", "Нова діра CVE-2026-3055 у Telegram")


def test_cve_matches_regardless_of_case():
    assert id_is_literal("CVE-2026-3055", "cve-2026-3055 exploited in the wild")


def test_cve_matches_across_unicode_dashes():
    # Телеграм-редактори люблять підміняти дефіс на en-dash.
    assert id_is_literal("CVE-2026-3055", "уразливість CVE‑2026–3055 закрито")


def test_cve_absent_from_text_is_not_literal():
    assert not id_is_literal("CVE-2026-3055", "Бот першої допомоги для Telegram")


# --- id_is_literal: MITRE -------------------------------------------------
def test_technique_present_verbatim_is_literal():
    assert id_is_literal("T1190", "Атака йшла через T1190 (публічний вебсервіс)")


def test_technique_not_matched_inside_longer_number():
    # T11901 не є згадкою T1190.
    assert not id_is_literal("T1190", "ідентифікатор T11901 у їхньому трекері")


def test_parent_technique_is_literal_when_subtechnique_mentioned():
    # Якщо в тексті T1078.004, то батьківська T1078 справді згадана.
    assert id_is_literal("T1078", "використали T1078.004 — хмарні акаунти")


def test_subtechnique_not_literal_when_only_parent_mentioned():
    # Зворотне НЕ діє: T1078 у тексті не підтверджує саме .004.
    assert not id_is_literal("T1078.004", "використали T1078 — валідні акаунти")


def test_technique_absent_from_text_is_not_literal():
    assert not id_is_literal("T1190", "Добірка сервісів для генерації логотипів")


def test_identifier_glued_to_letters_is_a_different_token():
    # «T1190bis» — інший токен, а не згадка T1190. Межа має різати не лише цифру.
    assert not id_is_literal("T1190", "внутрішній тег T1190bis у їхньому трекері")
    assert not id_is_literal("CVE-2019-0708", "мітка CVE-2019-0708rce у звіті")


# --- context_is_quoted ----------------------------------------------------
TEXT = ("Дослідники повідомили про активну експлуатацію вразливості "
        "в публічному вебінтерфейсі поштового шлюзу компанії.")


def test_context_found_in_text_is_quoted():
    assert context_is_quoted("активну експлуатацію вразливості", TEXT)


def test_context_ignores_whitespace_and_case_differences():
    assert context_is_quoted("Активну   Експлуатацію\nВразливості", TEXT)


def test_context_absent_from_text_is_not_quoted():
    assert not context_is_quoted("виконання коду в ядрі Windows", TEXT)


def test_missing_context_is_not_quoted():
    assert not context_is_quoted(None, TEXT)
    assert not context_is_quoted("", TEXT)


def test_too_short_context_is_rejected():
    # «шлюзу» є в тексті, але як доказ це ніщо — вимагаємо змістовний фрагмент.
    short = "шлюзу"
    assert len(short) < MIN_CONTEXT_CHARS
    assert not context_is_quoted(short, TEXT)


def test_whole_document_dump_is_not_a_citation():
    # Модель, що замість цитати віддає весь документ, доказу не наводить.
    long_text = "а" * (MAX_CONTEXT_CHARS + 50)
    assert not context_is_quoted(long_text, long_text)


# --- grounding_of ---------------------------------------------------------
def test_literal_wins_over_quote():
    assert grounding_of("T1190", None, "через T1190 зайшли") is Grounding.LITERAL


def test_quote_grounds_identifier_absent_from_text():
    g = grounding_of("T1190", "активну експлуатацію вразливості", TEXT)
    assert g is Grounding.QUOTED


def test_no_id_and_no_valid_quote_is_unsupported():
    g = grounding_of("T1190", "виконання коду в ядрі Windows", TEXT)
    assert g is Grounding.UNSUPPORTED


# --- filter_grounded ------------------------------------------------------
def test_filter_keeps_grounded_drops_the_rest():
    kept, dropped = filter_grounded(
        ["CVE-2026-3055", "T1190", "T1566"],
        {"T1190": "активну експлуатацію вразливості"},
        "Ось CVE-2026-3055. " + TEXT,
    )
    assert kept == ["CVE-2026-3055", "T1190"]
    assert dropped == [("T1566", Grounding.UNSUPPORTED)]


def test_filter_in_literal_only_mode_drops_quoted():
    kept, dropped = filter_grounded(
        ["T1190"],
        {"T1190": "активну експлуатацію вразливості"},
        TEXT,
        accept=ACCEPT_LITERAL,
    )
    assert kept == []
    assert dropped == [("T1190", Grounding.QUOTED)]


def test_filter_deduplicates_preserving_order():
    kept, _ = filter_grounded(["T1190", "t1190"], {}, "через T1190 зайшли")
    assert kept == ["T1190"]


def test_filter_on_empty_input_returns_empty():
    assert filter_grounded([], {}, TEXT) == ([], [])


# --- ВІДОМА СЛАБКІСТЬ режиму "quoted" ------------------------------------
# Ці тести не описують бажану поведінку — вони ЗАКРІПЛЮЮТЬ межу, за якою
# цитата перестає бути доказом. Виміряно 07.09.2026 на бойовому корпусі
# (hermes3:8b, A/B старого й нового промпта): новий промпт прибирає вироджений
# T1190, але модель починає чіпляти інші техніки, підкріплюючи їх СПРАВЖНІМИ
# цитатами не по темі (T1056 «Input Capture» на пост про пошук відео в Google).
# Тому дефолт у проді — "literal"; "quoted" лишається як свідомо слабший режим.

def test_one_quote_can_ground_several_unrelated_ids_in_quoted_mode():
    quote = "активну експлуатацію вразливості"
    kept, _ = filter_grounded(["T1190", "T1566"],
                              {"T1190": quote, "T1566": quote},
                              TEXT, accept=ACCEPT_QUOTED)
    assert kept == ["T1190", "T1566"], "цитата не привʼязана до конкретної мітки"


def test_literal_mode_rejects_that_same_pair():
    kept, dropped = filter_grounded(["T1190", "T1566"],
                                    {"T1190": TEXT[:40], "T1566": TEXT[:40]},
                                    TEXT, accept=ACCEPT_LITERAL)
    assert kept == []
    assert [g for _, g in dropped] == [Grounding.QUOTED, Grounding.QUOTED]


def test_short_post_lets_almost_the_whole_document_pass_as_a_quote():
    # Типовий телеграм-пост коротший за MAX_CONTEXT_CHARS, тож «цитатою» може
    # стати практично весь текст — доказова сила такої цитати нульова.
    short = "Знайшли дірку в поштовому шлюзі, вже експлуатують."
    assert len(short) < MAX_CONTEXT_CHARS
    kept, _ = filter_grounded(["T1190"], {"T1190": short}, short, accept=ACCEPT_QUOTED)
    assert kept == ["T1190"]


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
