"""
Юніт-тести для lemmatizer.py — лематизація української для пошуку.

МОТИВАЦІЯ. Manticore не має власного стемера/лематизатора для української, і
stem_ru українські флексії НЕ стягує (чергування голосних у корені: вразлив-
ІСТЬ / вразлив-ОСТІ). Тому «вразливість» не знаходило «вразливості».

РІШЕННЯ. Лематизуємо україномовні токени до нормальної форми ОДНАКОВО на двох
боках: на індексації (shipper кладе в content лему) і на запиті (ui лематизує
ввід). Тоді будь-яка словоформа з обох боків зводиться до спільної леми й
матчиться. Латиницю НЕ чіпаємо — англ. словоформи покриває Manticore stem_en.

Модуль НЕ імпортує streamlit / shipper.config — тестується ізольовано.
Потребує pymorphy3 + pymorphy3-dicts-uk (як networkx-тести — без залежності
просто не запуститься, це очікувано).

Запуск:
    pytest tests/test_lemmatizer.py -v
або як звичайний скрипт (без pytest):
    python tests/test_lemmatizer.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lemmatizer import lemmatize_text, lemmatize_query, MAX_QUERY_CHARS  # noqa: E402


# --- запит: обрізання ДО лематизації (захист від DoS) ------------------------
def test_query_is_truncated_before_lemmatizing():
    """Довгий ввід у пошуку не має ганятись через морфоаналізатор цілком.

    pymorphy3.parse() недешевий на КОЖНОМУ унікальному токені, а lru_cache
    рятує лише від повторів. Автентифікований користувач міг би вставити
    мегабайти унікальних кириличних псевдослів і навантажити CPU спільного
    боксу — обрізання в sanitize_match_query() відбувається вже ПІСЛЯ
    лематизації, тобто занадто пізно. Тому запит ріжемо ДО неї."""
    huge = "вразливості " * 50_000
    out = lemmatize_query(huge)
    assert len(out) <= MAX_QUERY_CHARS * 2, len(out)


def test_query_lemmatizes_like_text():
    """У межах ліміту поведінка та сама, що й у lemmatize_text."""
    assert lemmatize_query("вразливості") == lemmatize_text("вразливості")


def test_query_handles_empty():
    assert lemmatize_query("") == ""
    assert lemmatize_query(None) == ""


def test_documents_are_NOT_truncated():
    """КРИТИЧНО: обрізання стосується ЛИШЕ запиту. Тіла постів (shipper) мають
    лематизуватись цілком, інакше повнотекстовий індекс втратить хвіст тексту."""
    long_doc = "вразливості " * 5_000
    out = lemmatize_text(long_doc)
    assert len(out) > MAX_QUERY_CHARS * 10, len(out)


# --- головна вимога: усі флексії -> одна лема -------------------------------
def test_ukrainian_inflections_collapse_to_one_lemma():
    """Різні відмінкові форми зводяться до однакової леми (тому й матчаться)."""
    forms = ["вразливість", "вразливості", "вразливостей", "вразливістю"]
    lemmas = {lemmatize_text(f) for f in forms}
    assert len(lemmas) == 1, lemmas


def test_query_form_matches_indexed_form():
    """Запит у одній формі й текст у іншій зводяться до одного рядка."""
    assert lemmatize_text("вразливості") == lemmatize_text("вразливість")


def test_specific_lemma_value():
    """Лема множини — саме називний однини."""
    assert lemmatize_text("вразливості") == "вразливість"


# --- латиниця не чіпається (нею опікується Manticore stem_en) ---------------
def test_latin_tokens_pass_through_unchanged():
    """Англ. токени лишаються байт-у-байт — англ. морфологію стягує stem_en у
    Manticore, а не uk-лематизатор (він би їх зіпсував/злемматизував хибно)."""
    assert lemmatize_text("vulnerabilities") == "vulnerabilities"
    assert lemmatize_text("Cisco IOS XE") == "Cisco IOS XE"


def test_mixed_language_string():
    """Мішаний рядок: кирилицю лематизуємо, латиницю лишаємо як є."""
    out = lemmatize_text("нові вразливості Cisco")
    assert "вразливість" in out
    assert "Cisco" in out
    assert "вразливості" not in out


# --- стійкість ---------------------------------------------------------------
def test_empty_and_none_are_safe():
    assert lemmatize_text("") == ""
    assert lemmatize_text(None) == ""


def test_punctuation_is_preserved():
    """Пунктуація не має ламати лематизацію токена поруч із нею."""
    out = lemmatize_text("вразливості!")
    assert "вразливість" in out


def test_is_idempotent_on_a_lemma():
    """Повторна лематизація вже-леми нічого не змінює."""
    once = lemmatize_text("атаки")
    assert lemmatize_text(once) == once


if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS " + name)
            passed += 1
        except Exception:
            print("FAIL " + name)
            traceback.print_exc()
            failed += 1
    print("\n%d passed, %d failed" % (passed, failed))
