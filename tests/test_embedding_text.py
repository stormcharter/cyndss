"""
Юніт-тести embedding_text.py — єдине джерело істини «який текст ембедиться».

Модуль спільний для шиппера (вектор у індексі) і UI (`concept_support`), тому
тут же перевіряється, що обидва користувачі згодні між собою: якщо вони
розійдуться, вектори документів в індексі й вектори запитів лежатимуть у різних
поданнях, і KNN мовчки погіршає.

Запуск:
    pytest tests/test_embedding_text.py -v
або як звичайний скрипт (без pytest):
    python tests/test_embedding_text.py
"""

import os
import sys

# Дозволяємо запуск і як `pytest`, і як `python tests/...` з кореня репо.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_text import (  # noqa: E402
    EMBED_TEXT_CHARS,
    SUMMARY_PLACEHOLDER,
    TITLE_PLACEHOLDERS,
    embedding_text,
)


# --- константи -------------------------------------------------------------
def test_constants_are_the_measured_ones():
    # Найдовший прийнятий кириличний пост — 3305 символів, найкоротший
    # відхилений — 2994: 1500 лишає безпечний запас під контекст nomic.
    assert EMBED_TEXT_CHARS == 1500
    assert SUMMARY_PLACEHOLDER == "Summary unavailable"
    assert "Untitled" in TITLE_PLACEHOLDERS


# --- title + summary -------------------------------------------------------
def test_joins_title_and_summary_with_dot():
    assert embedding_text(" Alpha ", " beta ", "оригінал") == "Alpha. beta"


def test_summary_without_title_is_just_the_summary():
    assert embedding_text("", "beta") == "beta"
    assert embedding_text(None, "beta") == "beta"


def test_none_is_treated_as_empty():
    assert embedding_text(None, None, None) == ""


# --- плейсхолдери: не несуть змісту ----------------------------------------
def test_summary_placeholder_is_dropped_and_content_is_used():
    # Вироджений випадок, що давав хибні злиття: «Untitled. Summary unavailable».
    assert embedding_text("Untitled", SUMMARY_PLACEHOLDER, "справжній текст") == "справжній текст"


def test_title_placeholder_is_dropped_next_to_a_real_summary():
    assert embedding_text("Untitled", "Real summary") == "Real summary"


def test_placeholder_summary_with_real_title_and_content_uses_content():
    # Заголовок без реального summary — це не подання змісту; повний текст
    # (обрізаний) інформативніший за один заголовок.
    assert embedding_text("Real title", SUMMARY_PLACEHOLDER, "тіло посту") == "тіло посту"


# --- запасні варіанти ------------------------------------------------------
def test_falls_back_to_content_when_no_summary():
    assert embedding_text("", "", "  повний текст ") == "повний текст"
    assert embedding_text("Alpha", "  ", "повний текст") == "повний текст"


def test_falls_back_to_title_when_no_summary_and_no_content():
    assert embedding_text("Alpha", "", "") == "Alpha"
    assert embedding_text("Alpha", SUMMARY_PLACEHOLDER) == "Alpha"


def test_placeholder_title_alone_is_empty():
    assert embedding_text("Untitled", "", "") == ""
    assert embedding_text("Untitled", SUMMARY_PLACEHOLDER, "") == ""


# --- стеля -----------------------------------------------------------------
def test_result_is_capped_for_the_embedder():
    assert len(embedding_text("", "", "к" * 5000)) == EMBED_TEXT_CHARS == 1500
    assert len(embedding_text("T", "s" * 5000)) == EMBED_TEXT_CHARS


def test_cap_does_not_touch_model_written_text():
    # Модельні title+summary в індексі ≤ ~1000 символів — стеля їх не чіпає.
    summary = "s" * 900
    assert embedding_text("T", summary) == "T. " + summary


# --- згода з concept_support (UI) ------------------------------------------
def test_concept_support_document_text_agrees_with_embedding_text():
    from concept_support import DOCUMENT_CHARS, document_text

    assert DOCUMENT_CHARS == EMBED_TEXT_CHARS
    docs = [
        {"title": "Alpha", "summary": "beta", "content": "тіло"},
        {"title": "", "summary": "", "content": "лише тіло"},
        {"title": "Alpha", "summary": "", "content": "тіло"},
        {"title": "Untitled", "summary": SUMMARY_PLACEHOLDER, "content": "тіло"},
        {"title": "Untitled", "summary": "Real summary"},
        {"title": None, "summary": None, "content": None},
        {"title": "T", "summary": "s" * 5000},
        {},
    ]
    for d in docs:
        assert document_text(d) == embedding_text(
            d.get("title"), d.get("summary"), d.get("content")), d


# --- збій видобування сутностей ------------------------------------------------
def test_unparsed_extraction_embeds_the_post_not_the_stub():
    """Коли модель не віддала розбірний JSON, екстрактор ставить заголовок-заглушку
    «Unparsed Threat Event», а в summary кладе перші 200 символів ОРИГІНАЛУ —
    це не переказ. Вектор має рахуватись від самого посту (до стелі), а не від
    англійської константи з обрубком кирилиці."""
    post = "Пост про атаку на банк. " * 30
    assert embedding_text("Unparsed Threat Event", post[:200], post) == post[:1500].strip()


def test_unparsed_title_matches_the_shipper_default():
    """Константа мусить збігатися зі значенням за замовчуванням у моделі шиппера,
    інакше правило вище мовчки перестане спрацьовувати."""
    from shipper.models import ExtractedEntities
    assert ExtractedEntities().title == "Unparsed Threat Event"


# --- образ шиппера ----------------------------------------------------------
def test_dockerfile_ships_the_shared_module():
    """Шиппер імпортує `embedding_text` з кореня образу; без COPY контейнер
    впаде на старті з ImportError, а локальні тести цього не побачать."""
    dockerfile = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "Dockerfile")
    with open(dockerfile, encoding="utf-8") as fh:
        lines = [line.strip() for line in fh]
    assert "COPY embedding_text.py ." in lines


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001 — простий раннер без pytest
                failed += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failed else 0)
