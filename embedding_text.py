"""Який текст ембедиться: єдине джерело істини для шиппера й UI.

Модуль спільний (як `prompt_guard.py` / `lemmatizer.py`): шиппер за ним рахує
вектор документа, що лягає в індекс, а `concept_support.document_text` — вектор
кандидата в UI. Якщо подання розійдуться, вектори документів і запитів лежатимуть
у різних просторах, і KNN мовчки погіршає.

ЧОМУ НЕ СИРИЙ ОРИГІНАЛ. Раніше шиппер ембедив сирий текст посту. Виміряно на
живому стеку (1486 подій, корпус 832 uk / 174 ru / 3 en; `nomic-embed-text`
англоцентричний):
  * 15 постів (>= 2994 символи кирилиці) ВТРАЧЕНО: Ollama відповідає HTTP 400
    «input length exceeds the context length» (і з truncate=true). Найдовший
    прийнятий пост — 3305 символів.
  * 462 пости відкинуто як семантичні дублікати (cosine >= 0.92), з них 451 у
    діапазоні 0.92-0.95; вибірка з 10 була 10/10 РІЗНИМИ постами. Навіть >= 0.95
    зливало різні інциденти. Справжніх дублів ~3 точні копії та кілька
    переказів — тобто ~30% корпусу мовчки пропадало.
  * Англійські запити проти українських оригінальних векторів: KNN марний.
Англійське «title. summary», яке пише модель, дає вектори, на яких працює й KNN,
і дедуп (хибних злиттів 16 із 1009 проти ~31%; майже всі — пости, на які модель
відповіла порожньою відпискою на кшталт «Untitled. No malicious activity
detected.»). Плейсхолдери, які пише сам шиппер, тут вважаються порожніми й у
подання не потрапляють.

СТЕЛЯ `EMBED_TEXT_CHARS`. Це захист від 400 на довгому тексті, а не обрізання
осмислених даних: модельні title+summary в індексі не довші за ~1000 символів.

Модуль чистий — без сторонніх імпортів (Dockerfile копіює його окремо).
"""

from __future__ import annotations

#: Стеля тексту, що йде в ембедер. Найдовший прийнятий Ollama кириличний пост —
#: 3305 символів, найкоротший відхилений — 2994; 1500 — безпечний запас.
#: Модельні title+summary в індексі ≤ ~1000 символів, тож стеля їх не чіпає.
EMBED_TEXT_CHARS = 1500

#: Значення, які шиппер пише в документ, коли модель не дала summary/title.
#: Змісту вони не несуть і мусять трактуватися як порожні.
SUMMARY_PLACEHOLDER = "Summary unavailable"
TITLE_PLACEHOLDERS = frozenset({"Untitled"})
#: Заголовок, який екстрактор шиппера ставить, коли модель не віддала розбірний
#: JSON (`shipper.models.ExtractedEntities`). У такому разі summary — це перші 200
#: символів ОРИГІНАЛУ, а не переказ, тож обидва поля порожні й ембедиться сам пост.
UNPARSED_TITLE = "Unparsed Threat Event"


def embedding_text(title, summary, content="") -> str:
    """Текст для ембедера: «title. summary»; без реального summary — `content`
    (сирий текст), а якщо й його нема — `title`. Результат ≤ `EMBED_TEXT_CHARS`.

    None трактується як порожній рядок; плейсхолдери — теж як порожні."""
    title = str(title or "").strip()
    summary = str(summary or "").strip()
    content = str(content or "").strip()
    if title == UNPARSED_TITLE:
        title = summary = ""
    if title in TITLE_PLACEHOLDERS:
        title = ""
    if summary == SUMMARY_PLACEHOLDER:
        summary = ""

    if summary:
        text = ". ".join(part for part in (title, summary) if part)
    elif content:
        text = content
    else:
        text = title
    return text[:EMBED_TEXT_CHARS]
