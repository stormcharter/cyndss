"""Маркер схеми ембедингів індексу + відмова писати в непереїханий індекс.

ЧОМУ. Вектор документа залежить від ДВОХ речей: моделі ембедера й того, ЯКИЙ
ТЕКСТ ембедиться (`embedding_text.py`). Вектори з сирого оригіналу й вектори з
англійського «title. summary» лежать у різних просторах; в одному індексі вони
мовчки псують і KNN, і дедуп. Тому схему записано маркером у Mongo (колекція
`shipper_state`, документ `{_id: "index_embedding", scheme, documents,
recorded_at}`): шиппер на старті звіряє маркер зі своєю схемою й, якщо індекс
непорожній і маркер не збігся, відмовляється працювати. Лікує це
`python tools/reembed_index.py --apply` (він же пише маркер).

Модуль на верхньому рівні — лише stdlib: `tools/reembed_index.py` імпортує
`scheme_for` і `EMBEDDING_SCHEME_VERSION` без pydantic/motor. Налаштування
шиппера тягнуться лінивим імпортом у `current_scheme()`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from embedding_text import EMBED_TEXT_CHARS

#: Версія ПОДАННЯ тексту. Змінюється разом із правилом `embedding_text()`;
#: стеля береться з тієї ж константи, що й саме правило, тож змінити її мовчки
#: (без зміни схеми й переіндексації) неможливо.
EMBEDDING_SCHEME_VERSION = f"title.summary|content[:{EMBED_TEXT_CHARS}]/v1"

#: `_id` документа-маркера в колекції `shipper_state`.
MARKER_ID = "index_embedding"

#: Підказка оператору, як вилікувати розбіжність.
_REMEDY = ("Зупиніть шиппер і виконайте `python tools/reembed_index.py --apply` "
           "(перераховує вектори всього індексу й пише маркер), потім запустіть "
           "шиппер знову. Якщо модель шиппера (OLLAMA_EMBED_MODEL) не типова — додайте "
           "`--embed-model <та сама модель>`, інакше маркер не збігеться; якщо в "
           "індексі є рядки без тексту в Mongo (осиротілі) — вони блокують маркер, "
           "додайте `--drop-orphans`, щоб видалити їх з індексу.")


class IndexSchemeMismatch(RuntimeError):
    """Індекс містить вектори іншої схеми (або схема невідома) — писати в нього не можна."""


def scheme_for(model: str) -> str:
    """Рядок схеми для моделі ембедера: «<модель>:<версія подання тексту>»."""
    return f"{model}:{EMBEDDING_SCHEME_VERSION}"


def current_scheme(cfg=None) -> str:
    """Схема, за якою шиппер рахує вектори ЗАРАЗ.

    Модель визначається так само, як у `EmbeddingClient`: «ollama» ->
    `ollama_embed_model`, «remote» -> `remote_embed_model`. «auto» пробує remote,
    а за помилки мовчки переходить на Ollama ПІД ЧАС роботи — тобто в одному
    індексі можуть опинитись вектори обох моделей, і «активної» моделі немає.
    Тому для «auto» схема описує обидві (`auto[remote|ollama]`): маркер не
    збігається з жодним чистим режимом, і перехід на/з «auto» вимагає
    переіндексації (це правильно — простори різні)."""
    if cfg is None:
        from .config import settings as cfg  # noqa: PLC0415 — лінивий імпорт, див. docstring модуля
    provider = cfg.embed_provider.strip().lower()
    if provider == "remote":
        model = cfg.remote_embed_model
    elif provider == "auto":
        model = f"auto[{cfg.remote_embed_model}|{cfg.ollama_embed_model}]"
    else:
        model = cfg.ollama_embed_model
    return scheme_for(model)


async def ensure_index_scheme(state_collection, count_documents, scheme: str) -> None:
    """Перевірити, що індекс можна безпечно доповнювати векторами схеми `scheme`.

    * маркер збігається -> ок;
    * маркера нема/інший, а індекс ПОРОЖНІЙ -> пишемо маркер (свіжий індекс);
    * маркера нема/інший, а в індексі є документи -> `IndexSchemeMismatch`.

    `count_documents` — async-виклик без аргументів (кількість документів у
    Manticore); викликається лише коли маркер не збігся."""
    found = await state_collection.find_one({"_id": MARKER_ID})
    found_scheme = found.get("scheme") if isinstance(found, dict) else None
    if found_scheme == scheme:
        return

    documents = await count_documents()
    if documents == 0:
        await record_index_scheme(state_collection, scheme, 0)
        return

    seen = f"у маркері: «{found_scheme}»" if found_scheme else "маркера немає"
    raise IndexSchemeMismatch(
        f"Індекс непорожній ({documents} документів), а схема його ембедингів не "
        f"збігається з поточною: {seen}; очікується: «{scheme}». "
        "Нові вектори не можна змішувати зі старими (інший простір). " + _REMEDY
    )


def marker_fields(scheme: str, documents: int, now=None) -> dict:
    """Тіло маркера (без `_id`). Спільне для шиппера й tools/reembed_index.py,
    щоб формат не розійшовся."""
    return {
        "scheme": scheme,
        "documents": int(documents),
        "recorded_at": now or datetime.now(timezone.utc),
    }


async def record_index_scheme(state_collection, scheme: str, documents: int) -> None:
    """Записати (upsert) маркер: який схемою пораховано вектори індексу."""
    await state_collection.update_one(
        {"_id": MARKER_ID},
        {"$set": marker_fields(scheme, documents)},
        upsert=True,
    )


async def check_index_scheme(sink, state_collection, logger, scheme: str | None = None) -> bool:
    """Обгортка для точок входу (main, replay): `True` — можна працювати; `False` —
    індекс іншої схеми, причину вже записано в лог рівнем error, а викликач має
    завершитись із ненульовим кодом, НЕ обробляючи жодної події."""
    try:
        await ensure_index_scheme(
            state_collection, sink.count_documents, scheme or current_scheme())
    except IndexSchemeMismatch as exc:
        logger.error("index_scheme_mismatch", error=str(exc))
        return False
    return True
