"""
Маркер схеми ембедингів індексу: шиппер не пише нові вектори в індекс, чиї
вектори пораховані інакше.

НАВІЩО. Вектори, пораховані з сирого оригіналу, і вектори з «title. summary»
лежать у РІЗНИХ просторах. Якщо шиппер додаватиме нові (англійські) вектори до
старого (сирого) індексу, KNN і дедуп мовчки змішають їх, і ніхто цього не
помітить. Тому схема записана маркером у Mongo (`shipper_state`), а шиппер на
старті звіряє його й відмовляється працювати з непереїханим індексом.

Усе на фейках — без мережі.

Запуск:
    python -m pytest tests/test_embedding_scheme.py
"""
import pytest

from shipper.config import Settings
from shipper.embedding_scheme import (
    EMBEDDING_SCHEME_VERSION,
    MARKER_ID,
    IndexSchemeMismatch,
    check_index_scheme,
    current_scheme,
    ensure_index_scheme,
    marker_fields,
    record_index_scheme,
    scheme_for,
)

SCHEME = "nomic-embed-text:title.summary|content[:1500]/v1"


class FakeState:
    """Мінімальний async-аналог колекції `shipper_state`."""

    def __init__(self, docs=None):
        self.docs = {d["_id"]: dict(d) for d in (docs or [])}
        self.writes = []

    async def find_one(self, query):
        doc = self.docs.get(query["_id"])
        return dict(doc) if doc is not None else None

    async def update_one(self, query, update, upsert=False):
        self.writes.append((query, update, upsert))
        doc = self.docs.get(query["_id"])
        if doc is None:
            if not upsert:
                return None
            doc = {"_id": query["_id"]}
            self.docs[query["_id"]] = doc
        doc.update(update.get("$set", {}))
        return None


def counter(value):
    calls = []

    async def count_documents():
        calls.append(1)
        return value

    count_documents.calls = calls
    return count_documents


def marker(scheme=SCHEME, documents=10):
    return {"_id": MARKER_ID, "scheme": scheme, "documents": documents}


# --- рядок схеми -----------------------------------------------------------
def test_scheme_version_is_the_agreed_constant():
    assert EMBEDDING_SCHEME_VERSION == "title.summary|content[:1500]/v1"
    assert MARKER_ID == "index_embedding"


def test_scheme_version_is_built_from_the_text_cap_constant():
    from embedding_text import EMBED_TEXT_CHARS

    assert f"content[:{EMBED_TEXT_CHARS}]" in EMBEDDING_SCHEME_VERSION
    assert EMBEDDING_SCHEME_VERSION == f"title.summary|content[:{EMBED_TEXT_CHARS}]/v1"


def test_changing_the_text_cap_changes_the_scheme(monkeypatch):
    # Стеля тексту не може змінитись мовчки: інша стеля = інший простір векторів,
    # а отже інша схема. Завантажуємо КОПІЮ модуля (під іншим іменем, без
    # перезавантаження справжнього — інакше зіпсуємо ідентичність класів в інших
    # тестах) з підміненою стелею.
    import importlib.util

    import embedding_text
    import shipper.embedding_scheme as real

    monkeypatch.setattr(embedding_text, "EMBED_TEXT_CHARS", 1000)
    spec = importlib.util.spec_from_file_location("_scheme_copy", real.__file__)
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)

    assert "content[:1000]" in copy.EMBEDDING_SCHEME_VERSION
    assert copy.EMBEDDING_SCHEME_VERSION != EMBEDDING_SCHEME_VERSION


def test_scheme_for_prefixes_the_model_name():
    assert scheme_for("nomic-embed-text") == SCHEME


def _cfg(**over):
    return Settings(_env_file=None, **over)


def test_current_scheme_for_ollama_uses_the_ollama_model():
    cfg = _cfg(embed_provider="ollama", ollama_embed_model="nomic-embed-text")
    assert current_scheme(cfg) == SCHEME


def test_current_scheme_for_remote_uses_the_remote_model():
    cfg = _cfg(embed_provider="remote", remote_embed_url="https://embed.example.com",
               remote_embed_model="Qwen/Qwen3-Embedding-0.6B")
    assert current_scheme(cfg) == f"Qwen/Qwen3-Embedding-0.6B:{EMBEDDING_SCHEME_VERSION}"


def test_current_scheme_for_auto_describes_both_possible_models():
    # «auto» може впасти з remote на Ollama під час роботи, тож схема описує обох.
    cfg = _cfg(embed_provider="auto", remote_embed_model="R", ollama_embed_model="O")
    got = current_scheme(cfg)
    assert "R" in got and "O" in got and got.endswith(EMBEDDING_SCHEME_VERSION)


# --- ensure_index_scheme ---------------------------------------------------
@pytest.mark.asyncio
async def test_matching_marker_passes_without_counting_or_writing():
    state = FakeState([marker()])
    count = counter(5)

    await ensure_index_scheme(state, count, SCHEME)

    assert state.writes == []
    assert count.calls == []


@pytest.mark.asyncio
async def test_empty_index_gets_the_marker_written():
    state = FakeState()

    await ensure_index_scheme(state, counter(0), SCHEME)

    assert state.docs[MARKER_ID]["scheme"] == SCHEME
    assert state.docs[MARKER_ID]["documents"] == 0


@pytest.mark.asyncio
async def test_non_empty_index_without_marker_is_refused():
    state = FakeState()

    with pytest.raises(IndexSchemeMismatch) as info:
        await ensure_index_scheme(state, counter(1009), SCHEME)

    assert MARKER_ID not in state.docs  # відмова нічого не пише
    assert "tools/reembed_index.py --apply" in str(info.value)
    assert "маркера немає" in str(info.value)


@pytest.mark.asyncio
async def test_refusal_remedy_mentions_embed_model_and_drop_orphans():
    # Оператор має знати обидва виходи: модель, що не збігається з .env, і
    # осиротілі рядки, які без --drop-orphans блокують маркер вічно.
    with pytest.raises(IndexSchemeMismatch) as info:
        await ensure_index_scheme(FakeState(), counter(5), SCHEME)

    message = str(info.value)
    assert "--embed-model" in message and "--drop-orphans" in message
    assert "tools/reembed_index.py --apply" in message


@pytest.mark.asyncio
async def test_wrong_marker_is_refused_and_names_both_schemes():
    old = "nomic-embed-text:raw/v0"
    state = FakeState([marker(old)])

    with pytest.raises(IndexSchemeMismatch) as info:
        await ensure_index_scheme(state, counter(1009), SCHEME)

    message = str(info.value)
    assert old in message and SCHEME in message
    assert "tools/reembed_index.py --apply" in message
    assert state.docs[MARKER_ID]["scheme"] == old  # маркер не чіпаємо


@pytest.mark.asyncio
async def test_empty_index_overrides_a_stale_marker():
    # Таблицю створено заново: старий маркер більше нічого не описує.
    state = FakeState([marker("old")])

    await ensure_index_scheme(state, counter(0), SCHEME)

    assert state.docs[MARKER_ID]["scheme"] == SCHEME


def test_mismatch_is_a_runtime_error():
    assert issubclass(IndexSchemeMismatch, RuntimeError)


# --- record_index_scheme ---------------------------------------------------
@pytest.mark.asyncio
async def test_record_upserts_scheme_count_and_time():
    state = FakeState()

    await record_index_scheme(state, SCHEME, 1009)

    doc = state.docs[MARKER_ID]
    assert doc["scheme"] == SCHEME
    assert doc["documents"] == 1009
    assert doc["recorded_at"] is not None
    assert state.writes[0][2] is True  # upsert


@pytest.mark.asyncio
async def test_record_overwrites_the_previous_marker():
    state = FakeState([marker("old", 1)])

    await record_index_scheme(state, SCHEME, 7)

    assert state.docs[MARKER_ID]["scheme"] == SCHEME
    assert state.docs[MARKER_ID]["documents"] == 7


# --- check_index_scheme: обгортка для main/replay ---------------------------
class FakeSink:
    def __init__(self, documents):
        self.documents = documents

    async def count_documents(self):
        return self.documents


class FakeLogger:
    def __init__(self):
        self.errors = []

    def error(self, event, **kw):
        self.errors.append((event, kw))


@pytest.mark.asyncio
async def test_check_returns_true_when_scheme_matches():
    log = FakeLogger()
    ok = await check_index_scheme(FakeSink(5), FakeState([marker()]), log, SCHEME)
    assert ok is True
    assert log.errors == []


@pytest.mark.asyncio
async def test_check_returns_false_and_logs_on_mismatch():
    log = FakeLogger()
    ok = await check_index_scheme(FakeSink(1009), FakeState(), log, SCHEME)
    assert ok is False
    event, kw = log.errors[0]
    assert event == "index_scheme_mismatch"
    assert "tools/reembed_index.py --apply" in kw["error"]


@pytest.mark.asyncio
async def test_check_defaults_to_the_current_scheme():
    from shipper.config import settings

    state = FakeState([marker(current_scheme(settings))])
    assert await check_index_scheme(FakeSink(5), state, FakeLogger()) is True


def test_marker_fields_is_the_single_description_of_the_marker_body():
    """Той самий склад полів пише і шиппер (motor), і tools/reembed_index.py
    (pymongo): формат маркера в одному місці."""
    from datetime import datetime, timezone

    now = datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)
    assert marker_fields(SCHEME, 1009, now) == {
        "scheme": SCHEME, "documents": 1009, "recorded_at": now}
    assert marker_fields(SCHEME, 5)["recorded_at"].tzinfo is not None
