"""
Що ембедить конвеєр: англійське «title. summary», а не сирий оригінал.

КОНТЕКСТ (виміряно на живому стеку, 1486 подій, корпус 832 uk / 174 ru / 3 en):
сирий український текст у англоцентричному nomic (1) втратив 15 довгих постів
через HTTP 400 «input length exceeds the context length», (2) дав 462 хибних
семантичних дублікати (cosine >= 0.92 між РІЗНИМИ постами), (3) зробив KNN
марним для англійських запитів. Тому вектор рахується з модельного
«title. summary» (див. embedding_text.py), один раз на документ, і слугує і
дедупу, і індексу.

Запуск:
    python -m pytest tests/test_pipeline_embedding.py
(потрібні pydantic/numpy/structlog).
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import structlog

from embedding_text import EMBED_TEXT_CHARS, SUMMARY_PLACEHOLDER
from shipper.config import settings
from shipper.dedup import SemanticDeduplicator
from shipper.models import ExtractedEntities, RawEvent, SourceMetadata, SourceType
from shipper.pipeline import Pipeline, ProcessStatus

TEXT = "Дослідники повідомили про атаку на поштовий шлюз великої компанії."


class RecordingEmbedder:
    """Запам'ятовує, який саме текст дістався ембедеру; віддає вектор за словником."""

    def __init__(self, vector=None, delay: float = 0.0) -> None:
        self.texts: list[str] = []
        self.vector = vector if vector is not None else [1.0, 0.0]
        self.delay = delay

    async def embed(self, text: str) -> np.ndarray:
        self.texts.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        return np.array(self.vector, dtype=np.float32)


class ScriptedExtractor:
    def __init__(self, entities: ExtractedEntities) -> None:
        self.entities = entities
        self.calls = 0

    async def extract(self, text: str) -> ExtractedEntities:
        self.calls += 1
        return self.entities


class EmptyVerifier:
    async def verify(self, extracted):
        return [], [], [], []


class RecordingSink:
    def __init__(self) -> None:
        self.documents = []

    async def push(self, document):
        self.documents.append(document)
        return SimpleNamespace(status="ok", error=None)


def _event(text: str = TEXT, mongo_id: str = "507f1f77bcf86cd799439011") -> RawEvent:
    return RawEvent(
        _id=mongo_id,
        source=SourceMetadata(source_type=SourceType.TELEGRAM),
        text=text,
        collected_at=datetime.now(timezone.utc),
    )


_DEFAULT = object()


def _pipeline(entities, *, embedder=None, dedup=_DEFAULT, sink=None, fast_mode=False):
    """За замовчуванням дедуп ВІДСУТНІЙ у аргументах — береться справжній
    `SemanticDeduplicator`; вимкнений дедуп задається явно `dedup=None`."""
    if dedup is _DEFAULT:
        dedup = SemanticDeduplicator()
    embedder = embedder or RecordingEmbedder()
    sink = sink or RecordingSink()
    extractor = ScriptedExtractor(entities)
    pipeline = Pipeline(
        embedder=embedder,
        extractor=extractor,
        verifier=EmptyVerifier(),
        dedup=dedup,
        sink=sink,
        fast_mode=fast_mode,
    )
    return pipeline, embedder, extractor, sink


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(settings, "pipeline_max_retries", 1)
    monkeypatch.setattr(settings, "pipeline_retry_base_seconds", 0.0)


# --- що ембедиться ---------------------------------------------------------
@pytest.mark.asyncio
async def test_embeds_title_and_summary_not_the_raw_text():
    pipeline, embedder, _ex, _sink = _pipeline(ExtractedEntities(
        title="Mail gateway attacked", summary="Attackers hit a mail gateway."))

    result = await pipeline.process(_event())

    assert result.status is ProcessStatus.INDEXED
    assert embedder.texts == ["Mail gateway attacked. Attackers hit a mail gateway."]


@pytest.mark.asyncio
async def test_missing_summary_embeds_raw_text_not_the_placeholder():
    # Модель не дала summary: у документ іде плейсхолдер, а ембедиться — оригінал.
    pipeline, embedder, _ex, sink = _pipeline(ExtractedEntities(title="Untitled"))

    await pipeline.process(_event())

    assert embedder.texts == [TEXT]
    assert sink.documents[0].summary == SUMMARY_PLACEHOLDER


@pytest.mark.asyncio
async def test_long_raw_text_is_capped_before_the_embedder():
    # Найкоротший відхилений Ollama пост — 2994 символи: стеля має спрацювати раніше.
    pipeline, embedder, _ex, _sink = _pipeline(ExtractedEntities())

    await pipeline.process(_event("к" * 5000))

    assert len(embedder.texts[0]) == EMBED_TEXT_CHARS


@pytest.mark.asyncio
async def test_fast_mode_embeds_the_capped_raw_text_without_extraction():
    pipeline, embedder, extractor, _sink = _pipeline(
        ExtractedEntities(title="ignored", summary="ignored"), fast_mode=True)

    await pipeline.process(_event("к" * 5000))

    assert extractor.calls == 0
    assert embedder.texts == ["к" * EMBED_TEXT_CHARS]


@pytest.mark.asyncio
async def test_one_embedding_serves_dedup_and_the_index():
    embedder = RecordingEmbedder(vector=[0.0, 1.0])
    pipeline, embedder, _ex, sink = _pipeline(
        ExtractedEntities(title="T", summary="S"),
        embedder=embedder, dedup=SemanticDeduplicator())

    await pipeline.process(_event())

    assert len(embedder.texts) == 1
    assert sink.documents[0].embedding == [0.0, 1.0]


# --- ретраї не платять за екстракцію вдруге ---------------------------------
class FlakyEmbedder(RecordingEmbedder):
    """Падає на перших `fail_times` викликах, далі працює."""

    def __init__(self, fail_times: int = 1) -> None:
        super().__init__()
        self.fail_times = fail_times

    async def embed(self, text: str) -> np.ndarray:
        if len(self.texts) < self.fail_times:
            self.texts.append(text)
            raise RuntimeError("embed: Ollama 500")
        return await super().embed(text)


class CountingVerifier:
    def __init__(self) -> None:
        self.calls = 0

    async def verify(self, extracted):
        self.calls += 1
        return [], [], [], []


class FlakySink(RecordingSink):
    def __init__(self, fail_times: int = 1) -> None:
        super().__init__()
        self.fail_times = fail_times
        self.attempts = 0

    async def push(self, document):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            return SimpleNamespace(status="failed", error="manticore down")
        return await super().push(document)


@pytest.mark.asyncio
async def test_embed_failure_retry_does_not_reextract_or_reverify(monkeypatch):
    # Ембединг тепер ПІСЛЯ екстракції+верифікації: збій ембедингу не повинен
    # змушувати ретрай знову платити за LLM і NVD.
    monkeypatch.setattr(settings, "pipeline_max_retries", 2)
    pipeline, embedder, extractor, sink = _pipeline(
        ExtractedEntities(title="T", summary="S"), embedder=FlakyEmbedder(1))
    verifier = CountingVerifier()
    pipeline.verifier = verifier

    result = await pipeline.process(_event())

    assert result.status is ProcessStatus.INDEXED
    assert extractor.calls == 1
    assert verifier.calls == 1
    assert len(embedder.texts) == 2     # перша спроба впала, друга пройшла
    assert len(sink.documents) == 1


@pytest.mark.asyncio
async def test_sink_failure_retry_does_not_reextract_or_reverify(monkeypatch):
    monkeypatch.setattr(settings, "pipeline_max_retries", 2)
    sink = FlakySink(1)
    pipeline, _emb, extractor, sink = _pipeline(
        ExtractedEntities(title="T", summary="S"), sink=sink, dedup=None)
    verifier = CountingVerifier()
    pipeline.verifier = verifier

    result = await pipeline.process(_event())

    assert result.status is ProcessStatus.INDEXED
    assert extractor.calls == 1 and verifier.calls == 1
    assert sink.attempts == 2


@pytest.mark.asyncio
async def test_stages_are_per_event_not_shared_between_events(monkeypatch):
    monkeypatch.setattr(settings, "pipeline_max_retries", 2)
    pipeline, _emb, extractor, _sink = _pipeline(
        ExtractedEntities(title="T", summary="S"), dedup=None)

    await pipeline.process(_event(mongo_id="507f1f77bcf86cd799439011"))
    await pipeline.process(_event(mongo_id="507f1f77bcf86cd799439012"))

    assert extractor.calls == 2


@pytest.mark.asyncio
async def test_fast_mode_retry_still_skips_extraction(monkeypatch):
    monkeypatch.setattr(settings, "pipeline_max_retries", 2)
    pipeline, embedder, extractor, _sink = _pipeline(
        ExtractedEntities(title="ignored"), embedder=FlakyEmbedder(1), fast_mode=True)

    result = await pipeline.process(_event())

    assert result.status is ProcessStatus.INDEXED
    assert extractor.calls == 0
    assert len(embedder.texts) == 2


@pytest.mark.asyncio
async def test_process_unsafe_still_works_called_directly_without_stages():
    pipeline, _emb, extractor, sink = _pipeline(ExtractedEntities(title="T", summary="S"))

    result = await pipeline._process_unsafe(_event())

    assert result.status is ProcessStatus.INDEXED
    assert extractor.calls == 1 and len(sink.documents) == 1


# --- вимкнений семантичний дедуп -------------------------------------------
@pytest.mark.asyncio
async def test_dedup_none_indexes_identical_events():
    # dedup=None означає «вимкнено»: однакові вектори не зливаються.
    pipeline, _emb, _ex, sink = _pipeline(
        ExtractedEntities(title="T", summary="S"), dedup=None)

    first = await pipeline.process(_event(mongo_id="507f1f77bcf86cd799439011"))
    second = await pipeline.process(_event(mongo_id="507f1f77bcf86cd799439012"))

    assert first.status is ProcessStatus.INDEXED
    assert second.status is ProcessStatus.INDEXED
    assert len(sink.documents) == 2


@pytest.mark.asyncio
async def test_explicit_dedup_still_merges_identical_vectors():
    pipeline, _emb, _ex, sink = _pipeline(
        ExtractedEntities(title="T", summary="S"), dedup=SemanticDeduplicator())

    first = await pipeline.process(_event(mongo_id="507f1f77bcf86cd799439011"))
    second = await pipeline.process(_event(mongo_id="507f1f77bcf86cd799439012"))

    assert first.status is ProcessStatus.INDEXED
    assert second.status is ProcessStatus.DUPLICATE
    assert len(sink.documents) == 1


# --- логи латентності ------------------------------------------------------
@pytest.mark.asyncio
async def test_embed_latency_log_measures_the_embedding_call():
    embedder = RecordingEmbedder(delay=0.06)
    pipeline, embedder, _ex, _sink = _pipeline(
        ExtractedEntities(title="T", summary="S"), embedder=embedder)

    with structlog.testing.capture_logs() as logs:
        await pipeline.process(_event())

    success = next(entry for entry in logs if entry["event"] == "pipeline_success")
    assert success["embed_latency_ms"] >= 50
    # Екстракція тут миттєва: її час не має «з'їдати» час ембедингу.
    assert success["extract_latency_ms"] < 50
    assert success["total_latency_ms"] >= success["embed_latency_ms"]
