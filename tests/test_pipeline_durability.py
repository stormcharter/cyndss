from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from shipper.config import settings
from shipper.dedup import SemanticDeduplicator
from shipper.models import ExtractedEntities, RawEvent, SourceMetadata, SourceType
from shipper.pipeline import Pipeline, ProcessStatus


class StaticEmbedder:
    async def embed(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class FailingExtractor:
    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, text: str) -> ExtractedEntities:
        self.calls += 1
        raise RuntimeError("extraction failed")


class SuccessfulExtractor:
    async def extract(self, text: str) -> ExtractedEntities:
        return ExtractedEntities(title="Indexed event", summary="summary")


class EmptyVerifier:
    async def verify(self, extracted: ExtractedEntities) -> tuple[list, list, list, list]:
        return [], [], [], []


class RecordingSink:
    def __init__(self, statuses: list[str] | None = None) -> None:
        self.documents = []
        self.statuses = list(statuses or ["ok"])

    async def push(self, document):
        self.documents.append(document)
        status = self.statuses.pop(0)
        error = "Manticore unavailable" if status != "ok" else None
        return SimpleNamespace(status=status, error=error)


def _event(mongo_id: str) -> RawEvent:
    return RawEvent(
        _id=mongo_id,
        source=SourceMetadata(source_type=SourceType.TELEGRAM),
        text="same cluster content",
        collected_at=datetime.now(timezone.utc),
    )


def _pipeline(extractor, dedup, sink) -> Pipeline:
    return Pipeline(
        embedder=StaticEmbedder(),
        extractor=extractor,
        verifier=EmptyVerifier(),
        dedup=dedup,
        sink=sink,
    )


@pytest.fixture(autouse=True)
def fast_pipeline_retries(monkeypatch):
    monkeypatch.setattr(settings, "pipeline_max_retries", 2)
    monkeypatch.setattr(settings, "pipeline_retry_base_seconds", 0.0)


@pytest.mark.asyncio
async def test_retry_does_not_self_deduplicate_failed_event() -> None:
    extractor = FailingExtractor()
    dedup = SemanticDeduplicator()
    sink = RecordingSink()

    result = await _pipeline(extractor, dedup, sink).process(_event("canonical"))

    assert result.status is ProcessStatus.FAILED
    assert not result.acknowledged
    assert extractor.calls == 2
    assert not dedup.window
    assert not sink.documents


@pytest.mark.asyncio
async def test_failed_canonical_does_not_swallow_duplicate_cluster() -> None:
    dedup = SemanticDeduplicator()
    sink = RecordingSink()

    failed = await _pipeline(FailingExtractor(), dedup, sink).process(
        _event("failed-canonical")
    )
    indexed = await _pipeline(SuccessfulExtractor(), dedup, sink).process(
        _event("next-cluster-event")
    )

    assert failed.status is ProcessStatus.FAILED
    assert indexed.status is ProcessStatus.INDEXED
    assert [document.mongo_id for document in sink.documents] == ["next-cluster-event"]
    assert [record.mongo_id for record in dedup.window] == ["next-cluster-event"]


@pytest.mark.asyncio
async def test_dead_lettered_canonical_does_not_swallow_duplicate_cluster() -> None:
    dedup = SemanticDeduplicator()
    sink = RecordingSink(["dead_lettered", "ok"])
    pipeline = _pipeline(SuccessfulExtractor(), dedup, sink)

    dead_lettered = await pipeline.process(_event("dead-lettered-canonical"))
    indexed = await pipeline.process(_event("next-cluster-event"))

    assert dead_lettered.status is ProcessStatus.DEAD_LETTERED
    assert dead_lettered.acknowledged
    assert indexed.status is ProcessStatus.INDEXED
    assert [document.mongo_id for document in sink.documents] == [
        "dead-lettered-canonical",
        "next-cluster-event",
    ]
    assert [record.mongo_id for record in dedup.window] == ["next-cluster-event"]


@pytest.mark.asyncio
async def test_duplicate_is_acknowledged_only_after_canonical_sink_ack() -> None:
    dedup = SemanticDeduplicator()
    sink = RecordingSink()
    pipeline = _pipeline(SuccessfulExtractor(), dedup, sink)

    canonical = await pipeline.process(_event("canonical"))
    duplicate = await pipeline.process(_event("duplicate"))

    assert canonical.status is ProcessStatus.INDEXED
    assert duplicate.status is ProcessStatus.DUPLICATE
    assert duplicate.acknowledged
    assert [document.mongo_id for document in sink.documents] == ["canonical"]
