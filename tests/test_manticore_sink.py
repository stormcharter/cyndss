"""Тести ManticoreSink: допоміжні функції, ретраї/DLQ, розбір по елементах, обмежена черга, ack."""

from datetime import datetime, timezone
import asyncio
import httpx
import pytest

from shipper.config import settings
from shipper.manticore_sink import ManticoreSink, SinkAck, ManticoreBulkMismatch
from shipper.models import (
    EnrichedDocument,
    IoCEntity,
    SourceType,
    VerifiedCVE,
    VerifiedRelation,
    VerifiedTTP,
    VerificationStatus,
)


def _make_doc(mongo_id: str = "665f1a2b3c4d5e6f7a8b9c0d") -> EnrichedDocument:
    return EnrichedDocument(
        mongo_id=mongo_id,
        source_type=SourceType.TELEGRAM,
        channel_name="test_channel",
        url="https://t.me/test/123",
        collected_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        title="Test event",
        language="en",
        text="Full text body",
        summary="Short summary",
        embedding=[0.1] * 768,
        verified_cves=[
            VerifiedCVE(cve_id="CVE-2024-3094", status=VerificationStatus.VERIFIED),
        ],
        verified_ttps=[
            VerifiedTTP(technique_id="T1190", status=VerificationStatus.VERIFIED),
        ],
        verified_iocs=[
            IoCEntity(ioc_type="ip", value="1.2.3.4", claimed_context=None),
        ],
        verified_relations=[
            VerifiedRelation(
                subject="CVE-2024-3094",
                predicate="exploited_via",
                object="T1190",
                status=VerificationStatus.VERIFIED,
            ),
        ],
        cve_count_verified=1,
        ttp_count_verified=1,
        ioc_count=1,
    )


def test_stable_id_consistency() -> None:
    mongo_id = "665f1a2b3c4d5e6f7a8b9c0d"
    id1 = ManticoreSink._stable_id(mongo_id)
    id2 = ManticoreSink._stable_id(mongo_id)
    assert id1 == id2, "stable_id should be deterministic"
    assert isinstance(id1, int)


def test_stable_id_different_mongo_ids_differ() -> None:
    id1 = ManticoreSink._stable_id("665f1a2b3c4d5e6f7a8b9c0d")
    id2 = ManticoreSink._stable_id("665f1a2b3c4d5e6f7a8b9c0e")
    assert id1 != id2


def test_stable_id_fallback_for_short_input() -> None:
    result = ManticoreSink._stable_id("short")
    assert isinstance(result, int)


def test_to_manticore_doc_shape() -> None:
    doc = _make_doc()
    result = ManticoreSink._to_manticore_doc(doc)
    assert result["mongo_id"] == doc.mongo_id
    assert result["source_type"] == "telegram"
    assert result["channel_name"] == "test_channel"
    assert result["title"] == "Test event"
    assert result["language"] == "en"
    assert result["cve_count_verified"] == 1
    assert result["ttp_count_verified"] == 1
    assert result["ioc_count"] == 1
    assert "CVE-2024-3094" in result["verified_cve_ids"]
    assert "T1190" in result["verified_ttp_ids"]
    assert len(result["embedding"]) == 768
    assert result["content"] == "Full text body"
    assert isinstance(result["collected_at"], int)


def test_to_manticore_doc_empty_lists() -> None:
    doc = _make_doc()
    doc.verified_cves = []
    doc.verified_ttps = []
    doc.cve_count_verified = 0
    doc.ttp_count_verified = 0
    doc.ioc_count = 0
    result = ManticoreSink._to_manticore_doc(doc)
    assert result["verified_cve_ids"] == ""
    assert result["verified_ttp_ids"] == ""
    assert result["cve_count_verified"] == 0


# --- Фейковий DLQ (ідемпотентний на подію) ---------------------------------

class _FakeDLQCollection:
    def __init__(self):
        self.docs: list[dict] = []
        self.fail_next = False

    async def update_one(self, filter, update, upsert=False):
        if self.fail_next:
            raise RuntimeError("DLQ unavailable")
        doc = {"_id": filter.get("_id", "?"), **update.get("$set", {})}
        self.docs.append(doc)
        return type("R", (), {"modified_count": 1, "upserted_id": None})()


# --- Helpers -----------------------------------------------------------------

def _patch_fast_retries(monkeypatch):
    monkeypatch.setattr(settings, "manticore_max_retries", 2)
    monkeypatch.setattr(settings, "manticore_retry_base_seconds", 0.0)
    monkeypatch.setattr(settings, "sink_queue_maxsize", 10)


def _bulk_ok_response(n: int) -> dict:
    """Імітує відповідь Manticore /bulk, у якій усі елементи успішні."""
    return {
        "items": [
            {"bulk": {"table": "events_enriched", "_id": i, "created": 1, "deleted": 0, "updated": 0, "result": "created", "status": 201}}
            for i in range(n)
        ],
        "current_line": n,
        "skipped_lines": 0,
        "errors": False,
        "error": "",
    }


def _bulk_partial_fail_response(n: int, fail_indices: set[int]) -> dict:
    """Simulate Manticore /bulk with some items failing."""
    items = []
    for i in range(n):
        if i in fail_indices:
            items.append({"replace": {"error": {"type": "KNN dimension mismatch"}, "status": 409}})
        else:
            items.append({"bulk": {"table": "events_enriched", "_id": i, "created": 1, "result": "created", "status": 201}})
    return {"items": items, "current_line": n, "skipped_lines": 0, "errors": bool(fail_indices), "error": ""}


# --- Тести ретраїв + DLQ ----------------------------------------------------

@pytest.mark.asyncio
async def test_flush_with_retry_succeeds_first_try(monkeypatch) -> None:
    _patch_fast_retries(monkeypatch)
    dlq = _FakeDLQCollection()
    sink = ManticoreSink(dead_letter_collection=dlq)
    docs = [_make_doc()]

    async def _flush(batch):
        return {d.mongo_id: (True, "") for d in batch}

    monkeypatch.setattr(sink, "_flush", _flush)
    acks = [SinkAck() for _ in docs]
    await sink._flush_with_retry(list(zip(docs, acks)))
    assert all(a.status == "ok" for a in acks)
    assert len(dlq.docs) == 0


@pytest.mark.asyncio
async def test_flush_with_retry_succeeds_after_retry(monkeypatch) -> None:
    _patch_fast_retries(monkeypatch)
    dlq = _FakeDLQCollection()
    sink = ManticoreSink(dead_letter_collection=dlq)
    docs = [_make_doc()]
    attempts = []

    async def _flush(batch):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("first failure")
        return {d.mongo_id: (True, "") for d in batch}

    monkeypatch.setattr(sink, "_flush", _flush)
    acks = [SinkAck() for _ in docs]
    await sink._flush_with_retry(list(zip(docs, acks)))
    assert len(attempts) == 2
    assert all(a.status == "ok" for a in acks)
    assert len(dlq.docs) == 0


@pytest.mark.asyncio
async def test_flush_with_retry_exhausted_writes_dead_letter(monkeypatch) -> None:
    _patch_fast_retries(monkeypatch)
    dlq = _FakeDLQCollection()
    sink = ManticoreSink(dead_letter_collection=dlq)
    docs = [_make_doc("abc123"), _make_doc("def456")]

    async def _flush(batch):
        raise httpx.ConnectError("manticore down")

    monkeypatch.setattr(sink, "_flush", _flush)
    acks = [SinkAck() for _ in docs]
    await sink._flush_with_retry(list(zip(docs, acks)))

    assert len(dlq.docs) == 2  # DLQ на кожну подію
    dlq_ids = {d["mongo_id"] for d in dlq.docs}
    assert "abc123" in dlq_ids
    assert "def456" in dlq_ids
    for d in dlq.docs:
        assert d["stage"] == "manticore_bulk"
        assert d["attempts"] == 2
        assert d["error_class"] == "ManticoreBulkError"
        assert "manticore down" in d["error"]
    # Acks resolved as dead_lettered (DLQ ok)
    assert all(a.status == "dead_lettered" for a in acks)


@pytest.mark.asyncio
async def test_dead_letter_failure_resolves_failed(monkeypatch) -> None:
    _patch_fast_retries(monkeypatch)
    dlq = _FakeDLQCollection()
    dlq.fail_next = True
    sink = ManticoreSink(dead_letter_collection=dlq)

    async def _flush(batch):
        raise httpx.ConnectError("manticore down")

    monkeypatch.setattr(sink, "_flush", _flush)
    docs = [_make_doc("abc123")]
    acks = [SinkAck() for _ in docs]
    await sink._flush_with_retry(list(zip(docs, acks)))

    # Запис у DLQ не вдався → ack розв'язано як "failed" (НЕ dead_lettered)
    assert acks[0].status == "failed"
    assert acks[0].error is not None


# --- Тести розбору bulk-відповіді по елементах -----------------------------

@pytest.mark.asyncio
async def test_flush_parses_per_item_ok(monkeypatch) -> None:
    _patch_fast_retries(monkeypatch)
    sink = ManticoreSink()
    docs = [_make_doc("a"), _make_doc("b")]

    async def _post_bulk(*args, **kwargs):
        return type("R", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: _bulk_ok_response(2),
        })()

    monkeypatch.setattr(sink.client, "post", _post_bulk)
    result = await sink._flush(docs)
    assert result["a"] == (True, "")
    assert result["b"] == (True, "")


@pytest.mark.asyncio
async def test_flush_parses_per_item_partial_failure(monkeypatch) -> None:
    _patch_fast_retries(monkeypatch)
    sink = ManticoreSink()
    docs = [_make_doc("ok1"), _make_doc("fail1")]

    async def _post_bulk(*args, **kwargs):
        return type("R", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: _bulk_partial_fail_response(2, {1}),
        })()

    monkeypatch.setattr(sink.client, "post", _post_bulk)
    result = await sink._flush(docs)
    assert result["ok1"] == (True, "")
    assert result["fail1"][0] is False
    assert "dimension" in result["fail1"][1].lower() or "knn" in result["fail1"][1].lower()


@pytest.mark.asyncio
async def test_flush_raises_on_item_count_mismatch(monkeypatch) -> None:
    _patch_fast_retries(monkeypatch)
    sink = ManticoreSink()
    docs = [_make_doc("a"), _make_doc("b")]

    async def _post_bulk(*args, **kwargs):
        return type("R", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: _bulk_ok_response(1),  # 1 елемент на 2 документи
        })()

    monkeypatch.setattr(sink.client, "post", _post_bulk)
    with pytest.raises(ManticoreBulkMismatch):
        await sink._flush(docs)


@pytest.mark.asyncio
async def test_item_level_failure_triggers_dlq(monkeypatch) -> None:
    """409 на рівні елемента після ретраїв → DLQ на подію, а не лише транспортний DLQ."""
    _patch_fast_retries(monkeypatch)
    dlq = _FakeDLQCollection()
    sink = ManticoreSink(dead_letter_collection=dlq)
    docs = [_make_doc("ok1"), _make_doc("bad1")]

    async def _flush(batch):
        return {
            doc.mongo_id: (
                doc.mongo_id != "bad1",
                "" if doc.mongo_id != "bad1" else "KNN dimension mismatch",
            )
            for doc in batch
        }

    monkeypatch.setattr(sink, "_flush", _flush)
    acks = [SinkAck() for _ in docs]
    await sink._flush_with_retry(list(zip(docs, acks)))

    assert acks[0].status == "ok"   # ok item acked
    assert acks[1].status == "dead_lettered"  # failed item parked
    assert len(dlq.docs) == 1
    assert dlq.docs[0]["mongo_id"] == "bad1"


# --- Тест обмеженої черги + backpressure -----------------------------------

@pytest.mark.asyncio
async def test_bounded_queue_blocks_when_full(monkeypatch) -> None:
    monkeypatch.setattr(settings, "sink_queue_maxsize", 2)
    monkeypatch.setattr(settings, "manticore_max_retries", 1)
    monkeypatch.setattr(settings, "manticore_retry_base_seconds", 0.0)
    monkeypatch.setattr(settings, "batch_size", 50)
    sink = ManticoreSink()

    async def _mock_flush(batch):
        return {d.mongo_id: (True, "") for d in batch}

    monkeypatch.setattr(sink, "_flush", _mock_flush)
    sink.start_flusher()

    # Місткість черги = 2. Кладемо 3 документи — третій має блокуватися, доки flusher не розвантажить чергу.
    acks = []
    push_tasks = []
    for i in range(3):
        doc = _make_doc(f"q{i}")
        ack = SinkAck()
        push_tasks.append(asyncio.create_task(_push_and_collect(sink, doc, ack)))
        acks.append(ack)

    # Даємо задачам час стати в чергу, а flusher-у — обробити
    await asyncio.sleep(0.5)

    # Зупиняємо flusher (він дренує решту черги)
    sink._stop_event.set()
    for t in push_tasks:
        await asyncio.wait_for(t, timeout=3.0)

    # Усі ack зрештою мають розв'язатися (дренаж їх розв'язує)
    for a in acks:
        assert a.event.is_set()


async def _push_and_collect(sink, doc, ack):
    await sink.queue.put((doc, ack))
    await ack.event.wait()


# --- SinkAck tests ----------------------------------------------------------

def test_sink_ack_resolve_idempotent() -> None:
    ack = SinkAck()
    assert not ack.event.is_set()
    ack.resolve("ok")
    assert ack.event.is_set()
    assert ack.status == "ok"
    # Повторний resolve нічого не змінює
    ack.resolve("dead_lettered")
    assert ack.status == "ok"  # без змін


def test_sink_ack_default_status() -> None:
    ack = SinkAck()
    assert ack.status == "failed"
    assert ack.error is None
