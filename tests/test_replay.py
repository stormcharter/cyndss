from datetime import datetime, timezone

from bson import ObjectId
import pytest

from shipper.pipeline import ProcessResult, ProcessStatus
from shipper.replay import replay_dead_letters, replay_missing


class FakeCursor:
    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries
        self.maximum: int | None = None

    def sort(self, field: str, direction: int):
        self.entries.sort(key=lambda entry: entry[field])
        return self

    def limit(self, maximum: int):
        self.maximum = maximum
        return self

    def __aiter__(self):
        entries = self.entries[: self.maximum] if self.maximum else self.entries

        async def iterate():
            for entry in entries:
                yield entry

        return iterate()


class FakeDeadLetters:
    def __init__(self, entries: list[dict]) -> None:
        self.entries = {entry["_id"]: entry for entry in entries}

    def find(self, query: dict) -> FakeCursor:
        matches = [
            entry
            for entry in self.entries.values()
            if entry.get("replayed_at") is query["replayed_at"]
            and entry.get("pipeline_revision") == query["pipeline_revision"]
        ]
        return FakeCursor(matches)

    async def update_one(self, query: dict, update: dict, upsert: bool = False):
        entry = self.entries.get(query["_id"])
        if entry is None:
            return None
        if "replayed_at" in query and entry.get("replayed_at") is not query["replayed_at"]:
            return None
        for field, increment in update.get("$inc", {}).items():
            entry[field] = entry.get(field, 0) + increment
        entry.update(update.get("$set", {}))
        return None


class FakeEvents:
    def __init__(self, documents: dict) -> None:
        self.documents = documents
        self.queries: list[dict] = []

    async def find_one(self, query: dict):
        self.queries.append(query)
        return self.documents.get(query["_id"])


class ResultPipeline:
    def __init__(self, result: ProcessResult) -> None:
        self.result = result
        self.events = []

    async def process(self, event, resume_token):
        self.events.append((event, resume_token))
        return self.result


def _entry(source_id: ObjectId | str) -> dict:
    mongo_id = str(source_id)
    return {
        "_id": f"dlq:2:{mongo_id}",
        "mongo_id": mongo_id,
        "pipeline_revision": "2",
        "failed_at": datetime.now(timezone.utc),
        "replayed_at": None,
    }


def _source_document(source_id: ObjectId | str) -> dict:
    return {
        "_id": source_id,
        "source_type": "telegram",
        "text": "event text suitable for replay",
        "collected_at": datetime.now(timezone.utc),
    }


@pytest.mark.asyncio
async def test_replay_loads_normal_source_by_object_id() -> None:
    object_id = ObjectId()
    entry = _entry(object_id)
    dead_letters = FakeDeadLetters([entry])
    events = FakeEvents({object_id: _source_document(object_id)})
    pipeline = ResultPipeline(ProcessResult(str(object_id), ProcessStatus.INDEXED))

    result = await replay_dead_letters(
        collection=dead_letters,
        events=events,
        pipeline=pipeline,
    )

    assert result == {"replayed": 1, "failed": 0}
    assert events.queries == [{"_id": object_id}]
    assert entry["replayed_at"] is not None
    assert len(pipeline.events) == 1


@pytest.mark.asyncio
async def test_replay_falls_back_to_legacy_string_id() -> None:
    string_id = "0123456789abcdef01234567"
    entry = _entry(string_id)
    dead_letters = FakeDeadLetters([entry])
    events = FakeEvents({string_id: _source_document(string_id)})
    pipeline = ResultPipeline(ProcessResult(string_id, ProcessStatus.INDEXED))

    result = await replay_dead_letters(
        collection=dead_letters,
        events=events,
        pipeline=pipeline,
    )

    assert result == {"replayed": 1, "failed": 0}
    assert events.queries == [
        {"_id": ObjectId(string_id)},
        {"_id": string_id},
    ]
    assert entry["replayed_at"] is not None


@pytest.mark.asyncio
async def test_missing_source_remains_unresolved() -> None:
    object_id = ObjectId()
    entry = _entry(object_id)
    dead_letters = FakeDeadLetters([entry])
    events = FakeEvents({})
    pipeline = ResultPipeline(ProcessResult(str(object_id), ProcessStatus.INDEXED))

    result = await replay_dead_letters(
        collection=dead_letters,
        events=events,
        pipeline=pipeline,
    )

    assert result == {"replayed": 0, "failed": 1}
    assert entry["replayed_at"] is None
    assert entry["replay_attempts"] == 1
    assert entry["last_replay_error"] == "source event missing"
    assert not pipeline.events


@pytest.mark.asyncio
async def test_dead_lettered_replay_remains_unresolved() -> None:
    object_id = ObjectId()
    entry = _entry(object_id)
    dead_letters = FakeDeadLetters([entry])
    events = FakeEvents({object_id: _source_document(object_id)})
    pipeline = ResultPipeline(
        ProcessResult(
            str(object_id),
            ProcessStatus.DEAD_LETTERED,
            error="Manticore unavailable",
            error_class="ManticoreBulkError",
        )
    )

    result = await replay_dead_letters(
        collection=dead_letters,
        events=events,
        pipeline=pipeline,
    )

    assert result == {"replayed": 0, "failed": 1}
    assert entry["replayed_at"] is None
    assert entry["replay_attempts"] == 1
    assert entry["last_replay_error"] == "Manticore unavailable"


@pytest.mark.asyncio
async def test_durable_duplicate_resolves_replay() -> None:
    object_id = ObjectId()
    entry = _entry(object_id)
    dead_letters = FakeDeadLetters([entry])
    events = FakeEvents({object_id: _source_document(object_id)})
    pipeline = ResultPipeline(
        ProcessResult(str(object_id), ProcessStatus.DUPLICATE)
    )

    result = await replay_dead_letters(
        collection=dead_letters,
        events=events,
        pipeline=pipeline,
    )

    assert result == {"replayed": 1, "failed": 0}
    assert entry["replayed_at"] is not None
    assert "replay_attempts" not in entry


# --- replay --missing: події, яких немає в індексі ---------------------------
class FakeEventStream:
    """Колекція raw_events: `find(query).sort("_id", 1)` -> асинхронна ітерація."""

    def __init__(self, documents: list[dict]) -> None:
        self.documents = list(documents)
        self.queries: list[dict] = []

    def find(self, query: dict | None = None) -> "FakeEventStream":
        self.queries.append(query or {})
        return self

    def sort(self, field: str, direction: int) -> "FakeEventStream":
        self.documents.sort(key=lambda doc: doc[field], reverse=direction < 0)
        return self

    def __aiter__(self):
        async def iterate():
            for document in self.documents:
                yield document

        return iterate()


class ScriptedPipeline:
    """Віддає статуси по черзі (останній повторюється); запам'ятовує виклики."""

    def __init__(self, *statuses) -> None:
        self.statuses = list(statuses) or [ProcessStatus.INDEXED]
        self.calls: list[tuple] = []

    async def process(self, event, resume_token):
        self.calls.append((event.id, resume_token))
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(status, Exception):
            raise status
        return ProcessResult(event.id, status)


def _ids(count: int) -> list[ObjectId]:
    return [ObjectId(f"{i:024x}") for i in range(1, count + 1)]


@pytest.mark.asyncio
async def test_missing_replays_only_events_absent_from_the_index_in_id_order() -> None:
    a, b, c, d = _ids(4)
    events = FakeEventStream([_source_document(x) for x in (d, b, c, a)])  # навмисно впереміш
    pipeline = ScriptedPipeline()

    result = await replay_missing(
        events=events, indexed_ids={str(b), str(d)}, pipeline=pipeline)

    assert [call[0] for call in pipeline.calls] == [str(a), str(c)]
    assert pipeline.calls[0][1] == {"_data": f"hist_{a}"}
    assert result == {"indexed": 2, "duplicate": 0, "failed": 0,
                      "skipped_malformed": 0, "skipped_empty": 0}


@pytest.mark.asyncio
async def test_missing_counts_results_by_status() -> None:
    ids = _ids(4)
    events = FakeEventStream([_source_document(x) for x in ids])
    pipeline = ScriptedPipeline(
        ProcessStatus.INDEXED, ProcessStatus.DUPLICATE,
        ProcessStatus.DEAD_LETTERED, ProcessStatus.FAILED)

    result = await replay_missing(events=events, indexed_ids=set(), pipeline=pipeline)

    assert result["indexed"] == 1
    assert result["duplicate"] == 1
    assert result["failed"] == 2


@pytest.mark.asyncio
async def test_missing_counts_empty_text_separately_from_failures() -> None:
    # Пост без тексту ніколи не індексується — це не збій, що його треба «лікувати».
    events = FakeEventStream([_source_document(x) for x in _ids(1)])
    pipeline = ScriptedPipeline(ProcessStatus.SKIPPED_EMPTY)

    result = await replay_missing(events=events, indexed_ids=set(), pipeline=pipeline)

    assert result["skipped_empty"] == 1
    assert result["failed"] == 0


@pytest.mark.asyncio
async def test_missing_skips_and_counts_malformed_documents() -> None:
    good, bad = _ids(2)
    broken = {"_id": bad, "source_type": "telegram"}  # немає text/collected_at
    events = FakeEventStream([_source_document(good), broken])
    pipeline = ScriptedPipeline()

    result = await replay_missing(events=events, indexed_ids=set(), pipeline=pipeline)

    assert result["skipped_malformed"] == 1
    assert result["indexed"] == 1
    assert [call[0] for call in pipeline.calls] == [str(good)]


@pytest.mark.asyncio
async def test_missing_limit_caps_processed_events_not_skipped_ones() -> None:
    a, b, c, d = _ids(4)
    events = FakeEventStream([_source_document(x) for x in (a, b, c, d)])
    pipeline = ScriptedPipeline()

    result = await replay_missing(
        events=events, indexed_ids={str(a)}, pipeline=pipeline, limit=2)

    assert [call[0] for call in pipeline.calls] == [str(b), str(c)]
    assert result["indexed"] == 2


@pytest.mark.asyncio
async def test_missing_survives_a_pipeline_exception() -> None:
    a, b = _ids(2)
    events = FakeEventStream([_source_document(x) for x in (a, b)])
    pipeline = ScriptedPipeline(RuntimeError("boom"), ProcessStatus.INDEXED)

    result = await replay_missing(events=events, indexed_ids=set(), pipeline=pipeline)

    assert result["failed"] == 1
    assert result["indexed"] == 1


@pytest.mark.asyncio
async def test_missing_with_everything_indexed_does_nothing() -> None:
    ids = _ids(3)
    events = FakeEventStream([_source_document(x) for x in ids])
    pipeline = ScriptedPipeline()

    result = await replay_missing(
        events=events, indexed_ids={str(x) for x in ids}, pipeline=pipeline)

    assert pipeline.calls == []
    assert result["indexed"] == 0
