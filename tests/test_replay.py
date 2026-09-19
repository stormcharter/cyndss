from datetime import datetime, timezone

from bson import ObjectId
import pytest

from shipper.pipeline import ProcessResult, ProcessStatus
from shipper.replay import replay_dead_letters


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
