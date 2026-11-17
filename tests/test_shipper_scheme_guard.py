"""
Точки входу шиппера (main, replay) не торкаються непереїханого індексу.

СЦЕНАРІЙ. Індекс містить вектори зі СИРОГО оригіналу, а шиппер уже рахує нові
з англійського «title. summary». Змішувати їх не можна (різні простори): шиппер
має на старті побачити розбіжність маркера, залогувати причину й вийти з
ненульовим кодом, НЕ беручи оренди й НЕ обробляючи жодної події. Те саме для
replay. Усе на фейках — без мережі, Mongo і Manticore.

Запуск:
    python -m pytest tests/test_shipper_scheme_guard.py
"""
import sys

import pytest

import shipper.main as main_module
import shipper.manticore_sink as sink_module
import shipper.replay as replay_module
import shipper.verifier as verifier_module
from shipper.config import settings
from shipper.embedding_scheme import MARKER_ID, current_scheme


class EmptyCursor:
    """Порожня DLQ: replay нічого не переобробляє."""

    def sort(self, *args):
        return self

    def limit(self, *args):
        return self

    def __aiter__(self):
        async def iterate():
            return
            yield  # noqa: B901 — зробити функцію асинхронним генератором

        return iterate()


class FakeStateCollection:
    def find(self, query):
        return EmptyCursor()

    def __init__(self, docs=None):
        self.docs = {d["_id"]: dict(d) for d in (docs or [])}

    async def find_one(self, query):
        doc = self.docs.get(query["_id"])
        return dict(doc) if doc is not None else None

    async def update_one(self, query, update, upsert=False):
        doc = self.docs.setdefault(query["_id"], {"_id": query["_id"]})
        doc.update(update.get("$set", {}))


class FakeDatabase:
    def __init__(self, collections):
        self.collections = collections

    def __getitem__(self, name):
        return self.collections.setdefault(name, FakeStateCollection())


class FakeMongoClient:
    """Підміна motor: `client[db][collection]`; спільний стан між викликами."""

    collections: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, db_name):
        return FakeDatabase(FakeMongoClient.collections)

    def close(self):
        pass


class FakeMitre:
    loads = 0

    async def load(self, client):
        FakeMitre.loads += 1
        return None


class FakeSink:
    documents = 0
    ids: set = set()
    list_calls = 0
    started = False
    stopped = False

    def __init__(self, *args, **kwargs):
        FakeSink.started = False
        FakeSink.stopped = False

    async def setup_schema(self):
        return None

    async def count_documents(self):
        return FakeSink.documents

    async def list_mongo_ids(self):
        FakeSink.list_calls += 1
        return set(FakeSink.ids)

    def start_flusher(self):
        FakeSink.started = True

    async def stop(self):
        FakeSink.stopped = True


class ForbiddenSource:
    """Якщо шиппер дійшов до оренди — гард не спрацював."""

    def __init__(self, *args, **kwargs):
        pass

    async def acquire_lease(self):
        raise AssertionError("оренду взято, хоча індекс іншої схеми")


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeMongoClient.collections = {}
    FakeSink.documents = 0
    FakeSink.ids = set()
    FakeSink.list_calls = 0
    FakeMitre.loads = 0
    monkeypatch.setattr(main_module.motor.motor_asyncio, "AsyncIOMotorClient", FakeMongoClient)
    monkeypatch.setattr(replay_module.motor.motor_asyncio, "AsyncIOMotorClient", FakeMongoClient)
    monkeypatch.setattr(main_module, "MitreAttackKB", FakeMitre)
    monkeypatch.setattr(verifier_module, "MitreAttackKB", FakeMitre)
    monkeypatch.setattr(main_module, "ManticoreSink", FakeSink)
    monkeypatch.setattr(sink_module, "ManticoreSink", FakeSink)
    monkeypatch.setattr(main_module, "MongoSource", ForbiddenSource)
    monkeypatch.setattr(sys, "argv", ["shipper.replay"])


def _state():
    return FakeMongoClient.collections.setdefault("shipper_state", FakeStateCollection())


@pytest.mark.asyncio
async def test_main_refuses_to_start_on_unmigrated_index():
    FakeSink.documents = 1009  # індекс непорожній, маркера немає

    code = await main_module.run()

    assert code == 1
    assert FakeSink.started is False  # flusher не стартував: нічого не пишемо
    assert FakeSink.stopped is True   # ресурси прибрано


@pytest.mark.asyncio
async def test_refusal_happens_before_downloading_mitre():
    """Під `restart: unless-stopped` відмова стартувати означає цикл перезапусків.
    Якщо перевірка схеми стоїть ПІСЛЯ завантаження STIX-бандла MITRE, кожен виток
    циклу тягне його з мережі. Тому відмова має статися раніше."""
    FakeSink.documents = 1009

    assert await main_module.run() == 1
    assert await replay_module._main() == 1
    assert FakeMitre.loads == 0


@pytest.mark.asyncio
async def test_main_refuses_when_marker_is_for_another_scheme():
    FakeSink.documents = 1009
    FakeMongoClient.collections["shipper_state"] = FakeStateCollection(
        [{"_id": MARKER_ID, "scheme": "nomic-embed-text:raw/v0"}])

    assert await main_module.run() == 1


@pytest.mark.asyncio
async def test_replay_refuses_on_unmigrated_index():
    FakeSink.documents = 1009

    code = await replay_module._main()

    assert code == 1
    assert FakeSink.started is False


@pytest.mark.asyncio
async def test_replay_passes_the_guard_on_a_fresh_index_and_writes_the_marker():
    FakeSink.documents = 0

    code = await replay_module._main()

    assert code == 0
    assert _state().docs[MARKER_ID]["scheme"] == current_scheme(settings)


# --- replay --missing ---------------------------------------------------------
@pytest.fixture
def recorded_missing(monkeypatch):
    calls = []

    async def fake_replay_missing(**kwargs):
        calls.append(kwargs)
        return {"indexed": 0, "duplicate": 0, "failed": 0, "skipped_malformed": 0}

    monkeypatch.setattr(replay_module, "replay_missing", fake_replay_missing)
    return calls


@pytest.mark.asyncio
async def test_missing_cli_feeds_the_indexed_ids_from_manticore(monkeypatch, recorded_missing):
    monkeypatch.setattr(sys, "argv", ["shipper.replay", "--missing", "--limit", "7"])
    FakeSink.documents = 3
    FakeSink.ids = {"a", "b", "c"}
    FakeMongoClient.collections["shipper_state"] = FakeStateCollection(
        [{"_id": MARKER_ID, "scheme": current_scheme(settings)}])

    code = await replay_module._main()

    assert code == 0
    assert recorded_missing[0]["indexed_ids"] == {"a", "b", "c"}
    assert recorded_missing[0]["limit"] == 7


@pytest.mark.asyncio
async def test_missing_cli_runs_the_scheme_guard_first(monkeypatch, recorded_missing):
    monkeypatch.setattr(sys, "argv", ["shipper.replay", "--missing"])
    FakeSink.documents = 1009  # маркера немає

    code = await replay_module._main()

    assert code == 1
    assert recorded_missing == []
    assert FakeSink.list_calls == 0


@pytest.mark.asyncio
async def test_plain_replay_does_not_touch_the_missing_path(recorded_missing):
    FakeSink.documents = 0

    assert await replay_module._main() == 0
    assert recorded_missing == []
    assert FakeSink.list_calls == 0
