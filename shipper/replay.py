"""
Replay dead-letter: переобробити відкладені події через повний конвеєр.

Використання:
    python -m shipper.replay [--limit N]

Читає нерозв'язані записи DLQ, дістає сиру подію з Mongo,
проганяє конвеєр наново (ембединг → екстракція → верифікація → replace у Manticore)
і при успіху позначає запис DLQ як відтворений.

``replace`` у Manticore + ``_stable_id(mongo_id)`` роблять replay ідемпотентним.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from bson import ObjectId
import motor.motor_asyncio
import structlog

from .config import settings
from .dead_letter import PIPELINE_REVISION, mark_replayed
from .mongo_source import MongoSource
from .pipeline import ProcessStatus

logger = structlog.get_logger(__name__)


async def _find_source_event(events, mongo_id: str) -> dict | None:
    """Знайти вихідну подію за звичайним ObjectId; для старих записів — за рядком."""
    if ObjectId.is_valid(mongo_id):
        document = await events.find_one({"_id": ObjectId(mongo_id)})
        if document is not None:
            return document
    return await events.find_one({"_id": mongo_id})


async def _record_replay_failure(collection, entry_id, error: str) -> None:
    """Лишити невдалий replay нерозв'язаним і зафіксувати останню спробу."""
    await collection.update_one(
        {"_id": entry_id, "replayed_at": None},
        {
            "$inc": {"replay_attempts": 1},
            "$set": {
                "last_replay_at": datetime.now(timezone.utc),
                "last_replay_error": error[:2000],
            },
        },
    )


async def replay_dead_letters(
    *,
    collection,
    events,
    pipeline,
    limit: int | None = None,
    revision: str = PIPELINE_REVISION,
) -> dict:
    """Прогнати відкладені події через повний конвеєр.

    Повертає {"replayed": int, "failed": int}.
    """
    query = {"replayed_at": None, "pipeline_revision": revision}
    cursor = collection.find(query).sort("failed_at", 1)
    if limit:
        cursor = cursor.limit(limit)

    done = 0
    failed = 0

    async for entry in cursor:
        mongo_id = entry["mongo_id"]
        doc = await _find_source_event(events, mongo_id)
        if doc is None:
            error = "source event missing"
            await _record_replay_failure(collection, entry["_id"], error)
            logger.warning("replay_source_missing", mongo_id=mongo_id)
            failed += 1
            continue

        try:
            event = MongoSource._to_raw_event(doc)
        except Exception as exc:
            error = f"malformed source event: {exc}"
            await _record_replay_failure(collection, entry["_id"], error)
            logger.warning("replay_malformed_event", mongo_id=mongo_id, error=str(exc))
            failed += 1
            continue

        resume_token = {"_data": f"hist_{mongo_id}"}
        try:
            result = await pipeline.process(event, resume_token)
        except Exception as exc:
            error = f"pipeline raised during replay: {exc}"
            await _record_replay_failure(collection, entry["_id"], error)
            logger.exception("replay_pipeline_error", mongo_id=mongo_id)
            failed += 1
            continue

        if result.status in {ProcessStatus.INDEXED, ProcessStatus.DUPLICATE}:
            await mark_replayed(collection, mongo_id, revision)
            done += 1
            logger.info("replay_success", mongo_id=mongo_id, status=result.status)
            continue

        error = result.error or f"unresolved replay status: {result.status.value}"
        await _record_replay_failure(collection, entry["_id"], error)
        failed += 1
        logger.warning(
            "replay_failed",
            mongo_id=mongo_id,
            status=result.status,
            error=error,
        )

    return {"replayed": done, "failed": failed}


async def _main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Replay dead-lettered events")
    parser.add_argument("--limit", type=int, default=None, help="Max events to replay")
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )

    mongo_client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongo_uri)
    dlq_col = mongo_client[settings.mongo_db][settings.dead_letter_collection]
    events_col = mongo_client[settings.mongo_db][settings.mongo_collection]

    # Збираємо конвеєр (те саме, що в main.py, але простіше — без потоку)
    import httpx
    from .dedup import EmbeddingClient, SemanticDeduplicator
    from .extractor import OllamaExtractor
    from .manticore_sink import ManticoreSink
    from .pipeline import Pipeline
    from .verifier import EntityVerifier, MitreAttackKB, NVDClient

    http_client = httpx.AsyncClient(timeout=None)
    mitre_kb = MitreAttackKB()
    try:
        await mitre_kb.load(http_client)
    except Exception:
        logger.warning("mitre_load_failed")

    embedder = EmbeddingClient(client=http_client)
    extractor = OllamaExtractor(client=http_client)
    nvd_client = NVDClient(client=http_client)
    verifier = EntityVerifier(mitre_kb=mitre_kb, nvd_client=nvd_client)
    dedup = SemanticDeduplicator()

    manticore_sink = ManticoreSink(dead_letter_collection=dlq_col)
    await manticore_sink.setup_schema()
    manticore_sink.start_flusher()

    pipeline = Pipeline(
        embedder=embedder,
        extractor=extractor,
        verifier=verifier,
        dedup=dedup,
        sink=manticore_sink,
        dead_letter_collection=dlq_col,
    )

    try:
        result = await replay_dead_letters(
            collection=dlq_col,
            events=events_col,
            pipeline=pipeline,
            limit=args.limit,
        )
        logger.info("replay_complete", **result)
    finally:
        await manticore_sink.stop()
        await http_client.aclose()
        mongo_client.close()


if __name__ == "__main__":
    asyncio.run(_main())
