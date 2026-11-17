"""
Replay dead-letter: переобробити відкладені події через повний конвеєр.

Використання:
    python -m shipper.replay [--limit N]            # DLQ
    python -m shipper.replay --missing [--limit N]  # події, яких немає в індексі

Режим --missing (див. replay_missing) дообробляє сирі події, втрачені до
виправлення схеми ембедингів. Обидва режими спершу звіряють схему індексу.

Режим DLQ читає нерозв'язані записи DLQ, дістає сиру подію з Mongo,
проганяє конвеєр наново (ембединг → екстракція → верифікація → replace у Manticore)
і при успіху позначає запис DLQ як відтворений.

``replace`` у Manticore + ``_stable_id(mongo_id)`` роблять replay ідемпотентним.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone

from bson import ObjectId
import motor.motor_asyncio
import structlog

from .config import settings
from .dead_letter import PIPELINE_REVISION, mark_replayed
from .embedding_scheme import check_index_scheme
from .mongo_source import STATE_COLLECTION, MongoSource
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


async def replay_missing(
    *,
    events,
    indexed_ids: set[str],
    pipeline,
    limit: int | None = None,
) -> dict:
    """Дообробити сирі події, яких НЕМАЄ в індексі (втрачені до виправлення схеми
    ембедингів: HTTP 400 на довгих постах, хибні семантичні «дублікати»).

    Проходить raw_events за зростанням `_id`, пропускає ті, чий str(_id) уже є в
    `indexed_ids`, решту проганяє через повний конвеєр. Ідемпотентно: replace у
    Manticore за стабільним id. `limit` обмежує кількість РЕАЛЬНО оброблених
    подій (пропущені проіндексовані не рахуються).

    Повертає {"indexed", "duplicate", "failed", "skipped_malformed",
    "skipped_empty"}. `skipped_empty` — пости без тексту: вони ніколи не
    індексуються, і видавати їх за збій було б брехнею."""
    counts = {"indexed": 0, "duplicate": 0, "failed": 0,
              "skipped_malformed": 0, "skipped_empty": 0}
    processed = 0

    cursor = events.find({"direct_indexed_at": {"$exists": False}}).sort("_id", 1)
    async for doc in cursor:
        mongo_id = str(doc.get("_id"))
        if mongo_id in indexed_ids:
            continue
        if limit and processed >= limit:
            break

        try:
            event = MongoSource._to_raw_event(doc)
        except Exception as exc:
            counts["skipped_malformed"] += 1
            logger.warning("replay_missing_malformed", mongo_id=mongo_id, error=str(exc))
            continue

        processed += 1
        try:
            result = await pipeline.process(event, {"_data": f"hist_{mongo_id}"})
        except Exception:
            counts["failed"] += 1
            logger.exception("replay_missing_pipeline_error", mongo_id=mongo_id)
            continue

        if result.status is ProcessStatus.INDEXED:
            counts["indexed"] += 1
        elif result.status is ProcessStatus.DUPLICATE:
            counts["duplicate"] += 1
        elif result.status is ProcessStatus.SKIPPED_EMPTY:
            counts["skipped_empty"] += 1
        else:
            counts["failed"] += 1
            logger.warning("replay_missing_failed", mongo_id=mongo_id,
                           status=result.status, error=result.error)
    return counts


async def _main() -> int:
    """CLI entry point. Код виходу: 0 — готово, 1 — відмова (індекс іншої схеми ембедингів)."""
    parser = argparse.ArgumentParser(description="Replay dead-lettered events")
    parser.add_argument("--limit", type=int, default=None, help="Max events to replay")
    parser.add_argument(
        "--missing", action="store_true",
        help="Замість DLQ: дообробити сирі події, яких немає в індексі Manticore",
    )
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
    from .dedup import EmbeddingClient, build_deduplicator
    from .extractor import OllamaExtractor
    from .manticore_sink import ManticoreSink
    from .pipeline import Pipeline
    from .verifier import EntityVerifier, MitreAttackKB, NVDClient

    http_client = httpx.AsyncClient(timeout=None)
    manticore_sink = ManticoreSink(dead_letter_collection=dlq_col)
    await manticore_sink.setup_schema()

    # Той самий гард, що в main.py: не дописуємо вектори нової схеми в індекс зі
    # старими (різні простори). Спершу перевірка — до flusher-а й до завантаження
    # MITRE, щоб відмова нічого зайвого не тягнула з мережі.
    state_col = mongo_client[settings.mongo_db][STATE_COLLECTION]
    if not await check_index_scheme(manticore_sink, state_col, logger):
        await manticore_sink.stop()
        await http_client.aclose()
        mongo_client.close()
        return 1

    mitre_kb = MitreAttackKB()
    try:
        await mitre_kb.load(http_client)
    except Exception:
        logger.warning("mitre_load_failed")

    embedder = EmbeddingClient(client=http_client)
    extractor = OllamaExtractor(client=http_client)
    nvd_client = NVDClient(client=http_client)
    verifier = EntityVerifier(mitre_kb=mitre_kb, nvd_client=nvd_client)
    dedup = build_deduplicator()  # None, якщо SEMANTIC_DEDUP_ENABLED=false (дефолт)
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
        if args.missing:
            # Що вже в індексі — з самого Manticore (повний перелік, не обрізаний
            # до 1000), а не з Mongo: лише індекс каже, що реально проіндексовано.
            indexed_ids = await manticore_sink.list_mongo_ids()
            logger.info("replay_missing_start", indexed=len(indexed_ids))
            result = await replay_missing(
                events=events_col,
                indexed_ids=indexed_ids,
                pipeline=pipeline,
                limit=args.limit,
            )
            logger.info("replay_missing_complete", **result)
        else:
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
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
