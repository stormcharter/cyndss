"""
Точка входу шипера.

Запускається як: python -m shipper.main
Зупиняється за SIGINT / SIGTERM з дренажем черг.
"""

import asyncio
import json
import signal
from collections import deque

import httpx
import motor.motor_asyncio
import structlog

from .config import settings
from .dedup import EmbeddingClient, SemanticDeduplicator
from .extractor import OllamaExtractor
from .manticore_sink import ManticoreSink
from .mongo_source import MongoSource

from .pipeline import Pipeline
from .redact import safe_settings_dump
from .verifier import EntityVerifier, MitreAttackKB, NVDClient


def _setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )


async def serve_readiness(
    host: str,
    port: int,
    source: MongoSource,
) -> asyncio.AbstractServer:
    """Ендпойнт готовності процесу шипера — лише на loopback."""

    async def handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await asyncio.wait_for(reader.read(1024), timeout=2.0)
            readiness = source.readiness()
            status_code = 200 if readiness["ready"] else 503
            body = json.dumps(readiness, separators=(",", ":")).encode("utf-8")
            writer.write(
                (
                    f"HTTP/1.1 {status_code} {'OK' if status_code == 200 else 'Service Unavailable'}\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
                + body
            )
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    return await asyncio.start_server(handle, host, port)


async def run() -> None:
    _setup_logging()
    logger = structlog.get_logger("shipper.main")
    # Через safe_settings_dump, а не сирим: у mongo_uri тепер живе пароль.
    logger.info("starting", config=safe_settings_dump(settings.model_dump()))

    # HTTP-клієнт: спільний пул з'єднань
    http_client = httpx.AsyncClient(timeout=settings.shipper_http_timeout_seconds)

    # Завантажуємо MITRE STIX-бандл (один раз при старті) — не критично, якщо немає мережі.
    mitre_kb = MitreAttackKB()
    try:
        await mitre_kb.load(http_client)
    except Exception as e:
        logger.warning("mitre_load_failed", error=str(e), msg="continuing without MITRE enrichment")

    # Збирання компонентів
    embedder = EmbeddingClient(client=http_client)
    extractor = OllamaExtractor(client=http_client)
    nvd_client = NVDClient(client=http_client)
    verifier = EntityVerifier(mitre_kb=mitre_kb, nvd_client=nvd_client)
    dedup = SemanticDeduplicator()
    # dead-letter колекція для відмов Manticore
    mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
        settings.mongo_uri,
        tz_aware=True,
    )
    dead_letter_col = mongo_client[settings.mongo_db][settings.dead_letter_collection]
    sink = ManticoreSink(dead_letter_collection=dead_letter_col)
    await sink.setup_schema()
    sink.start_flusher()

    pipeline = Pipeline(
        embedder=embedder,
        extractor=extractor,
        verifier=verifier,
        dedup=dedup,
        sink=sink,
        dead_letter_collection=dead_letter_col,
        fast_mode=settings.fast_mode,
    )

    source = MongoSource()
    await source.acquire_lease()
    readiness_server = await serve_readiness(
        "127.0.0.1",
        settings.shipper_health_port,
        source,
    )


    # Graceful shutdown
    stop_event = asyncio.Event()

    def _on_signal():
        logger.info("shutdown_signal_received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    # Основний цикл: читаємо Change Stream і обробляємо

    stream_task = asyncio.create_task(_consume_stream(source, pipeline, stop_event))

    await stop_event.wait()
    logger.info("shutting_down")

    stream_task.cancel()
    await _wait_bounded(stream_task, "stream", logger)
    readiness_server.close()
    await readiness_server.wait_closed()

    await _wait_bounded(sink.stop(), "sink", logger)
    await _wait_bounded(source.close(), "mongo_source", logger)
    mongo_client.close()
    await _wait_bounded(http_client.aclose(), "http_client", logger)

    logger.info("shutdown_complete")


async def _wait_bounded(awaitable, component: str, logger) -> None:
    """Обмежити один крок зупинки, щоб недоступна залежність не підвісила вихід."""
    try:
        await asyncio.wait_for(
            awaitable,
            timeout=settings.shipper_shutdown_timeout_seconds,
        )
    except asyncio.CancelledError:
        if component != "stream":
            raise
    except asyncio.TimeoutError:
        logger.error("shutdown_step_timed_out", component=component)
    except Exception as exc:
        logger.error(
            "shutdown_step_failed",
            component=component,
            error=str(exc),
        )


async def _consume_stream(
    source: MongoSource,
    pipeline: Pipeline,
    stop_event: asyncio.Event,
) -> None:
    """Обробляти обмежену кількість подій паралельно, а чекпойнт просувати строго по порядку."""
    logger = structlog.get_logger("shipper.stream")
    logger.info("stream_starting")
    pending: deque[asyncio.Task] = deque()

    async def finish_oldest() -> bool:
        task = pending.popleft()
        event, resume_token, result = await task
        if result.acknowledged:
            await source.commit_token(resume_token, str(event.id))
            logger.debug(
                "token_committed",
                mongo_id=event.id,
                status=result.status,
            )
            return True
        logger.error(
            "checkpoint_not_advanced",
            mongo_id=event.id,
            status=result.status,
            error=result.error,
            error_class=result.error_class,
        )
        return False

    async def process(event, resume_token):
        result = await pipeline.process(event, resume_token)
        return event, resume_token, result

    try:
        async for event, resume_token in source.stream():
            if stop_event.is_set():
                logger.info("stream_interrupted")
                break
            pending.append(
                asyncio.create_task(
                    process(event, resume_token),
                    name=f"shipper-event-{event.id}",
                )
            )
            # Забирати всі вже завершені задачі по порядку, а не лише коли вікно
            # заповнене: інакше хвіст потоку ніколи не чекпойнтиться, доки не
            # прийдуть нові події (баг живучості при малому трафіку).
            while pending and pending[0].done():
                if not await finish_oldest():
                    stop_event.set()
                    break
            if len(pending) >= settings.pipeline_concurrency:
                if not await finish_oldest():
                    stop_event.set()
                    break

        while pending and not stop_event.is_set():
            if not await finish_oldest():
                stop_event.set()
                break
        if not stop_event.is_set():
            logger.error("stream_stopped_without_shutdown")
            stop_event.set()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("stream_crashed", error=str(exc))
        stop_event.set()
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(run())
