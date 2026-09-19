"""
Джерело подій: MongoDB Change Stream.

Принципове рішення: resume_token зберігається в самій Mongo, у службовій
колекції shipper_state. Це дає exactly-once-after-restart: якщо шипер
упав — він продовжить із того місця, де зупинився.

Вимога: Mongo має працювати як Replica Set (навіть одновузловий).
Без цього Change Streams не працюють.
"""

from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import asyncio
import random
import time
import uuid

import pymongo
import structlog
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from .config import settings
from .models import RawEvent, SourceMetadata, SourceType

logger = structlog.get_logger(__name__)

STATE_COLLECTION = "shipper_state"
STATE_DOC_ID = "main_pipeline"

class LeaseLostError(RuntimeError):
    """Процес більше не може довести ексклюзивне володіння орендою."""


class MalformedSourceEvent(ValueError):
    """Кидається замість мовчазного чекпойнту невалідного вихідного документа."""




class MongoSource:
    """Надійний потік insert-подій MongoDB з ексклюзивною поновлюваною орендою."""

    def __init__(self):
        self.client = AsyncIOMotorClient(settings.mongo_uri, tz_aware=True)
        self.db = self.client[settings.mongo_db]
        self.events = self.db[settings.mongo_collection]
        self.state = self.db[STATE_COLLECTION]
        self.owner_id = uuid.uuid4().hex
        self._lease_held = False
        self._lease_claimed = False
        self._lease_lost = False
        self._lease_lost_event = asyncio.Event()
        self._closed = False
        self._connected = False
        self._last_contact_at: float | None = None
        self._last_ack_at: float | None = None
        self._last_processed_id: str | None = None
        self._checkpoint_lock = asyncio.Lock()
        self._lease_task: asyncio.Task[None] | None = None

    @property
    def _lease_seconds(self) -> float:
        return float(settings.shipper_lease_seconds)

    @property
    def _stale_seconds(self) -> float:
        return float(settings.shipper_stale_seconds)

    @property
    def _reconnect_base_seconds(self) -> float:
        return float(settings.shipper_reconnect_base_seconds)

    @property
    def _reconnect_max_seconds(self) -> float:
        return float(settings.shipper_reconnect_max_seconds)

    async def acquire_lease(self) -> None:
        """Атомарно захопити ексклюзивну оренду конвеєра або впасти на старті."""
        if self._lease_held:
            return
        if self._closed:
            raise RuntimeError("cannot acquire a closed Mongo source")

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=self._lease_seconds)
        try:
            result = await self.state.update_one(
                {
                    "_id": STATE_DOC_ID,
                    "$or": [
                        {"lease_owner": self.owner_id},
                        {"lease_owner": {"$exists": False}},
                        {"lease_expires_at": {"$lte": now}},
                        {"lease_expires_at": {"$exists": False}},
                    ],
                },
                {
                    "$set": {
                        "lease_owner": self.owner_id,
                        "lease_expires_at": expires_at,
                        "lease_renewed_at": now,
                    },
                    "$setOnInsert": {"checkpoint_revision": 0},
                },
                upsert=True,
            )
        except pymongo.errors.DuplicateKeyError as exc:
            raise RuntimeError(
                "another Shipper worker owns the pipeline lease"
            ) from exc

        if (
            getattr(result, "matched_count", 0) != 1
            and getattr(result, "upserted_id", None) is None
        ):
            raise RuntimeError("another Shipper worker owns the pipeline lease")
        self._lease_claimed = True

        state_doc = await self.state.find_one(
            {"_id": STATE_DOC_ID, "lease_owner": self.owner_id}
        )
        if state_doc is None:
            raise LeaseLostError("Shipper pipeline lease was lost during acquisition")

        self._last_processed_id = state_doc.get("last_processed_id")
        self._lease_held = True
        self._lease_lost = False
        self._lease_lost_event.clear()
        if self._lease_task is None or self._lease_task.done():
            self._lease_task = asyncio.create_task(
                self._renew_lease_loop(),
                name="shipper-lease-renewal",
            )
        logger.info("shipper_lease_acquired", owner=self.owner_id)

    async def renew_lease(self) -> None:
        """Поновити лише ту оренду, що не сплила і досі належить цьому воркеру."""
        self._assert_lease()
        now = datetime.now(timezone.utc)
        result = await self.state.update_one(
            {
                "_id": STATE_DOC_ID,
                "lease_owner": self.owner_id,
                "lease_expires_at": {"$gt": now},
            },
            {
                "$set": {
                    "lease_expires_at": now
                    + timedelta(seconds=self._lease_seconds),
                    "lease_renewed_at": now,
                }
            },
        )
        if getattr(result, "matched_count", 0) != 1:
            self._mark_lease_lost()
            raise LeaseLostError("Shipper pipeline lease was lost")

    async def release_lease(self) -> None:
        """Звільнити запис лише цього власника — ніколи чужий."""
        lease_task = self._lease_task
        self._lease_task = None
        self._lease_held = False
        if lease_task is not None and lease_task is not asyncio.current_task():
            lease_task.cancel()
            try:
                await lease_task
            except asyncio.CancelledError:
                pass

        if self._lease_claimed:
            await self.state.update_one(
                {"_id": STATE_DOC_ID, "lease_owner": self.owner_id},
                {
                    "$unset": {
                        "lease_owner": "",
                        "lease_expires_at": "",
                        "lease_renewed_at": "",
                    }
                },
            )
            self._lease_claimed = False
            logger.info("shipper_lease_released", owner=self.owner_id)

    async def _renew_lease_loop(self) -> None:
        interval = max(self._lease_seconds / 3.0, 0.5)
        try:
            while self._lease_held and not self._closed:
                await asyncio.sleep(interval)
                if self._lease_held:
                    await self.renew_lease()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Мережева помилка робить володіння невідомим. Якщо продовжити до
            # наступної спроби, можна перетнутися з воркером, який захопив сплилу оренду.
            self._mark_lease_lost()
            logger.error("shipper_lease_renewal_failed", error=str(exc))

    def _mark_lease_lost(self) -> None:
        self._lease_held = False
        self._lease_lost = True
        self._connected = False
        self._lease_lost_event.set()

    def _assert_lease(self) -> None:
        if not self._lease_held or self._lease_lost:
            raise LeaseLostError("Shipper pipeline lease is not held")

    def mark_connected(self) -> None:
        self._connected = True
        self.record_contact()

    def mark_disconnected(self) -> None:
        self._connected = False

    def record_contact(self, now: float | None = None) -> None:
        self._last_contact_at = time.monotonic() if now is None else now

    def record_ack(self, now: float | None = None) -> None:
        self._last_ack_at = time.monotonic() if now is None else now

    def readiness(self, now: float | None = None) -> dict[str, object]:
        current = time.monotonic() if now is None else now
        contact_age = self._age(self._last_contact_at, current)
        ack_age = self._age(self._last_ack_at, current)
        stale = contact_age is None or contact_age > self._stale_seconds
        ready = (
            self._lease_held
            and not self._lease_lost
            and self._connected
            and not stale
        )
        return {
            "ready": ready,
            "lease_held": self._lease_held,
            "connected": self._connected,
            "last_contact_age_seconds": contact_age,
            "last_ack_age_seconds": ack_age,
            "stale": stale,
        }

    @staticmethod
    def _age(value: float | None, now: float) -> float | None:
        if value is None:
            return None
        return max(0.0, now - value)

    async def stream(self) -> AsyncIterator[tuple[RawEvent, dict]]:
        """Віддавати insert-події від надійного чекпойнту з дозбиранням історії без втрат."""
        if not self._lease_held:
            await self.acquire_lease()

        reconnect_delay = self._reconnect_base_seconds
        while not self._closed:
            self._assert_lease()
            state_doc = await self.state.find_one(
                {"_id": STATE_DOC_ID, "lease_owner": self.owner_id}
            )
            if state_doc is None:
                self._mark_lease_lost()
                raise LeaseLostError("Shipper pipeline lease record disappeared")

            last_processed_id = state_doc.get("last_processed_id")
            resume_token = state_doc.get("resume_token")
            if self._is_history_token(resume_token):
                resume_token = None
            elif resume_token is not None and not isinstance(resume_token, dict):
                logger.warning("stream_resume_token_malformed_reconciliation")
                await self._clear_resume_token()
                resume_token = None

            kwargs: dict[str, object] = {
                # direct_indexed_at ставить UI, коли аналітик індексує документ
                # напряму, повз конвеєр (фіча з гілки mosinnii-final, у нас поки
                # НЕ портована). Поле зараз ніхто не пише, тому умова завжди
                # істинна — лишаємо як готовий запобіжник від подвійної обробки,
                # щоб при перенесенні прямого індексування не ловити дублікати.
                "pipeline": [{
                    "$match": {
                        "operationType": "insert",
                        "fullDocument.direct_indexed_at": {"$exists": False},
                    }
                }],
                "full_document": "updateLookup",
                "max_await_time_ms": 1000,
            }
            if resume_token is not None:
                kwargs["resume_after"] = resume_token

            try:
                async with self.events.watch(**kwargs) as change_stream:
                    self.mark_connected()
                    if resume_token is None:
                        async for item in self._bootstrap_history(
                            change_stream,
                            last_processed_id,
                        ):
                            yield item

                    logger.info(
                        "stream_change_stream_started",
                        resumed=resume_token is not None,
                        last_processed_id=last_processed_id,
                    )
                    reconnect_delay = self._reconnect_base_seconds
                    while not self._closed:
                        self._assert_lease()
                        change = await change_stream.try_next()
                        self.record_contact()
                        if change is None:
                            continue
                        full = change.get("fullDocument")
                        if not isinstance(full, dict):
                            raise MalformedSourceEvent(
                                "insert change is missing fullDocument"
                            )
                        event = self._to_raw_event(full)
                        token = change.get("_id")
                        if not isinstance(token, dict):
                            raise MalformedSourceEvent(
                                "insert change is missing its resume token"
                            )
                        yield event, token
            except asyncio.CancelledError:
                raise
            except (LeaseLostError, MalformedSourceEvent):
                self.mark_disconnected()
                raise
            except Exception as exc:
                self.mark_disconnected()
                if self._lease_lost:
                    raise LeaseLostError("Shipper pipeline lease was lost") from exc
                if self._is_resume_token_error(exc):
                    logger.warning(
                        "stream_resume_token_invalid_reconciliation",
                        error=str(exc),
                        last_processed_id=last_processed_id,
                    )
                    await self._clear_resume_token()
                else:
                    logger.error("stream_change_stream_error", error=str(exc))
                delay = min(reconnect_delay, self._reconnect_max_seconds)
                jitter = random.uniform(0.0, min(delay, 1.0))
                await self._sleep_or_lease_loss(delay + jitter)
                reconnect_delay = min(delay * 2.0, self._reconnect_max_seconds)
            else:
                self.mark_disconnected()
                if not self._closed:
                    await asyncio.sleep(0)

    async def _bootstrap_history(
        self,
        change_stream,
        last_processed_id: str | None,
    ) -> AsyncIterator[tuple[RawEvent, dict]]:
        # Відкрити живий потік ДО читання межі — це закриває гонку: вставки після
        # цього моменту з'являться в потоці, навіть якщо їхній _id сортується
        # раніше за історичну межу.
        boundary_doc = await self.events.find_one(sort=[("_id", -1)])
        boundary_id = boundary_doc.get("_id") if boundary_doc else None
        history_filter = self._history_filter(last_processed_id, boundary_id)
        count = 0
        if history_filter is not None:
            cursor = self.events.find(history_filter).sort("_id", 1)
            async for doc in cursor:
                self._assert_lease()
                event = self._to_raw_event(doc)
                count += 1
                yield event, {"_data": f"hist_{event.id}"}
        logger.info(
            "stream_historical_done",
            count=count,
            history_boundary=str(boundary_id) if boundary_id else None,
        )

    async def _sleep_or_lease_loss(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._lease_lost_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            return
        raise LeaseLostError("Shipper pipeline lease was lost")

    async def _clear_resume_token(self) -> None:
        now = datetime.now(timezone.utc)
        result = await self.state.update_one(
            {
                "_id": STATE_DOC_ID,
                "lease_owner": self.owner_id,
                "lease_expires_at": {"$gt": now},
            },
            {"$set": {"resume_token": None, "updated_at": now}},
        )
        if getattr(result, "matched_count", 0) != 1:
            self._mark_lease_lost()
            raise LeaseLostError(
                "Shipper pipeline lease was lost while clearing checkpoint"
            )

    @staticmethod
    def _is_history_token(token: object) -> bool:
        return (
            isinstance(token, dict)
            and isinstance(token.get("_data"), str)
            and token["_data"].startswith("hist_")
        )

    @staticmethod
    def _history_filter(
        last_processed_id: str | None,
        boundary_id: object | None,
    ) -> dict | None:
        if boundary_id is None:
            return None
        bounds: dict[str, object] = {"$lte": boundary_id}
        if last_processed_id:
            try:
                bounds["$gt"] = ObjectId(last_processed_id)
            except Exception:
                bounds["$gt"] = last_processed_id
        return {
            "_id": bounds,
            "direct_indexed_at": {"$exists": False},
        }

    @staticmethod
    def _is_resume_token_error(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        if code in {260, 280, 286}:
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "resume token",
                "resumetoken",
                "change stream history",
                "history lost",
                "invalid resume",
            )
        )

    async def commit_token(
        self,
        token: dict | None,
        last_processed_id: str,
    ) -> None:
        """Закомітити одну підтверджену позицію під живою ексклюзивною орендою."""
        if not last_processed_id:
            raise ValueError("last_processed_id must not be empty")
        self._assert_lease()

        async with self._checkpoint_lock:
            now = datetime.now(timezone.utc)
            set_fields: dict[str, object] = {
                "last_processed_id": self._advance_last_id(last_processed_id),
                "updated_at": now,
            }
            if token is not None and not self._is_history_token(token):
                if not isinstance(token, dict):
                    raise ValueError("resume token must be a mapping")
                set_fields["resume_token"] = token
            result = await self.state.update_one(
                {
                    "_id": STATE_DOC_ID,
                    "lease_owner": self.owner_id,
                    "lease_expires_at": {"$gt": now},
                },
                {"$set": set_fields, "$inc": {"checkpoint_revision": 1}},
                upsert=False,
            )
            if getattr(result, "matched_count", 0) != 1:
                self._mark_lease_lost()
                raise LeaseLostError(
                    "Shipper pipeline lease was lost before checkpoint"
                )
            self._last_processed_id = str(set_fields["last_processed_id"])
            self.record_ack()

    def _advance_last_id(self, candidate: str) -> str:
        current = self._last_processed_id
        if current is None:
            return candidate
        try:
            return candidate if ObjectId(candidate) > ObjectId(current) else current
        except Exception:
            return candidate if candidate > current else current

    @staticmethod
    def _to_raw_event(doc: dict) -> RawEvent:
        """Розібрати вихідний документ, не вигадуючи відсутнє походження."""
        if not isinstance(doc, dict):
            raise MalformedSourceEvent("source document must be an object")
        try:
            doc_id = str(doc["_id"])
            text = doc["text"]
            collected_at = doc["collected_at"]
        except KeyError as exc:
            raise MalformedSourceEvent(
                f"source document is missing required field {exc.args[0]}"
            ) from exc
        if not isinstance(text, str):
            raise MalformedSourceEvent("source text must be a string")
        if len(text) > settings.shipper_max_event_text_chars:
            raise MalformedSourceEvent("source text exceeds configured size limit")
        if isinstance(collected_at, datetime) and collected_at.tzinfo is None:
            collected_at = collected_at.replace(tzinfo=timezone.utc)

        nested_source = doc.get("source")
        source_doc = nested_source if isinstance(nested_source, dict) else {}
        source_type_raw = source_doc.get(
            "source_type",
            doc.get("source_type"),
        )
        try:
            source_type = SourceType(source_type_raw)
        except ValueError as exc:
            raise MalformedSourceEvent(
                f"unsupported source_type {source_type_raw!r}"
            ) from exc

        try:
            return RawEvent(
                _id=doc_id,
                source=SourceMetadata(
                    source_type=source_type,
                    channel_id=source_doc.get(
                        "channel_id",
                        doc.get("channel_id"),
                    ),
                    channel_name=source_doc.get(
                        "channel_name",
                        doc.get("channel_name"),
                    ),
                    message_id=source_doc.get(
                        "message_id",
                        doc.get("message_id"),
                    ),
                    url=source_doc.get("url", doc.get("url")),
                ),
                text=text,
                collected_at=collected_at,
                raw_payload=doc.get("raw_payload", {}),
            )
        except Exception as exc:
            raise MalformedSourceEvent(f"invalid source document: {exc}") from exc

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.mark_disconnected()
        try:
            if self._lease_claimed or self._lease_task is not None:
                await self.release_lease()
        finally:
            self.client.close()
