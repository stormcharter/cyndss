"""
Сінк у Manticore: мікробатчинг через HTTP-ендпойнт /bulk.

Архітектура: вхідні EnrichedDocument кладуться в обмежену asyncio.Queue.
Окрема корутина-flusher витягує їх пачками по batch_size (або все, що
готове) і надсилає в Manticore одним запитом у форматі NDJSON.

На відміну від попередньої версії, push() блокується доти, доки документ
не буде надійно записано в Manticore (або надійно відкладено в dead-letter).
Це гарантує, що checkpoint просувається лише після підтвердження (ack).

Формат /bulk у Manticore — Elasticsearch-сумісний: на кожен документ
два рядки JSON, розділені \\n. Перший — команда (insert/replace), другий
— сам документ.
"""

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field

import httpx
import structlog
from motor.motor_asyncio import AsyncIOMotorCollection

from .config import settings
from .dead_letter import write_dead_letter
from .models import EnrichedDocument
from .schema import build_create_table_sql
from lemmatizer import lemmatize_text  # спільна нормалізація для індексу та запиту

logger = structlog.get_logger(__name__)


# DDL для таблиці — один раз при старті. Сама схема (включно з морфологією пошуку
# та розмірністю KNN=768 під nomic-embed-text) живе в shipper/schema.py як
# чиста функція — так її можна тестувати без .env. Змінюєш модель ембедера
# або опції пошуку — правиш schema.py і перестворюєш таблицю (reset_manticore.py).
CREATE_TABLE_SQL = build_create_table_sql(settings.manticore_table)


class ManticoreBulkMismatch(Exception):
    """Manticore /bulk повернув іншу кількість елементів, ніж було надіслано."""

    def __init__(self, expected: int, got: int):
        super().__init__(f"expected {expected} items, got {got}")
        self.expected = expected
        self.got = got


@dataclass
class SinkAck:
    """Flusher розв'язує його рівно один раз на кожен переданий документ.

    Значення status:
    - "ok": надійно записано в Manticore (або елемент пройшов після часткової відмови)
    - "dead_lettered": надійно відкладено в DLQ (Manticore відмовив, запис у DLQ вдався)
    - "failed": НЕ надійно (запис у DLQ теж не вдався); споживач НЕ має комітити
    """

    event: asyncio.Event = field(default_factory=asyncio.Event)
    status: str = "failed"
    error: str | None = None

    def resolve(self, status: str, error: str | None = None) -> None:
        if not self.event.is_set():
            self.status = status
            self.error = error
            self.event.set()


class ManticoreSink:
    """Черга і flusher для пакетного запису з ретраями та dead-letter.

    push() блокується до ack (межа надійності запису).
    """

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        dead_letter_collection: AsyncIOMotorCollection | None = None,
    ):
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=30.0)
        self.queue: asyncio.Queue[tuple[EnrichedDocument, SinkAck]] = asyncio.Queue(
            maxsize=settings.sink_queue_maxsize,
        )
        self._flusher_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._dead_letter_collection = dead_letter_collection
    # --- Публічний API -----------------------------------------------------

    async def setup_schema(self) -> None:
        """Створює таблицю, якщо її немає. Використовує SQL-ендпойнт."""
        response = await self.client.post(
            f"{settings.manticore_url}/sql?mode=raw",
            content=CREATE_TABLE_SQL,
            headers={"Content-Type": "text/plain"},
        )
        response.raise_for_status()
        logger.info("manticore_table_ready", table=settings.manticore_table)

    async def push(self, doc: EnrichedDocument) -> SinkAck:
        """Ставить документ у чергу й чекає на надійний результат: сінк або DLQ."""
        if self._stop_event.is_set():
            raise RuntimeError("Manticore sink is stopping")
        if self._flusher_task is None or self._flusher_task.done():
            raise RuntimeError("Manticore sink flusher is not running")
        ack = SinkAck()
        await self.queue.put((doc, ack))
        await ack.event.wait()
        return ack

    def start_flusher(self) -> None:
        if self._stop_event.is_set():
            raise RuntimeError("cannot restart a stopped Manticore sink")
        if self._flusher_task is None:
            self._flusher_task = asyncio.create_task(
                self._flush_loop(),
                name="manticore-batch-flusher",
            )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._flusher_task:
            await self._flusher_task
            self._flusher_task = None
        if self._owns_client:
            await self.client.aclose()

    @property
    def pending(self) -> int:
        """Глибина черги — видно через логи або метрики."""
        return self.queue.qsize()

    # --- Flusher ------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """Збирає до ``batch_size`` документів, поки не сплине дедлайн батча."""
        while not self._stop_event.is_set():
            try:
                first = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            batch: list[tuple[EnrichedDocument, SinkAck]] = [first]
            deadline = asyncio.get_running_loop().time() + settings.batch_timeout_seconds
            while len(batch) < settings.batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(
                        await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    )
                except asyncio.TimeoutError:
                    break
            logger.debug(
                "manticore_flush_batch",
                batch_size=len(batch),
                pending=self.pending,
            )
            try:
                await self._flush_with_retry(batch)
            except Exception:
                logger.exception("flush_loop_unhandled", batch_size=len(batch))
                for _, ack in batch:
                    if not ack.event.is_set():
                        ack.resolve("failed", "flush_loop_exception")

        batch: list[tuple[EnrichedDocument, SinkAck]] = []
        while not self.queue.empty():
            batch.append(self.queue.get_nowait())
        if batch:
            try:
                await self._flush_with_retry(batch)
            except Exception:
                logger.exception("flush_loop_drain_failed", batch_size=len(batch))
                for _, ack in batch:
                    if not ack.event.is_set():
                        ack.resolve("failed", "drain_exception")

    # --- Запис із ретраями + розбір по кожному елементу ---------------------

    async def _flush_with_retry(
        self,
        batch: list[tuple[EnrichedDocument, SinkAck]],
    ) -> None:
        pending = list(batch)
        last_errors: dict[str, str] = {}
        max_retries = settings.manticore_max_retries

        for attempt in range(1, max_retries + 1):
            docs = [doc for doc, _ in pending]
            try:
                statuses = await self._flush(docs)
                still_pending: list[tuple[EnrichedDocument, SinkAck]] = []
                for doc, ack in pending:
                    ok, error = statuses.get(
                        doc.mongo_id,
                        (False, "Manticore response omitted this document"),
                    )
                    if ok:
                        ack.resolve("ok")
                    else:
                        last_errors[doc.mongo_id] = error or "item failed"
                        still_pending.append((doc, ack))
                pending = still_pending
                if not pending:
                    return
            except (httpx.HTTPError, ManticoreBulkMismatch) as exc:
                error = str(exc)
                for doc, _ in pending:
                    last_errors[doc.mongo_id] = error

            if attempt < max_retries:
                delay = settings.manticore_retry_base_seconds * (2 ** (attempt - 1))
                delay += random.uniform(0, delay * 0.1)
                logger.warning(
                    "manticore_flush_retry",
                    attempt=attempt,
                    max_retries=max_retries,
                    delay_seconds=round(delay, 2),
                    failed_items=len(pending),
                )
                await asyncio.sleep(delay)

        logger.error(
            "manticore_flush_exhausted",
            failed_items=len(pending),
            attempts=max_retries,
        )
        for doc, ack in pending:
            error = last_errors.get(doc.mongo_id, "Manticore delivery failed")
            parked = False
            if self._dead_letter_collection is not None:
                parked = await write_dead_letter(
                    self._dead_letter_collection,
                    mongo_id=doc.mongo_id,
                    stage="manticore_bulk",
                    error=error,
                    error_class="ManticoreBulkError",
                    attempts=max_retries,
                )
            if parked:
                ack.resolve("dead_lettered", error)
            else:
                logger.error("dead_letter_write_failed", mongo_id=doc.mongo_id)
                ack.resolve("failed", error)

    # --- Сирий POST у Manticore + розбір по кожному елементу ----------------

    async def _flush(self, batch: list[EnrichedDocument]) -> dict[str, tuple[bool, str]]:
        """POST /bulk і статус по кожному елементу.

        Повертає {mongo_id: (is_ok, error_msg)} для кожного документа.
        Невідомі/нерозбірні елементи вважаються помилкою (безпечний бік).
        """
        lines: list[str] = []
        for doc in batch:
            doc_id = self._stable_id(doc.mongo_id)
            lines.append(
                json.dumps(
                    {
                        "replace": {
                            "table": settings.manticore_table,
                            "id": doc_id,
                            "doc": self._to_manticore_doc(doc),
                        }
                    },
                    default=str,
                )
            )
        body = "\n".join(lines) + "\n"
        response = await self.client.post(
            f"{settings.manticore_url}/bulk",
            content=body,
            headers={"Content-Type": "application/x-ndjson"},
        )
        response.raise_for_status()

        payload = response.json()
        items = payload.get("items") or []

        if len(items) != len(batch):
            # Деякі версії Manticore агрегують увесь bulk в один елемент
            # ({created: N, errors: false, skipped_lines: 0}) замість елемента
            # на кожен рядок. Приймаємо агрегат лише тоді, коли кожен рядок
            # зарахований як успішний запис; інакше відповідності довіряти
            # не можна і батч не має бути підтверджений.
            aggregate = items[0].get("bulk") if len(items) == 1 else None
            written = (
                sum(int(aggregate.get(k) or 0) for k in ("created", "updated", "replaced"))
                if isinstance(aggregate, dict)
                else 0
            )
            if (
                isinstance(aggregate, dict)
                and not payload.get("errors")
                and not payload.get("skipped_lines")
                and written == len(batch)
            ):
                return {doc.mongo_id: (True, "") for doc in batch}
            raise ManticoreBulkMismatch(len(batch), len(items))

        result: dict[str, tuple[bool, str]] = {}
        for doc, item in zip(batch, items):
            # Успіх Manticore: в елементі є ключ "bulk" зі статусом 201/200
            bulk = item.get("bulk")
            if isinstance(bulk, dict):
                status = bulk.get("status")
                if isinstance(status, int) and 200 <= status < 300:
                    result[doc.mongo_id] = (True, "")
                    continue
            # Відмова Manticore: в елементі є ключ "replace" з "error"
            replace = item.get("replace")
            if isinstance(replace, dict) and "error" in replace:
                err = replace["error"]
                msg = err.get("type") if isinstance(err, dict) else str(err)
                result[doc.mongo_id] = (False, msg or "item failed")
                continue
            # Невідома форма — вважаємо помилкою (безпечно: ретрай або DLQ)
            result[doc.mongo_id] = (False, f"unexpected item shape: {item}")

        logger.info("manticore_flush_ok", batch_size=len(batch))
        return result

    # --- Допоміжні ----------------------------------------------------------

    @staticmethod
    def _stable_id(mongo_id: str) -> int:
        """Повертає стабільне додатне знакове ціле для ідентифікатора джерела."""
        try:
            return int(mongo_id[-16:], 16)
        except ValueError:
            digest = hashlib.sha256(mongo_id.encode("utf-8")).digest()
            return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1

    @staticmethod
    def _to_manticore_doc(doc: EnrichedDocument) -> dict:
        """Перетворює EnrichedDocument на dict під схему Manticore."""
        return {
            "mongo_id": doc.mongo_id,
            "source_type": doc.source_type.value,
            "channel_name": doc.channel_name or "",
            "url": doc.url or "",
            "collected_at": int(doc.collected_at.timestamp()),
            "title": doc.title or "Untitled",
            "language": doc.language or "en",
            "content": lemmatize_text(doc.text),
            "summary": doc.summary or "",
            "embedding": doc.embedding,
            "verified_cve_ids": ",".join([
                c.cve_id for c in doc.verified_cves if c.status.value == "verified"
            ]),
            "verified_ttp_ids": ",".join([
                t.technique_id for t in doc.verified_ttps if t.status.value == "verified"
            ]),
            "cve_count_verified": doc.cve_count_verified,
            "ttp_count_verified": doc.ttp_count_verified,
            "ioc_count": doc.ioc_count,
        }
