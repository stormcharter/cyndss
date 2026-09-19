import asyncio
import time
from dataclasses import dataclass
from enum import Enum

import structlog
from .config import settings
from .dead_letter import write_dead_letter, PIPELINE_REVISION
from .dedup import EmbeddingClient, SemanticDeduplicator
from .extractor import OllamaExtractor
from .grounding import (
    ACCEPT_LITERAL,
    ACCEPT_QUOTED,
    filter_grounded,
    harvest_literal_ids,
)
from .models import EnrichedDocument, ExtractedEntities, RawEvent, VerificationStatus
from .verifier import EntityVerifier

logger = structlog.get_logger(__name__)

# Який рівень заземлення приймаємо в кожному режимі (див. settings.grounding_mode).
_ACCEPT_BY_MODE = {
    "literal": ACCEPT_LITERAL,
    "quoted": ACCEPT_QUOTED,
}


class ProcessStatus(str, Enum):
    INDEXED = "indexed"
    DUPLICATE = "duplicate"
    SKIPPED_EMPTY = "skipped_empty"
    DEAD_LETTERED = "dead_lettered"
    FAILED = "failed"


@dataclass
class ProcessResult:
    mongo_id: str
    status: ProcessStatus
    error: str | None = None
    error_class: str | None = None

    @property
    def acknowledged(self) -> bool:
        """True, якщо чекпойнт можна безпечно просунути (запис надійний)."""
        return self.status is not ProcessStatus.FAILED


class Pipeline:
    def __init__(
        self,
        embedder,
        extractor,
        verifier,
        dedup,
        sink,
        dead_letter_collection=None,
        fast_mode: bool = False,
    ):
        self.embedder = embedder
        self.extractor = extractor
        self.verifier = verifier
        self.dedup = dedup
        self.sink = sink
        self._dead_letter_collection = dead_letter_collection
        self.fast_mode = fast_mode

    async def process(self, event: RawEvent, resume_token: dict | None = None) -> ProcessResult:
        if not event.text or not event.text.strip() or len(event.text.strip()) < 5:
            logger.info("pipeline_skipped_empty_text", mongo_id=event.id)
            return ProcessResult(event.id, ProcessStatus.SKIPPED_EMPTY)

        last_exc: Exception | None = None
        max_retries = settings.pipeline_max_retries

        for attempt in range(1, max_retries + 1):
            try:
                return await self._process_unsafe(event)
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    delay = settings.pipeline_retry_base_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "pipeline_retry",
                        mongo_id=event.id,
                        attempt=attempt,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

        # Обмежені ретраї вичерпано — пробуємо надійно відкласти в dead-letter.
        stage = self._stage_of(last_exc)
        source_pos = None
        if resume_token and "_data" in resume_token:
            source_pos = str(resume_token["_data"])

        parked = False
        if self._dead_letter_collection is not None:
            parked = await write_dead_letter(
                self._dead_letter_collection,
                mongo_id=event.id,
                stage=stage,
                error=str(last_exc),
                error_class=type(last_exc).__name__,
                attempts=max_retries,
                source_position=source_pos,
            )

        if parked:
            return ProcessResult(
                event.id,
                ProcessStatus.DEAD_LETTERED,
                error=str(last_exc),
                error_class=type(last_exc).__name__,
            )
        return ProcessResult(
            event.id,
            ProcessStatus.FAILED,
            error=str(last_exc),
            error_class=type(last_exc).__name__,
        )

    async def _process_unsafe(self, event: RawEvent) -> ProcessResult:
        log = logger.bind(mongo_id=event.id, source=event.source.source_type.value)
        t0 = time.monotonic()

        embedding = await self.embedder.embed(event.text)
        t1 = time.monotonic()

        if self.fast_mode:
            # Швидкий шлях для масового корпусу: відібраний вихідний текст уже авторитетний;
            # лишаємо ембединг/дедуп/push, пропускаємо LLM-екстракцію та верифікацію.
            extracted = ExtractedEntities(
                title=(event.text.strip().splitlines() or [""])[0][:200] or "Untitled",
            )
            v_cves, v_ttps, v_iocs, v_rels = [], [], [], []
            t2 = t3 = t1  # екстракцію/верифікацію пропущено; поля латентності в логах лишаємо визначеними
        else:
            extracted = await self.extractor.extract(event.text)
            # Гейт СТОЇТЬ ДО верифікації навмисно: NVD/MITRE підтверджують лише
            # існування ідентифікатора й вигаданий T1190 проходять наскрізь.
            # Заодно не витрачаємо зовнішні виклики на завідомий шум.
            extracted = self._ground(extracted, event.text, log)
            t2 = time.monotonic()

            v_cves, v_ttps, v_iocs, v_rels = await self.verifier.verify(extracted)
            t3 = time.monotonic()
        duplicate = self.dedup.find_duplicate(embedding)
        if duplicate.is_duplicate:
            log.info(
                "pipeline_duplicate",
                duplicate_of=duplicate.duplicate_of_mongo_id,
                similarity=duplicate.similarity,
                embed_latency_ms=round((t1 - t0) * 1000, 1),
            )
            return ProcessResult(event.id, ProcessStatus.DUPLICATE)

        v_cves_list = [c for c in v_cves if c.status == VerificationStatus.VERIFIED]
        v_ttps_list = [t for t in v_ttps if t.status == VerificationStatus.VERIFIED]

        doc = EnrichedDocument(
            mongo_id=event.id,
            source_type=event.source.source_type,
            channel_name=event.source.channel_name,
            url=event.source.url,
            collected_at=event.collected_at,
            text=event.text,
            title=extracted.title,
            language=extracted.detected_language,
            summary=extracted.summary or "Summary unavailable",
            embedding=embedding.tolist(),
            verified_cves=v_cves,
            verified_ttps=v_ttps,
            verified_iocs=v_iocs,
            verified_relations=v_rels,
            cve_count_verified=len(v_cves_list),
            ttp_count_verified=len(v_ttps_list),
            ioc_count=len(v_iocs),
        )

        ack = await self.sink.push(doc)
        t4 = time.monotonic()

        if ack.status == "ok":
            self.dedup.remember_indexed(event.id, event.text, embedding)
            log.info(
                "pipeline_success",
                cve_verified=len(v_cves_list),
                ttp_verified=len(v_ttps_list),
                embed_latency_ms=round((t1 - t0) * 1000, 1),
                extract_latency_ms=round((t2 - t1) * 1000, 1),
                verify_latency_ms=round((t3 - t2) * 1000, 1),
                push_latency_ms=round((t4 - t3) * 1000, 1),
                total_latency_ms=round((t4 - t0) * 1000, 1),
            )
            return ProcessResult(event.id, ProcessStatus.INDEXED)

        if ack.status == "dead_lettered":
            log.warning(
                "pipeline_sink_dead_lettered",
                mongo_id=event.id,
                error=ack.error,
            )
            return ProcessResult(
                event.id,
                ProcessStatus.DEAD_LETTERED,
                error=ack.error,
                error_class="ManticoreBulkError",
            )

        # ack.status == "failed" — DLQ сінка теж не вдався; просувати чекпойнт не можна.
        raise RuntimeError(f"sink not acknowledged: {ack.error}")

    @staticmethod
    def _ground(extracted, text: str, log):
        """Лишає в мітках тільки те, що спирається на документ.

        Дві дії в одному місці, бо критерій у них спільний — присутність у
        тексті: (1) відкидаємо назване моделлю, але не заземлене; (2) додаємо
        ідентифікатори, які стоять у тексті дослівно, навіть якщо модель їх не
        згадала. Повертає КОПІЮ `extracted`: title/summary та мова лишаються як
        були, міняються лише списки CVE/TTP."""
        mode = (settings.grounding_mode or "quoted").strip().lower()
        if mode == "off":
            return extracted

        # .get з фолбеком недосяжний у проді: Settings приймає лише три значення
        # (Literal-тип), тож невідомий режим — це тільки monkeypatch у тестах.
        accept = _ACCEPT_BY_MODE.get(mode, ACCEPT_LITERAL)
        cves, dropped_cves = filter_grounded(
            extracted.extracted_cve_ids, extracted.cve_contexts, text, accept)
        ttps, dropped_ttps = filter_grounded(
            extracted.mitre_tactics_techniques, extracted.ttp_contexts, text, accept)

        # Символьний добір: ідентифікатори, що стоять у тексті дослівно, беремо
        # НЕЗАЛЕЖНО від моделі. Виміряно на веб-корпусі: 16 статей із 45 містять
        # CVE дослівно, а модель називає одиниці — гейт сам по собі лише ріже,
        # тут ми повертаємо втрачене. Галюцинація неможлива: рядок узято з
        # документа, існування підтвердить верифікатор.
        harvested_cves, harvested_ttps = harvest_literal_ids(text)
        added = 0
        for pool, found in ((cves, harvested_cves), (ttps, harvested_ttps)):
            for ident in found:
                if ident not in pool:
                    pool.append(ident)
                    added += 1

        if dropped_cves or dropped_ttps or added:
            log.info(
                "grounding_dropped",
                mode=mode,
                cves=[f"{i}:{g.value}" for i, g in dropped_cves],
                ttps=[f"{i}:{g.value}" for i, g in dropped_ttps],
                harvested=added,
            )
        return extracted.model_copy(update={
            "extracted_cve_ids": cves,
            "mitre_tactics_techniques": ttps,
        })

    @staticmethod
    def _stage_of(exc: Exception | None) -> str:
        """Віднести виняток до назви стадії конвеєра для DLQ."""
        if exc is None:
            return "pipeline"
        name = type(exc).__name__
        msg = str(exc).lower()
        if "embed" in msg or "dimension" in msg:
            return "embedding"
        if "extract" in msg or "ollama" in name.lower():
            return "extraction"
        if "verif" in msg or "nvd" in msg or "mitre" in msg:
            return "verification"
        return "pipeline"
