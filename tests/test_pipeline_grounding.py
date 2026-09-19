"""
Гейт заземлення в конвеєрі: безпідставна мітка не доходить до верифікації.

ЧОМУ САМЕ ТУТ. Верифікація (NVD/MITRE) підтверджує лише ІСНУВАННЯ
ідентифікатора — вигаданий T1190 вона проходить наскрізь і лягає в індекс як
«verified». Тому відсів має стояти РАНІШЕ за верифікацію: так шум не потрапляє
в документ і заодно не витрачаються зовнішні виклики NVD.

Запуск:
    python -m pytest tests/test_pipeline_grounding.py
(потрібні pydantic/numpy/structlog — чистим stdlib цей файл не ганяється).
"""
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from shipper.config import Settings, settings
from shipper.dedup import SemanticDeduplicator
from shipper.models import ExtractedEntities, RawEvent, SourceMetadata, SourceType
from shipper.pipeline import Pipeline

TEXT = ("Дослідники повідомили про активну експлуатацію вразливості "
        "CVE-2026-3055 у публічному вебінтерфейсі поштового шлюзу.")


class StaticEmbedder:
    async def embed(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class ScriptedExtractor:
    """Віддає рівно те, що «сказала модель» — без мережі."""

    def __init__(self, entities: ExtractedEntities) -> None:
        self.entities = entities

    async def extract(self, text: str) -> ExtractedEntities:
        return self.entities


class RecordingVerifier:
    """Запамʼятовує, що саме дійшло до верифікації."""

    def __init__(self) -> None:
        self.seen: list[ExtractedEntities] = []

    async def verify(self, extracted: ExtractedEntities):
        self.seen.append(extracted)
        return [], [], [], []


class RecordingSink:
    def __init__(self) -> None:
        self.documents = []

    async def push(self, document):
        self.documents.append(document)
        return SimpleNamespace(status="ok", error=None)


def _event(text: str = TEXT) -> RawEvent:
    return RawEvent(
        _id="507f1f77bcf86cd799439011",
        source=SourceMetadata(source_type=SourceType.TELEGRAM),
        text=text,
        collected_at=datetime.now(timezone.utc),
    )


async def _run(entities: ExtractedEntities, text: str = TEXT):
    verifier = RecordingVerifier()
    pipeline = Pipeline(
        embedder=StaticEmbedder(),
        extractor=ScriptedExtractor(entities),
        verifier=verifier,
        dedup=SemanticDeduplicator(),
        sink=RecordingSink(),
    )
    await pipeline.process(_event(text))
    assert verifier.seen, "верифікатор не був викликаний"
    return verifier.seen[0]


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(settings, "pipeline_max_retries", 1)


def test_shipped_default_is_literal():
    """Дефолт у проді — найсуворіший режим.

    Виміряно 07.09.2026 на бойовому корпусі (hermes3:8b): новий промпт прибирає
    вироджений T1190, але модель починає навішувати ІНШІ техніки, підкріплюючи
    їх справжніми, але не по темі цитатами (T1056 «Input Capture» на пост про
    пошук відео в Google). Тобто "quoted" міняє форму вигадки, а не прибирає її."""
    assert Settings().grounding_mode == "literal"


@pytest.mark.asyncio
async def test_hallucinated_technique_never_reaches_verification():
    # Рівно бойовий випадок: T1190 без жодної опори в тексті.
    got = await _run(ExtractedEntities(mitre_tactics_techniques=["T1190"]))
    assert got.mitre_tactics_techniques == []


@pytest.mark.asyncio
async def test_literally_mentioned_cve_passes():
    got = await _run(ExtractedEntities(extracted_cve_ids=["CVE-2026-3055"]))
    assert got.extracted_cve_ids == ["CVE-2026-3055"]


@pytest.mark.asyncio
async def test_quoted_technique_passes_in_quoted_mode(monkeypatch):
    monkeypatch.setattr(settings, "grounding_mode", "quoted")
    got = await _run(ExtractedEntities(
        mitre_tactics_techniques=["T1190"],
        ttp_contexts={"T1190": "у публічному вебінтерфейсі поштового шлюзу"},
    ))
    assert got.mitre_tactics_techniques == ["T1190"]


@pytest.mark.asyncio
async def test_default_mode_drops_a_technique_backed_only_by_a_quote():
    # Саме той випадок, що ловився на проді: цитата справжня, техніка — ні.
    got = await _run(ExtractedEntities(
        mitre_tactics_techniques=["T1190"],
        ttp_contexts={"T1190": "у публічному вебінтерфейсі поштового шлюзу"},
    ))
    assert got.mitre_tactics_techniques == []


@pytest.mark.asyncio
async def test_fabricated_quote_does_not_pass(monkeypatch):
    monkeypatch.setattr(settings, "grounding_mode", "quoted")
    got = await _run(ExtractedEntities(
        mitre_tactics_techniques=["T1190"],
        ttp_contexts={"T1190": "виконання коду в ядрі Windows"},
    ))
    assert got.mitre_tactics_techniques == []


@pytest.mark.asyncio
async def test_literal_mode_drops_even_a_true_quote(monkeypatch):
    monkeypatch.setattr(settings, "grounding_mode", "literal")
    got = await _run(ExtractedEntities(
        mitre_tactics_techniques=["T1190"],
        ttp_contexts={"T1190": "у публічному вебінтерфейсі поштового шлюзу"},
    ))
    assert got.mitre_tactics_techniques == []


@pytest.mark.asyncio
async def test_gate_can_be_switched_off(monkeypatch):
    # Аварійний важіль: повернути стару поведінку без правки коду.
    monkeypatch.setattr(settings, "grounding_mode", "off")
    got = await _run(ExtractedEntities(mitre_tactics_techniques=["T1190"]))
    assert got.mitre_tactics_techniques == ["T1190"]


@pytest.mark.asyncio
async def test_title_and_summary_survive_the_gate():
    got = await _run(ExtractedEntities(
        title="Mail gateway exploited",
        summary="Attackers exploited the public web interface.",
        detected_language="uk",
        mitre_tactics_techniques=["T1190"],
    ))
    assert got.title == "Mail gateway exploited"
    assert got.summary == "Attackers exploited the public web interface."
    assert got.detected_language == "uk"


@pytest.mark.asyncio
async def test_mixed_batch_keeps_only_the_grounded_one(monkeypatch):
    monkeypatch.setattr(settings, "grounding_mode", "quoted")
    got = await _run(ExtractedEntities(
        extracted_cve_ids=["CVE-2026-3055", "CVE-2019-0708"],
        mitre_tactics_techniques=["T1190", "T1566"],
        ttp_contexts={"T1566": "активну експлуатацію вразливості"},
    ))
    assert got.extracted_cve_ids == ["CVE-2026-3055"]
    assert got.mitre_tactics_techniques == ["T1566"]


# --- символьний збір: те, що в тексті, беремо без участі моделі -----------
@pytest.mark.asyncio
async def test_literal_cve_is_picked_up_even_if_the_model_stayed_silent():
    # Найчастіший випадок на веб-корпусі: CVE стоїть у статті, модель мовчить.
    got = await _run(ExtractedEntities(), TEXT)
    assert got.extracted_cve_ids == ["CVE-2026-3055"]


@pytest.mark.asyncio
async def test_harvested_and_model_ids_are_merged_without_duplicates():
    got = await _run(ExtractedEntities(extracted_cve_ids=["CVE-2026-3055"]), TEXT)
    assert got.extracted_cve_ids == ["CVE-2026-3055"]


@pytest.mark.asyncio
async def test_harvest_does_not_resurrect_ungrounded_model_labels():
    got = await _run(ExtractedEntities(mitre_tactics_techniques=["T1190"]), TEXT)
    assert got.mitre_tactics_techniques == []


@pytest.mark.asyncio
async def test_technique_written_out_in_the_text_is_harvested():
    got = await _run(ExtractedEntities(), "Ланцюг атаки: T1566 -> ескалація.")
    assert got.mitre_tactics_techniques == ["T1566"]


@pytest.mark.asyncio
async def test_harvest_is_skipped_when_the_gate_is_off(monkeypatch):
    # "off" — аварійний важіль, що повертає СТАРУ поведінку цілком.
    monkeypatch.setattr(settings, "grounding_mode", "off")
    got = await _run(ExtractedEntities(), TEXT)
    assert got.extracted_cve_ids == []
