"""
Моделі даних для шипера.

Тут зафіксовано три ключові схеми:
1. RawEvent — що лежить у MongoDB (те, що туди кладе юзербот).
2. ExtractedEntities — що LLM видобуває з тексту.
3. EnrichedDocument — те, що зрештою їде в Manticore.

Pydantic використовується заради строгої валідації: якщо юзербот змінить формат,
шипер упаде зі зрозумілою помилкою, а не запише сміття в індекс.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- Джерела ----------

class SourceType(str, Enum):
    """Звідки прийшло повідомлення. Розширюється в міру додавання колекторів."""
    TELEGRAM = "telegram"
    RSS = "rss"
    YOUTUBE = "youtube"
    WEBSITE = "website"
    # Авторитетні джерела (NVD/MITRE-подібні бюлетені) та завантажені аналітиком
    # файли з UI (див. rag_context.make_upload_doc). Обробляються так само, як
    # решта — тип потрібен лише щоб шиппер не відкидав такий документ як
    # MalformedSourceEvent.
    AUTHORITATIVE = "authoritative"
    UPLOAD = "upload"


# ---------- Сирий документ у Mongo ----------

class SourceMetadata(BaseModel):
    """Метадані джерела: канал, автор, посилання."""
    # Telegram віддає channel_id/message_id як int, і migrate_history.py кладе їх
    # у Mongo числами. Зводимо числа до рядка, інакше такі документи відбраковуються
    # на валідації (malformed_event_skipped) і не потрапляють в індекс.
    model_config = ConfigDict(coerce_numbers_to_str=True)

    source_type: SourceType
    channel_id: Optional[str] = None  # для Telegram — id каналу
    channel_name: Optional[str] = None
    message_id: Optional[str] = None  # унікальний id усередині джерела
    url: Optional[str] = None  # пряме посилання на пост, якщо є


class RawEvent(BaseModel):
    """
    Документ, який юзербот кладе в колекцію raw_events.

    Поле _id — ObjectId Mongo, у коді серіалізується в рядок.
    text — головне поле, по ньому йде вся подальша обробка.
    raw_payload — оригінальний об'єкт від джерела, про всяк випадок
        (раптом знадобиться видобути щось іще).
    """
    id: str = Field(alias="_id")
    source: SourceMetadata
    text: str
    collected_at: datetime
    raw_payload: dict = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


# ---------- Сутності, видобуті LLM ----------

class CVEEntity(BaseModel):
    """CVE-кандидат, видобутий LLM. До валідації проти NVD — claimed_*."""
    cve_id: str  # наприклад "CVE-2024-12345"
    claimed_context: Optional[str] = None  # фрагмент тексту, де згадано CVE


class MitreTTPEntity(BaseModel):
    """Техніка MITRE ATT&CK. Наприклад T1190 або T1078.004."""
    technique_id: str
    claimed_context: Optional[str] = None


class IoCEntity(BaseModel):
    """Індикатор компрометації (IoC). Тип — ip, domain, url, sha256, md5, sha1."""
    ioc_type: str
    value: str
    claimed_context: Optional[str] = None


class ClaimedRelation(BaseModel):
    """
    Зв'язок, заявлений LLM між двома сутностями.
    Наприклад: CVE-2024-XXXX exploited_via T1190.
    Шипер перевірить, чи справді такий зв'язок правдоподібний.
    """
    subject: str  # CVE-ID або TTP-ID
    predicate: str  # exploited_via, mitigated_by, related_to, ...
    object: str


class ExtractedEntities(BaseModel):
    """Те, що ЗАЯВИЛА модель. Ще не факт — заявка.

    `*_contexts` — цитати, якими модель обґрунтувала кожен ідентифікатор
    ({ID: фрагмент}). На них стоїть заземлення в `shipper/grounding.py`:
    мітка без дослівного ID у тексті й без справжньої цитати відкидається ДО
    звернення до NVD/MITRE. Порожній словник = модель нічого не процитувала.
    """
    title: str = "Unparsed Threat Event"
    summary: str | None = None
    extracted_cve_ids: list[str] = Field(default_factory=list)
    mitre_tactics_techniques: list[str] = Field(default_factory=list)
    cve_contexts: dict[str, str] = Field(default_factory=dict)
    ttp_contexts: dict[str, str] = Field(default_factory=dict)
    detected_language: str = "unknown"


# ---------- Верифіковані сутності ----------

class VerificationStatus(str, Enum):
    """Статус перевірки сутності проти зовнішнього джерела істини."""
    VERIFIED = "verified"  # підтверджено в NVD/MITRE
    NOT_FOUND = "not_found"  # формат валідний, але в базі немає (галюцинація LLM)
    INVALID_FORMAT = "invalid_format"  # формат не підходить (наприклад CVE-99-99)
    UNREACHABLE = "unreachable"  # джерело істини недоступне — потрібна повторна перевірка


class VerifiedCVE(BaseModel):
    """CVE після перевірки в NVD."""
    cve_id: str
    status: VerificationStatus
    cvss_score: Optional[float] = None  # підтягнуто з NVD
    description: Optional[str] = None  # офіційний опис із NVD
    published_date: Optional[datetime] = None
    cwe_ids: list[str] = Field(default_factory=list)


class VerifiedTTP(BaseModel):
    """Техніка MITRE після перевірки в STIX-бандлі."""
    technique_id: str
    status: VerificationStatus
    name: Optional[str] = None
    tactic: Optional[str] = None  # до якої тактики належить
    description: Optional[str] = None


class VerifiedRelation(BaseModel):
    """Зв'язок після перевірки. Якщо subject і object — обидва verified, і в їхніх
    описах є перетин — relation вважається plausible."""
    subject: str
    predicate: str
    object: str
    status: VerificationStatus  # verified / claimed_unverified


# ---------- Вихідний документ для Manticore ----------

class EnrichedDocument(BaseModel):
    """
    Те, що зрештою потрапляє в Manticore.

    Поля розділено за шарами:
    - ідентифікація та джерело (для метаданих і фільтрації),
    - текст і ембединг (для пошуку),
    - верифіковані сутності (для нейро-символічної частини — фільтрація
        за CVE/TTP, пояснюваність, довіра).
    """

    def to_manticore_json(self):
        # Перетворює Pydantic на JSON-сумісний словник
        return self.model_dump(mode='json')
    
    # Ідентифікація
    mongo_id: str  # посилання назад у Mongo (джерело істини)
    source_type: SourceType
    channel_name: Optional[str] = None
    url: Optional[str] = None
    collected_at: datetime
    title: str
    language: str
    # Текст і пошук
    text: str
    summary: Optional[str] = None
    embedding: list[float]  # розмірність 768 (nomic-embed-text)

    # Верифіковані сутності
    verified_cves: list[VerifiedCVE] = Field(default_factory=list)
    verified_ttps: list[VerifiedTTP] = Field(default_factory=list)
    verified_iocs: list[IoCEntity] = Field(default_factory=list)
    verified_relations: list[VerifiedRelation] = Field(default_factory=list)

    # Скаляри для швидкої фільтрації в Manticore
    cve_count_verified: int = 0
    ttp_count_verified: int = 0
    ioc_count: int = 0
