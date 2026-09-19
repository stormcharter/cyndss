"""
Dead-letter: записувати невдалі спроби збагачення в Mongo, щоб вони не губилися
мовчки і їх можна було переобробити пізніше.

Ідемпотентний дизайн на подію: один запис на (mongo_id, pipeline_revision),
безпечно перезаписувати при ретраї. Поле ``source_position`` дає replay змогу
відновити курсор, не звертаючись до shipper_state.
"""

from datetime import datetime, timezone

import structlog
from motor.motor_asyncio import AsyncIOMotorCollection

logger = structlog.get_logger("shipper.dead_letter")

# Підняти, коли змінюється семантика payload у DLQ. Кожна ревізія створює нові
# записи DLQ для того самого mongo_id (старі застарілі, але нешкідливі).
PIPELINE_REVISION = "2"


async def write_dead_letter(
    collection: AsyncIOMotorCollection,
    *,
    mongo_id: str,
    stage: str,
    error: str,
    error_class: str,
    attempts: int,
    pipeline_revision: str = PIPELINE_REVISION,
    source_position: str | None = None,
) -> bool:
    """Ідемпотентний dead-letter на подію. Повертає True лише якщо запис надійний."""
    _id = f"dlq:{pipeline_revision}:{mongo_id}"
    now = datetime.now(timezone.utc)
    try:
        await collection.update_one(
            {"_id": _id},
            {
                "$set": {
                    "mongo_id": mongo_id,
                    "stage": stage,
                    "error": error[:2000],
                    "error_class": error_class,
                    "attempts": attempts,
                    "pipeline_revision": pipeline_revision,
                    "source_position": source_position,
                    "failed_at": now,
                    "updated_at": now,
                    "replayed_at": None,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        return True
    except Exception as exc:
        # Причину треба бачити: якщо падає САМ dead-letter, конвеєр лишається
        # без останньої страховки, а мовчазне False у логах виглядає як «усе ок».
        logger.error("dead_letter_write_failed", mongo_id=mongo_id, error=str(exc))
        return False  # той, хто викликає, має вважати подію НЕ відкладеною


async def mark_replayed(
    collection: AsyncIOMotorCollection,
    mongo_id: str,
    pipeline_revision: str = PIPELINE_REVISION,
) -> None:
    """Mark a DLQ entry as replayed (idempotent)."""
    await collection.update_one(
        {"_id": f"dlq:{pipeline_revision}:{mongo_id}"},
        {"$set": {"replayed_at": datetime.now(timezone.utc)}},
    )
