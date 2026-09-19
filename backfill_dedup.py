"""
Одноразовий бекфіл дедуплікації для НАЯВНИХ (legacy) документів raw_events.

Навіщо: старі колектори не писали message_id/content_hash, тож після оновлення
повторний імпорт сприйняв би старі пости як «нові» й погнав би їх через ембеддинг.
Цей скрипт проставляє content_hash усім документам без нього — і тоді майбутні
повторні імпорти просто пропустять їх на рівні унікального індексу (БЕЗ GPU/LLM).

Чистий Mongo, ідемпотентно, безпечно запускати повторно.
Запуск:  python backfill_dedup.py
"""

import asyncio
import motor.motor_asyncio
from pymongo.errors import DuplicateKeyError

from collector_dedup import content_hash, ensure_dedup_indexes

MONGO_URI = "mongodb://127.0.0.1:27017/?directConnection=true"


async def main():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    col = client["threat_intel"]["raw_events"]

    await ensure_dedup_indexes(col)

    to_do = await col.count_documents({"content_hash": {"$exists": False}})
    print(f"[i] Документів без content_hash: {to_do}")

    updated = 0
    conflicts = 0
    cursor = col.find({"content_hash": {"$exists": False}}, projection={"text": 1})
    async for doc in cursor:
        h = content_hash(doc.get("text", ""))
        try:
            await col.update_one({"_id": doc["_id"]}, {"$set": {"content_hash": h}})
            updated += 1
        except DuplicateKeyError:
            # Точний дубль уже захешованого документа — лишаємо як є (один із копій
            # уже має хеш і блокуватиме повторний імпорт цього тексту).
            conflicts += 1

    print(f"[+] Бекфіл завершено: проставлено content_hash — {updated}, "
          f"точних дублів пропущено — {conflicts}")


if __name__ == "__main__":
    asyncio.run(main())
