"""
Спільні утиліти дешевої та ТОЧНОЇ дедуплікації для колекторів
(migrate_history.py — бекфіл історії, tg.py — живий збір).

Філософія: важку СМИСЛОВУ дедуплікацію (ембеддинги, cosine ≥ 0.92) лишаємо
шиперу. А дешеві й точні перевірки робимо ще ДО запису в Mongo, на рівні
унікальних індексів самої БД — щоб дубль навіть не доходив до ембеддингу:

  1) (channel_id, message_id) — природний ключ Telegram. Той самий пост того ж
     каналу не запишеться вдруге. Дає інкрементну докачку через min_id.
  2) content_hash — sha256 від нормалізованого тексту. Ловить точні репости
     (зокрема МІЖ каналами) без жодного ембеддингу.

Обидва індекси — partial (лише для документів, де відповідні поля існують),
тому старі (legacy) документи без цих полів не конфліктують при створенні індексу.
"""

import re
import hashlib

import pymongo
from pymongo.errors import OperationFailure

UQ_CHANNEL_MESSAGE = "uq_channel_message"
UQ_CONTENT_HASH = "uq_content_hash"


def content_hash(text: str) -> str:
    """sha256 від нормалізованого тексту: lowercase + схлопнуті пробіли + trim."""
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


async def ensure_dedup_indexes(collection) -> None:
    """Ідемпотентно створює унікальні partial-індекси дедуплікації.

    Якщо в базі вже є дублі по цих ключах — створення унікального індексу впаде;
    у такому разі попереджаємо й працюємо далі (інкремент через min_id усе одно
    обмежує вибірку, а дублі добере шиперський семантичний рівень).
    """
    specs = [
        dict(
            keys=[("channel_id", pymongo.ASCENDING), ("message_id", pymongo.ASCENDING)],
            name=UQ_CHANNEL_MESSAGE,
            partialFilterExpression={
                "channel_id": {"$exists": True},
                "message_id": {"$exists": True},
            },
        ),
        dict(
            keys=[("content_hash", pymongo.ASCENDING)],
            name=UQ_CONTENT_HASH,
            partialFilterExpression={"content_hash": {"$exists": True}},
        ),
    ]
    for spec in specs:
        try:
            await collection.create_index(
                spec["keys"],
                name=spec["name"],
                unique=True,
                partialFilterExpression=spec["partialFilterExpression"],
            )
        except OperationFailure as e:
            print(
                f"[!] Не вдалося створити унікальний індекс '{spec['name']}': {e}. "
                "Імовірно в базі вже є дублі — очистіть raw_events або приберіть їх, "
                "після чого індекс створиться. Поки що працюю без цього індексу."
            )
