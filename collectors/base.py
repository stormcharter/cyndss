"""
Спільна інфраструктура для пакетних колекторів.

RawEventWriter — тонка обгортка над MongoDB, яка пише сирі документи у
raw_events рівно в тому ж форматі, що й tg.py. Поля повністю збігаються з
тим, що читає shipper (shipper/mongo_source.py -> _to_raw_event):

    source_type   рядок: "telegram" | "youtube" | "website" | "rss"
    channel_name  назва джерела (канал / сайт / відео)
    url           пряме посилання на матеріал (для youtube/website — ключ дедуплікації)
    message_id    унікальний id всередині джерела (для youtube — id відео)
    text          основний текст, з яким далі працює shipper
    collected_at  дата публікації або час збору
    raw_payload   будь-які додаткові метадані «про запас»

Дедуплікація на рівні бази: перед вставкою перевіряємо, чи немає вже такого
url (а якщо url немає — пари channel_name+text). Це не дає повторно залити
 те саме при повторному запуску колектора. Семантичну дедуплікацію поверх
цього все одно робить shipper.
"""

import datetime
import json
import os

import pymongo
from dotenv import load_dotenv

load_dotenv()

# Колектори працюють на хості, тож звертаються до Mongo через localhost.
# Shipper всередині docker використовує іншу адресу (mongo:27017) зі свого .env —
# тут навмисно окрема змінна, щоб не плутати ці два контексти.
MONGO_URI = os.getenv(
    "COLLECTOR_MONGO_URI",
    "mongodb://127.0.0.1:27017/?directConnection=true",
)
DB_NAME = os.getenv("MONGO_DB", "threat_intel")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "raw_events")

# Коротші тексти shipper усе одно відсіє (pipeline.process), тож не засмічуємо базу.
MIN_TEXT_LENGTH = 20

SOURCES_FILE = os.getenv("SOURCES_FILE", "channels.json")


class RawEventWriter:
    """Запис сирих подій у MongoDB з дедуплікацією по url/тексту."""

    def __init__(self):
        self.client = pymongo.MongoClient(MONGO_URI)
        self.collection = self.client[DB_NAME][COLLECTION_NAME]

    def _already_exists(self, url, text, channel_name) -> bool:
        # url — природний унікальний ключ для youtube/website
        if url and self.collection.find_one({"url": url}):
            return True
        # запасний варіант — як у migrate_history.py: пара (канал, текст)
        return self.collection.find_one(
            {"channel_name": channel_name, "text": text}
        ) is not None

    def write(
        self,
        *,
        source_type: str,
        text: str,
        channel_name: str | None = None,
        url: str | None = None,
        message_id: str | None = None,
        collected_at: datetime.datetime | None = None,
        raw_payload: dict | None = None,
    ) -> bool:
        """Записує один документ. Повертає True, якщо реально вставили."""
        text = (text or "").strip()
        if len(text) < MIN_TEXT_LENGTH:
            return False

        if self._already_exists(url, text, channel_name):
            label = url or (channel_name or "")[:40]
            print(f"    [=] Пропущено дублікат: {label}")
            return False

        doc = {
            "source_type": source_type,
            "channel_name": channel_name,
            "url": url,
            "message_id": message_id,
            "text": text,
            "collected_at": collected_at or datetime.datetime.now(),
            "raw_payload": raw_payload or {},
        }
        self.collection.insert_one(doc)
        print(f"    [+] Додано: {url or (channel_name or '')[:40]}")
        return True

    def close(self):
        self.client.close()


def load_sources(key: str) -> list[str]:
    """Читає список джерел заданого типу з channels.json (ключі youtube / websites)."""
    try:
        with open(SOURCES_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"[!] Файл {SOURCES_FILE} не знайдено.")
        return []
    return [s for s in config.get(key, []) if s and isinstance(s, str)]
