import os
import json
import datetime
from dotenv import load_dotenv
from telethon import TelegramClient, events
import motor.motor_asyncio
from pymongo.errors import DuplicateKeyError

from collector_dedup import content_hash, ensure_dedup_indexes

# 1. Завантаження налаштувань
load_dotenv()

API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')

# Підключення до Mongo (Docker). На сервері — порт 27018 (27017 зайнятий системним
# mongod), задається через COLLECTOR_MONGO_URI у .env. Локально працює на дефолті.
MONGO_URI_LOCAL = os.getenv(
    "COLLECTOR_MONGO_URI",
    "mongodb://127.0.0.1:27017/?directConnection=true",
)
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI_LOCAL)
db = mongo_client[os.getenv("MONGO_DB", "threat_intel")]
collection = db[os.getenv("MONGO_COLLECTION", "raw_events")]

# 2. Завантаження каналів
with open('channels.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    CHANNELS_TO_MONITOR = tuple(config.get('channels', []))

client = TelegramClient('bot_session', API_ID, API_HASH)


# 3. Обробник повідомлень (Тільки збір)
@client.on(events.NewMessage(chats=CHANNELS_TO_MONITOR))
async def handler(event):
    text = event.raw_text
    if not text or len(text) < 10:
        return

    chat = await event.get_chat()
    print(f"\n[!] Отримано допис з каналу: {chat.title}")

    # Формуємо сирий документ (узгоджено з логікою шипера: channel_id, message_id).
    raw_event = {
        "source_type": "telegram",
        "channel_id": chat.id,
        "channel_name": chat.title,
        "message_id": event.message.id,
        "text": text,
        "content_hash": content_hash(text),
        "collected_at": datetime.datetime.now(),
    }

    # Дедуп робить сама БД через унікальні індекси. Дубль не дійде до шипера/ембеддингу.
    try:
        await collection.insert_one(raw_event)
        print("[+] Сирі дані успішно записані в MongoDB. Естафету передано шиперу.")
        print("-" * 50)
    except DuplicateKeyError:
        print("[=] Дубль (вже є в базі) — пропущено, шипер не турбуємо.")
        print("-" * 50)
    except Exception as e:
        print(f"[-] Помилка запису в базу даних: {e}")


# 4. Запуск
print("Юзербот запущений (Режим Collector). Очікую повідомлення...")
client.start()
# Гарантуємо індекси дедуплікації до початку прийому повідомлень (ідемпотентно).
client.loop.run_until_complete(ensure_dedup_indexes(collection))
client.run_until_disconnected()
