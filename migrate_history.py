import os
import json
import datetime
from dotenv import load_dotenv
from telethon import TelegramClient
import motor.motor_asyncio
from pymongo.errors import DuplicateKeyError

from collector_dedup import content_hash, ensure_dedup_indexes

# 1. Завантаження налаштувань
load_dotenv()

API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')

# Скільки постів тягнемо при ПЕРШОМУ заборі каналу (сід-бекфіл).
# Далі канал докачується лише новими постами (інкремент через min_id) — без ліміту.
SEED_LIMIT_PER_CHANNEL = 1000

# Підключення до Mongo (Docker). На сервері — порт 27018 через COLLECTOR_MONGO_URI у .env.
MONGO_URI_LOCAL = os.getenv(
    "COLLECTOR_MONGO_URI",
    "mongodb://127.0.0.1:27017/?directConnection=true",
)
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI_LOCAL)
db = mongo_client[os.getenv("MONGO_DB", "threat_intel")]
collection = db[os.getenv("MONGO_COLLECTION", "raw_events")]

# 2. Завантаження списку каналів
with open('channels.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    CHANNELS_TO_MONITOR = config.get('channels', [])

# Використовуємо ту саму сесію, щоб не проходити авторизацію заново
client = TelegramClient('bot_session', API_ID, API_HASH)


async def high_water_mark(channel_id):
    """Найбільший message_id, який ми вже зберегли для цього каналу (або None).

    Це «верхня точка» інкременту: наступного разу докачуємо тільки те, що новіше.
    Запит швидкий — спирається на індекс (channel_id, message_id).
    """
    doc = await collection.find_one(
        {"channel_id": channel_id, "message_id": {"$exists": True}},
        sort=[("message_id", -1)],
        projection={"message_id": 1},
    )
    return doc["message_id"] if doc else None


async def migrate():
    print("[+] Підключення до Telegram...")
    await client.start()

    # Гарантуємо унікальні індекси дедуплікації (ідемпотентно).
    await ensure_dedup_indexes(collection)

    print(f"[+] Починаємо збір історії для {len(CHANNELS_TO_MONITOR)} каналів...\n")

    for channel in CHANNELS_TO_MONITOR:
        print("=" * 50)
        print(f"🔗 Обробка каналу: {channel}")
        print("=" * 50)

        try:
            chat = await client.get_entity(channel)
            chat_title = chat.title
            channel_id = chat.id

            hw = await high_water_mark(channel_id)
            if hw:
                # Інкремент: тільки повідомлення НОВІШІ за останнє збережене.
                # reverse=True → йдемо хронологічно (старіші нові → новіші),
                # тож переривання не лишає дірок: max(message_id) завжди суцільний фронт.
                print(f"[i] Відомий останній пост #{hw}. Докачую лише новіші (інкремент)…")
                msg_iter = client.iter_messages(chat, min_id=hw, reverse=True)
            else:
                # Перший раз для цього каналу — сід: останні N постів.
                print(f"[i] Канал ще не імпортувався. Сід: останні {SEED_LIMIT_PER_CHANNEL} постів…")
                msg_iter = client.iter_messages(chat, limit=SEED_LIMIT_PER_CHANNEL)

            count_added = 0
            count_skipped = 0

            async for message in msg_iter:
                text = message.raw_text

                # Пропускаємо порожні або занадто короткі пости (картинки без опису тощо)
                if not text or len(text) < 10:
                    continue

                raw_event = {
                    "source_type": "telegram",
                    "channel_id": channel_id,
                    "channel_name": chat_title,
                    "message_id": message.id,
                    "text": text,
                    "content_hash": content_hash(text),
                    "collected_at": message.date.replace(tzinfo=None) if message.date else datetime.datetime.now(),
                }

                # Дедуп робить сама БД через унікальні індекси — жодного скану по тексту.
                try:
                    await collection.insert_one(raw_event)
                    count_added += 1
                except DuplicateKeyError:
                    count_skipped += 1

            print(f"[+] Канал {channel} оброблений!")
            print(f"    - Додано нових постів: {count_added}")
            print(f"    - Пропущено дублікатів: {count_skipped}\n")

        except Exception as e:
            print(f"[-] Помилка при обробці каналу {channel}: {e}\n")

    print("[==] Міграція завершена! Всі нові пости залиті в MongoDB.")


if __name__ == "__main__":
    # Запуск асинхронного клієнта Telethon
    with client:
        client.loop.run_until_complete(migrate())
