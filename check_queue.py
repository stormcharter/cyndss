import pymongo
from datetime import datetime

# Підключення до твоєї локальної MongoDB
MONGO_URI_LOCAL = "mongodb://127.0.0.1:27017/?directConnection=true"
client = pymongo.MongoClient(MONGO_URI_LOCAL)
db = client["threat_intel"]

def get_queue_status():
    # Рахуємо скільки всього сирих постів у Монзі
    total_raw = db["raw_events"].count_documents({})
    
    # Рахуємо скільки вже успішно оброблено і лежить в Мантикорі
    # (Для цього смикаємо ту саму Монгу, якщо ти зберігаєш статус, або просто дивимось лічильник)
    try:
        # Підключаємось до Мантикори через MySQL протокол, щоб взяти точну цифру
        import pymysql
        conn = pymysql.connect(host="127.0.0.1", port=9306, autocommit=True)
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM events_enriched")
            total_enriched = cursor.fetchone()[0]
        conn.close()
    except Exception:
        total_enriched = "Не вдалося підключитися до Manticore"

    print("=" * 40)
    print(f"📊 СТАТУС ОЧЕРЕДІ НА {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 40)
    print(f"📥 Всього викачано в MongoDB:  {total_raw}")
    
    if isinstance(total_enriched, int):
        remaining = max(0, total_raw - total_enriched)
        print(f"🟢 Оброблено Мантикорою:      {total_enriched}")
        print(f"⏳ Залишилось в черзі:        {remaining}")
        if total_raw > 0:
            progress = (total_enriched / total_raw) * 100
            print(f"📈 Прогрес:                  {progress:.1f}%")
    else:
        print(f"⚠️  Мантикора: {total_enriched}")
    print("=" * 40)

if __name__ == "__main__":
    get_queue_status()