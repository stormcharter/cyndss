from pymongo import MongoClient

def reset_shipper_state():
    try:
        # Підключаємося до Mongo (перевір рядок підключення; якщо через контейнери — localhost:27017)
        client = MongoClient("mongodb://127.0.0.1:27017/?directConnection=true")
        db = client["threat_intel"]
        collection = db["shipper_state"]
        
        # Видаляємо маркер конвеєра
        result = collection.delete_one({"_id": "main_pipeline"})
        
        if result.deleted_count > 0:
            print("🟢 Маркер успішно видалено! Шиппер почне читати базу з нуля.")
        else:
            print("ℹ️ Маркер не знайдено в колекції shipper_state. Можливо, назва інша?")
            
    except Exception as e:
        print(f"❌ Помилка: {e}")

if __name__ == "__main__":
    reset_shipper_state()