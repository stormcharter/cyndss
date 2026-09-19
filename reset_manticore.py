"""
Перестворення таблиці Manticore під нову схему.

Коли змінюється DDL (наприклад, у архітектурі B2 поле content стало `indexed`
без `stored`), команда CREATE TABLE IF NOT EXISTS не чіпає вже існуючу таблицю.
Тому стару таблицю треба видалити — shipper при наступному старті створить її
заново за актуальною схемою (shipper/manticore_sink.py).

Повний цикл перезаливки під нову схему:
    1. python reset_manticore.py     # видалити таблицю events_enriched
    2. python drop_marker.py         # скинути позицію shipper'а (читати Mongo з нуля)
    3. перезапустити shipper         # створить таблицю і перельє дані з Mongo

УВАГА: крок 2 змусить shipper заново прогнати ВСІ документи через ембеддинг і
LLM — на великій базі це довго. Якщо історія не потрібна, можна зробити лише
крок 1: тоді в нову таблицю потраплять тільки нові події.
"""

import pymysql


def drop_table():
    try:
        conn = pymysql.connect(host="127.0.0.1", port=9306, autocommit=True)
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS events_enriched")
        conn.close()
        print("🟢 Таблицю events_enriched видалено.")
        print("ℹ️ Shipper при старті створить її заново за новою схемою.")
        print("ℹ️ Щоб залити існуючі дані з Mongo: python drop_marker.py + перезапуск shipper.")
    except Exception as e:
        print(f"❌ Помилка підключення до Manticore (127.0.0.1:9306): {e}")


if __name__ == "__main__":
    drop_table()
