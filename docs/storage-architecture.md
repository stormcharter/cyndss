# 🛡️ Архітектура Сховища Даних та Демонстрація Схеми Платформи CTI

Цей документ описує структуру даних, конфігурацію та патерни взаємодії розподіленого гібридного сховища платформи кіберрозвідки (Cyber Threat Intelligence). Платформа побудована за сучасним промисловим патерном **«Search Index ↔️ Document Store»** (аналог зв'язки Elasticsearch + БД у комерційних SOC-системах), що повністю виключає дублювання важких текстових даних та оптимізує аналітичні вибірки.

---

## 🏗️ Паттерн CQRS / Розділення Пошукового та Документного Шару

1. **MongoDB (`raw_events`)** — Первинне документоорієнтоване сховище («Книга на полиці»). Зберігає незмінні сирі OSINT-події, повні оригінальні тексти документів, лонги, транскрипти YouTube-відео та первинні метадані колекторів.
2. **Manticore Search (`events_enriched`)** — Швидкий реляційно-векторний пошуковий шар («Предметний покажчик»). Зберігає інвертований індекс слів для `MATCH`-пошуку, 768-вимірні вектори для семантичного KNN-пошуку, агреговані метрики безпеки та `mongo_id` як унікальний покажчик (Foreign Key) на оригінал.

### Конвеєр обробки запиту (Query Pipeline):
Під час пошуку інтерфейс Streamlit знаходить релевантні події через Manticore Search (за векторами або ключовими словами), отримує компактний масив `mongo_id` і за один швидкий батч-запит підтягує оригінальні тексти з MongoDB. Цей текст передається у контекст LLM (Hermes 3) або розгортається в картці події для аналітика SOC. Пошуковий шар залишається максимально легким та швидким.

---

## 1. Схема Таблиці Manticore Search (`events_enriched`)

Таблиця спроектована з використанням колоночного рушія (Columnar Engine) для миттєвої фільтрації та нативного KNN-індексу. Головна архітектурна особливість — поле `content` є **«сліпим індексом»** (має атрибут `indexed`, але не має `stored`).

### Результат виконання команди `DESCRIBE events_enriched;` у CLI Manticore:

| Field | Type | Properties | Призначення в системі CTI |
| :--- | :--- | :--- | :--- |
| **id** | bigint | columnar fast_fetch | Внутрішній унікальний ідентифікатор запису Manticore |
| **title** | text | indexed stored | Заголовок події (індексується та повертається в UI) |
| **content** | text | **indexed** | **Сліпий повнотекстовий індекс. Текст не зберігається, але доступний для пошуку через `MATCH()`** |
| **summary** | text | indexed stored | Кратка вижимка (AI Summary), згенерована моделлю Hermes 3 |
| **mongo_id** | string | columnar fast_fetch | Ключ зв'язку з оригінальним документом у MongoDB (`_id`) |
| **source_type** | string | columnar fast_fetch | Тип джерела: `TELEGRAM` (📡), `YOUTUBE` (▶️), `WEBSITE` (🌐) |
| **channel_name**| string | columnar fast_fetch | Назва каналу, пабліка або домен сайту-першоджерела |
| **url** | string | columnar fast_fetch | Пряме посилання на першоджерело для верифікації аналітиком |
| **collected_at**| timestamp | columnar fast_fetch | Таймстамп збору події (використовується для `ORDER BY DESC`) |
| **language** | string | columnar fast_fetch | Мова оригіналу (`uk`, `en`, `ru`) для мультиязичної фільтрації |
| **embedding** | float_vector| **columnar knn fast_fetch**| **768-вимірний вектор тексту від `nomic-embed-text` для семантичного KNN-пошуку** |
| **verified_cve_ids**| string | columnar fast_fetch | Рядок верифікованих ідентифікаторів CVE (через кому) |
| **verified_ttp_ids**| string | columnar fast_fetch | Рядок верифікованих технік MITRE ATT&CK |
| **cve_count_verified**| uint | columnar fast_fetch | Кількість підтверджених уразливостей у події |
| **ttp_count_verified**| uint | columnar fast_fetch | Кількість підтверджених тактик/технік ворога в події |
| **ioc_count** | uint | columnar fast_fetch | Кількість витягнутих індикаторів компрометації (IP, URL, хеші) |

---

## 2. Структура Документа в MongoDB (`threat_intel.raw_events`)

Документ у базу даних Mongo надходить у сирому вигляді від модульних колекторів, після чого збагачується метаданими та зберігається як єдине джерело правди (Single Source of Truth).

### Приклад JSON-структури документа:
```json
{
  "_id": ObjectId("6a166875496c80dfa7dadfa6"),
  "title": "Ubuntu Vulnerability CVE-2026-3888 Exploited for Root Access",
  "content": "A critical vulnerability in Ubuntu snap-confine and systemd-tmpfiles interaction allows local users to gain root privileges...",
  "summary": "A vulnerability in Ubuntu, identified as CVE-2026-3888, allows a regular user to gain root access via temporary directory replacement.",
  "metadata": {
    "source_type": "WEBSITE",
    "channel_name": "CyberSecurityNews",
    "url": "[https://example.com/ubuntu-cve-2026-3888](https://example.com/ubuntu-cve-2026-3888)",
    "collected_at": 1779836259,
    "language": "en"
  },
  "extracted_entities": {
    "cve_ids": ["CVE-2026-3888"],
    "ttp_ids": ["T1068", "T1543"]
  }
}

--- 
### Перевірка схема мантікори
docker compose exec manticore mysql -P 9306 -h 127.0.0.1 --execute="DESCRIBE events_enriched;"

### 3 останні записи у мантікору:
docker compose exec manticore mysql -P 9306 -h 127.0.0.1 --execute="SELECT id, title, mongo_id, source_type, channel_name, verified_cve_ids, cve_count_verified FROM events_enriched ORDER BY collected_at DESC LIMIT 3\G"

### Скільки всього записів у мантикорі
docker compose exec manticore mysql -P 9306 -h 127.0.0.1 --execute="SELECT source_type, COUNT(*) as total FROM events_enriched GROUP BY source_type;"

### Сліпий індекс
docker compose exec manticore mysql -P 9306 -h 127.0.0.1 --execute="SELECT id, title, mongo_id FROM events_enriched WHERE MATCH('ubuntu') LIMIT 2\G"

### Останній документ у монго
docker compose exec mongo mongosh --eval "db.getSiblingDB('threat_intel').raw_events.find().sort({_id:-1}).limit(1).toArray()"