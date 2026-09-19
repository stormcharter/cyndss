# Shipper — конвеєр збагачення Mongo → Manticore

Асинхронний брокер, який слухає MongoDB Change Stream і для кожної нової події
будує збагачений документ для пошукового індексу Manticore. Це нейро-символічний
міст платформи: статистичне припущення LLM (які CVE/техніки згадано) проходить
через детерміновані фільтри — заземлення в тексті та довідники NVD/MITRE — і лише
після цього потрапляє в індекс.

Запускається як `python -m shipper.main` (у Docker — сервіс `shipper` у compose).

## Що робить

Для кожної події з `raw_events` (`shipper/pipeline.py`):

1. **Гейт порожнього тексту** — текст коротший за 5 символів (медіа без підпису)
   пропускається без жодного зовнішнього виклику.
2. **Ембединг** — `nomic-embed-text` через Ollama (або віддалений OpenAI-сумісний
   ендпойнт, `EMBED_PROVIDER`), 768 вимірів, L2-нормалізація.
3. **Видобування сутностей** — LLM через OpenAI-сумісний `chat/completions`
   (`LLM_BASE_URL`, за замовчуванням OpenRouter / Hermes 3; локально — Ollama через
   `/v1`). Строгий JSON: заголовок і реферат англійською, список CVE та технік
   ATT&CK, кожна — з цитатою-опорою з тексту. Текст документа передається як
   недовірений, у делімітерах (`prompt_guard.py`) — захист від непрямої ін'єкції промпту.
4. **Заземлення міток** (`grounding.py`) — мітка проходить далі лише якщо
   ідентифікатор є в тексті дослівно (`literal`, дефолт) або модель навела цитату,
   яка справді міститься в документі (`quoted`). Гейт стоїть ДО верифікації:
   NVD/MITRE підтверджують лише існування ідентифікатора, а не його доречність.
5. **Символьна верифікація** (`verifier.py`) — CVE проти NVD API 2.0 (CVSS, опис,
   CWE), техніки проти STIX-бандла MITRE ATT&CK (відкликані та застарілі
   відкидаються при завантаженні довідника). Статуси: `verified` / `not_found` /
   `invalid_format` / `unreachable`.
6. **Семантична дедуплікація** (`dedup.py`) — косинусна схожість ≥ 0.92 із вікном
   останніх 1000 проіндексованих векторів = перефразований репост, в індекс не йде.
7. **Мікробатч у Manticore** (`manticore_sink.py`) — `replace` через HTTP `/bulk`
   пачками по 50 документів або кожні 2 секунди. У поле `content` кладеться
   **лематизований** текст (`lemmatizer.py`) — та сама нормалізація, що й на
   пошуковому запиті в UI.

Лічильники `cve_count_verified` / `ttp_count_verified` рахують лише підтверджені
мітки; саме вони потім стають «якорями» в семантичній мережі та графі гіпотез.

Точна дедуплікація (унікальні індекси за `(channel_id, message_id)` та
`content_hash`) живе на рівні колекторів і Mongo — див. `collector_dedup.py`.

## Надійність

- **Оренда конвеєра (lease).** Шипер атомарно бере ексклюзивну оренду в колекції
  `shipper_state` і поновлює її; другий інстанс не стартує, доки оренда не спливе
  (`SHIPPER_LEASE_SECONDS`, `SHIPPER_STALE_SECONDS`).
- **Чекпойнт після ack.** `resume_token` комітиться в Mongo лише після того, як
  Manticore підтвердив запис (або подія надійно відкладена). Після рестарту конвеєр
  продовжує з місця падіння; при втраченому токені — дозбирає історію за `_id`.
- **Ретраї та dead-letter.** Відмови конвеєра або `/bulk` після обмежених ретраїв
  паркуються в `shipper_dead_letters` (`dead_letter.py`), а не губляться.
  Переграти: `python -m shipper.replay [--limit N]` (`replay.py`).
- **Готовність.** Loopback-ендпойнт `GET http://127.0.0.1:9100/` → JSON зі станом
  оренди/з'єднання і код 200/503.
- **Секрети в логах.** Конфіг логується через `redact.py` — паролі в URI та
  ключі в query-рядках маскуються.

## Модулі

```
main.py           — точка входу: збирання компонентів, сигнали, ендпойнт готовності
config.py         — Settings з .env (pydantic-settings), fail-fast на кривих значеннях
mongo_source.py   — Change Stream + оренда + чекпойнт (resume_token у shipper_state)
pipeline.py       — оркестратор одного документа (кроки 1–7 вище)
models.py         — RawEvent / ExtractedEntities / EnrichedDocument (pydantic)
dedup.py          — EmbeddingClient + SemanticDeduplicator (ковзне вікно)
extractor.py      — промпт екстрактора та розбір JSON-відповіді LLM
grounding.py      — заземлення міток CVE/MITRE у тексті (чиста stdlib)
verifier.py       — NVDClient + MitreAttackKB
manticore_sink.py — обмежена черга + flusher → /bulk, ретраї, DLQ
schema.py         — DDL таблиці Manticore (морфологія, інфікс, KNN 768)
dead_letter.py    — ідемпотентний запис у DLQ
replay.py         — переобробка відкладених подій
redact.py         — маскування секретів у дампі конфігу
```

## Схема таблиці Manticore

Створюється автоматично при старті (`schema.py`):

```sql
CREATE TABLE IF NOT EXISTS events_enriched (
    mongo_id STRING,
    source_type STRING,
    channel_name STRING,
    url STRING,
    collected_at TIMESTAMP,
    title TEXT stored indexed,
    language STRING,
    content TEXT indexed,            -- індексується для пошуку, але НЕ зберігається
    summary TEXT stored indexed,
    embedding FLOAT_VECTOR KNN_TYPE='hnsw' KNN_DIMS='768' HNSW_SIMILARITY='cosine',
    verified_cve_ids STRING,
    verified_ttp_ids STRING,
    cve_count_verified INT,
    ttp_count_verified INT,
    ioc_count INT
) engine='columnar' morphology='stem_en, stem_ru' min_infix_len='3' index_exact_words='1'
```

**Розділення сховищ.** `content` оголошено як `indexed` без `stored`: повний текст
шукається повнотекстово, але в Manticore не зберігається і в `SELECT` не
повертається. Оригінал живе лише в Mongo (`raw_events.text`), і UI тягне його
звідти за `mongo_id` — для показу картки й як контекст для LLM. Manticore лишається
легким індексом (вектор + реферат + метадані), дублювання повного тексту немає.

## Запуск

```bash
cp .env.example .env
# заповнити .env: LLM_API_KEY, за бажанням NVD_API_KEY

docker compose up -d
docker compose exec ollama ollama pull nomic-embed-text

docker compose logs -f shipper
```

Replica set `rs0` ініціюється автоматично healthcheck-ом контейнера `mongo`.
Локальний варіант без хмари (LLM через Ollama `/v1`) описаний у `.env.example`.

Змінити схему індексу (морфологію, розмірність) → правити `schema.py` і
перестворити таблицю (`reset_manticore.py`); дешево переіндексувати без
повторної екстракції — `tools/reindex_lemmatize.py`.

## Перевірка роботи

Після запуску покласти тестовий документ у Mongo:

```javascript
db.raw_events.insertOne({
  source_type: "telegram",
  channel_name: "test_cyber_channel",
  text: "Critical vulnerability CVE-2024-3094 in xz-utils allows remote code execution. Exploited via T1190.",
  collected_at: new Date()
})
```

За кілька секунд у логах шипера має з'явитися запис `pipeline_success` з
`cve_verified: 1, ttp_verified: 1`, а в Manticore — новий документ.

## Що лишилося за межами

- Метрики Prometheus (латентність конвеєра, частка дублікатів і `verified`).
- HNSW-дедуплікація в самому Manticore замість вікна в RAM (для дублікатів з
  інтервалом більшим за годину).
- Перевірка зв'язків між сутностями за полем `relationships` STIX-бандла — зараз
  `verified_relations` і `verified_iocs` завжди порожні.
