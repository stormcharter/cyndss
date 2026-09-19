"""
DDL таблиці Manticore — чиста функція, без залежності від shipper.config.

Винесено з manticore_sink.py, щоб побудову схеми можна було тестувати
ізольовано (без .env / pydantic-settings) і щоб опції пошуку (морфологія,
інфікс) жили в одному явному місці.
"""

from __future__ import annotations


def build_create_table_sql(table_name: str) -> str:
    """Повертає CREATE TABLE для індексу збагачених подій.

    content TEXT indexed (без stored): тіло повнотекстово ІНДЕКСУЄТЬСЯ, але не
    зберігається в Manticore — оригінал лежить у Mongo, UI тягне по mongo_id
    (архітектура B2). Розмірність KNN зафіксована під nomic-embed-text (768).

    Опції пошуку (корінь бага «vulnerability != vulnerabilities»):
      * morphology='stem_en, stem_ru' — стемінг на індексації Й на запиті.
        stem_en стягує англ. словоформи (vulnerability == vulnerabilities) —
        це головний фікс. stem_ru стягує РОСІЙСЬКІ флексії (корпус має ru-пости).
        УВАГА: власного стемера для української Manticore НЕ має, і stem_ru
        українські флексії НЕ стягує (перевірено: «вразливість»/«вразливості»
        лишаються різними токенами через чергування голосних і/о в корені).
        Українська морфологія покривається інфіксом нижче (та лематизацією на
        рівні запиту — окремий крок).
      * min_infix_len='3' — підрядковий добір: запит виду «вразлив*» ловить усі
        українські форми (вразливість/вразливості/вразливостей). Мовно-незалежна
        страховка; корпус ~1.5к док., вартість інфіксного індексу мізерна.
      * index_exact_words='1' — поряд зі стемами лишає ТОЧНІ форми, тож точний
        збіг усе ще можливий (=слово) і не програє в ранжуванні.
    """
    return f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    mongo_id STRING,
    source_type STRING,
    channel_name STRING,
    url STRING,
    collected_at TIMESTAMP,
    title TEXT stored indexed,
    language STRING,
    content TEXT indexed,
    summary TEXT stored indexed,
    embedding FLOAT_VECTOR KNN_TYPE='hnsw' KNN_DIMS='768' HNSW_SIMILARITY='cosine',
    verified_cve_ids STRING,
    verified_ttp_ids STRING,
    cve_count_verified INT,
    ttp_count_verified INT,
    ioc_count INT
) engine='columnar' morphology='stem_en, stem_ru' min_infix_len='3' index_exact_words='1'
"""
