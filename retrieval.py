"""
Гібридний пошук: злиття видач кількох стратегій в один список контексту.

НАВІЩО. Агентський шлях шукав одним способом — AND-матчинг усіх слів запиту, а
якщо нічого не знайшлось, то KNN. Питання «say about vulnerabilities» вимагало
від документа ОБИДВА слова, давало 1-2 випадкові збіги, і агент чесно відповідав
«Знайдені джерела не містять релевантної інформації». Лікується не промптом, а
пошуком: кілька стратегій паралельно (лематизований OR по оригіналу, OR по
англійському перекладі, векторний KNN) + злиття їхніх видач.

Злиття КРУГОВЕ (round-robin), а не «спершу вся перша стратегія, потім друга»:
інакше стратегія з великою видачею з'їдає ліміт цілком і гібрид вироджується в
неї саму — тобто ми знову маємо один спосіб пошуку, лише дорожчий.

Модуль чистий — без streamlit/pymysql, тестується ізольовано.
"""

from __future__ import annotations


def _hit_key(hit: dict):
    """Ідентичність документа. id — з індексу Manticore; mongo_id — запасний
    ключ для гілок, де числового id немає (напр. документи з Mongo напряму).

    None означає «ідентифікувати нічим» — такий запис merge_hits пропускає:
    його не можна ані дедуплікувати, ані дотягнути оригінал із Mongo."""
    if hit.get("id") is not None:
        return ("id", hit["id"])
    if hit.get("mongo_id"):
        return ("mongo", str(hit["mongo_id"]))
    return None


def merge_hits(groups, limit: int = 8) -> list[dict]:
    """Зливає видачі стратегій пошуку в один список без дублів.

    `groups` — послідовність списків документів у порядку пріоритету
    (перша стратегія бере перший слот кожного кола). Впала стратегія може
    віддати None — це не помилка. Повертає щонайбільше `limit` документів,
    самі об'єкти не змінюються."""
    if not groups or limit <= 0:
        return []

    lists = [list(g) for g in groups if g]
    if not lists:
        return []

    merged, seen = [], set()
    for round_index in range(max(len(g) for g in lists)):
        for group in lists:
            if round_index >= len(group):
                continue
            hit = group[round_index]
            if not isinstance(hit, dict):
                continue          # захист від сміття з несподіваної гілки
            key = _hit_key(hit)
            if key is None or key in seen:
                continue
            seen.add(key)
            merged.append(hit)
            if len(merged) >= limit:
                return merged
    return merged
