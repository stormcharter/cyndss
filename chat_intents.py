"""
Розпізнавання наміру повідомлення в чаті ONYX.

НАВІЩО. Раніше кожна дія мала власне поле й власну кнопку: одне поле під запит
до ради агентів, друге — під ціль гіпотез, окремі кнопки під експорт GraphML /
GEXF. Науковий керівник справедливо назвав це незручним. Тепер вхід ОДИН —
чат, тому з тексту треба зрозуміти, що робити:

    «згенуй мені семантичну мережу понять»  -> semantic_network
    «збудуй граф гіпотез про ransomware»    -> hypothesis_graph (goal=ransomware)
    «покажи граф звʼязків»                  -> cti_graph
    «розкажи про GitMCP вразливість»        -> analysis  (звичайна відповідь агентів)

Свідомо БЕЗ LLM: класифікація наміру має бути миттєвою, детермінованою і
працювати навіть коли зовнішній API лежить. Це маршрутизатор, а не аналітик.

Модуль чистий — жодного streamlit/мережі, тестується ізольовано.
"""

from __future__ import annotations

import re

INTENT_ANALYSIS = "analysis"
INTENT_SEMANTIC_NETWORK = "semantic_network"
INTENT_HYPOTHESIS_GRAPH = "hypothesis_graph"
INTENT_CTI_GRAPH = "cti_graph"

# Апостроф в українській набирають щонайменше трьома символами — зводимо до
# одного, щоб «звʼязків» / «зв'язків» / «зв’язків» матчились однаково.
_APOSTROPHES = "'’ʼ`´"

# Тригери наміру. Кожен елемент — кортеж підрядків, які мають бути ВСІ
# (логічне І), щоб випадкова згадка слова «граф» чи «мережа» не спрацьовувала.
_HYPOTHESIS_TRIGGERS = (
    ("гіпотез",), ("гипотез",), ("hypothes",),
    ("ланцюг", "міркув"),
)
_SEMANTIC_TRIGGERS = (
    ("семантичн", "мереж"), ("семантическ", "сет"),
    ("мереж", "понять"), ("сеть", "понятий"), ("карт", "понять"),
    ("semantic", "network"), ("concept", "map"),
)
_CTI_GRAPH_TRIGGERS = (
    ("граф", "звязк"), ("граф", "связ"), ("граф", "кореляц"),
    ("граф", "корреляц"), ("cti", "graph"), ("graph", "link"),
)

# Маркери, після яких у команді йде ЦІЛЬ: «граф гіпотез ПРО ransomware».
# Порядок важливий: довші/специфічніші раніше, щоб «щодо» не з'їлось «до».
_GOAL_MARKERS = (" щодо ", " про ", " для ", " до ", " навколо ",
                 " about ", " for ", " around ", ":")

# Ціль у лапках — пріоритетний спосіб задати кілька слів.
_QUOTED_RE = re.compile(r"[«\"'“„]([^«»\"'“”„]{2,80})[»\"'”]")

# --- Стартові концепти: «граф гіпотез ВІД фішингу ДО ransomware» -----------
# Задати старт можна лише явно. Без нього обхід іде односторонньо від цілі
# назад (як було), зі стартом — двофронтово назустріч. Дві різні команди, бо
# це два різні алгоритми, і користувач має бачити, який саме він запустив.
_START_MARKERS = (" від ", " from ")

# Що відділяє старт від цілі. Стрілка — коротка форма для тих, хто не хоче
# писати «до».
_PATH_SEPARATORS = (" до ", " to ", "→", "->")

# Кілька стартів: «від фішингу, CVE-2024-1234 та lockbit до ransomware».
_START_SPLIT_RE = re.compile(r"\s*(?:,|;|\bта\b|\bі\b|\band\b|\+)\s*", re.IGNORECASE)

_TRIM = "«»\"'“”„.,!?;: \t"


def _normalize(text: str) -> str:
    """Нижній регістр + єдиний апостроф — для порівняння з тригерами."""
    out = text.lower()
    for apo in _APOSTROPHES:
        out = out.replace(apo, "")
    return out


def _matches(norm: str, triggers) -> bool:
    return any(all(part in norm for part in group) for group in triggers)


def _extract_goal(text: str) -> str | None:
    """Витягує ціль команди: спершу з лапок, інакше — хвіст після маркера
    («про», «для», «about»…). Повертає None, якщо цілі не вказано — вигадувати
    її тут не можна, це рішення UI (спитати або підставити свою)."""
    quoted = _QUOTED_RE.search(text)
    if quoted:
        return quoted.group(1).strip() or None

    lowered = text.lower()
    best = None
    for marker in _GOAL_MARKERS:
        pos = lowered.find(marker)
        if pos == -1:
            continue
        # Беремо НАЙПІЗНІШИЙ маркер: у «збудуй граф гіпотез про ransomware»
        # ціль іде після останнього службового слова, а не після першого.
        end = pos + len(marker)
        if best is None or end > best:
            best = end
    if best is None:
        return None

    # Регістр НЕ чіпаємо: ціль іде у граф як ідентифікатор вузла (CVE-…, MITRE …).
    tail = text[best:].strip()
    tail = tail.strip("«»\"'“”„.,!?;: \t")
    return tail or None


def _split_start_goal(text: str) -> tuple[list[str], str | None]:
    """Розбирає «… від X [, Y та Z] до GOAL» → (["X","Y","Z"], "GOAL").

    Якщо явного старту немає або після нього немає роздільника цілі —
    повертає ([], звичайна_ціль). Неповну команду («від X» без цілі) свідомо
    трактуємо як відсутність старту: домислювати ціль тут не можна.

    Індекси рахуємо по `text.lower()`, а НЕ по `_normalize()`: той прибирає
    апострофи й зсуває позиції відносно оригіналу."""
    lowered = text.lower()

    start_end = -1
    for marker in _START_MARKERS:
        pos = lowered.find(marker)
        if pos != -1 and (start_end == -1 or pos + len(marker) < start_end):
            start_end = pos + len(marker)
    if start_end == -1:
        return [], _extract_goal(text)

    rest = text[start_end:]
    rest_low = rest.lower()
    sep_pos, sep_len = -1, 0
    for sep in _PATH_SEPARATORS:
        pos = rest_low.find(sep)
        if pos != -1 and (sep_pos == -1 or pos < sep_pos):
            sep_pos, sep_len = pos, len(sep)
    if sep_pos == -1:
        return [], _extract_goal(text)

    goal = rest[sep_pos + sep_len:].strip().strip(_TRIM)
    starts = [part.strip().strip(_TRIM)
              for part in _START_SPLIT_RE.split(rest[:sep_pos])]
    starts = [part for part in starts if part]
    if not goal or not starts:
        return [], _extract_goal(text)
    # Регістр НЕ чіпаємо: старт іде у граф як ідентифікатор вузла.
    return starts, goal


def detect_intent(message) -> dict:
    """Повертає {"kind": <INTENT_*>, "goal": str|None, "query": str}.

    "goal" — тема/ціль КОМАНДИ; заповнюється для всіх графових намірів. Для
    звичайного питання (analysis) лишається None: там темою є весь текст.

    Чому це принципово: у «згенуй мені семантичну мережу понять» слова
    «семантичну мережу понять» — сама команда, а не тема. Віддавши цей текст у
    пошук, ми знайшли б документи зі словами «мережа» й «поняття» й побудували
    мережу по них — тобто по випадковому зрізу замість усього корпусу. Немає
    теми -> None, і UI бере корпус цілком."""
    text = (str(message) if message else "").strip()
    if not text:
        return {"kind": INTENT_ANALYSIS, "goal": None, "starts": [], "query": ""}

    norm = _normalize(text)

    # Порядок перевірок: гіпотези раніше за граф звʼязків, бо «граф гіпотез»
    # містить слово «граф» і інакше впало б не в ту гілку.
    for triggers, kind in ((_HYPOTHESIS_TRIGGERS, INTENT_HYPOTHESIS_GRAPH),
                           (_SEMANTIC_TRIGGERS, INTENT_SEMANTIC_NETWORK),
                           (_CTI_GRAPH_TRIGGERS, INTENT_CTI_GRAPH)):
        if _matches(norm, triggers):
            starts, goal = _split_start_goal(text)
            return {"kind": kind, "goal": goal, "starts": starts, "query": text}

    return {"kind": INTENT_ANALYSIS, "goal": None, "starts": [], "query": text}
