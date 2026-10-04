"""Пункти 7-8: легенда простою мовою й анкета «як розв'язати проблему?».

7  Кожен ланцюг семантичної мережі мовна модель переказує для людини без
   технічної освіти. Без моделі працює шаблонна легенда.
8  Анкета — одне питання «як розв'язати проблему?» з варіантами-ланцюгами.
   Респондент (вторинний користувач) обирає, який варіант найкращий за часом,
   який за ціною і який за впливом. Відповідей може бути скільки завгодно
   (Л1, Л2, Л3, …); синтез зводить їх голосуванням і порівнює з вибором
   розрахунку. ШІ-експерти відповідають на ту саму анкету: перший раунд —
   незалежно й без балів розрахунку, другий (Дельфі) — бачачи анонімні
   відповіді інших.

Ланцюг тут — словник із `semantic_bridge.rank_chains`: кроки з балами 1-5,
сумарні `cost`/`time` у балах і `impact` у частках одиниці.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from semantic_bridge import extract_json

Generate = Callable[[str, str, str], str]

CRITERION_TITLE = {"cost": "Ціна", "time": "Час", "impact": "Вплив"}


def level(points: float, steps: int) -> str:
    """Сума балів -> слово за середнім балом на крок (для людини без шкал)."""
    mean = points / steps if steps else 0
    return "низька" if mean <= 2 else "середня" if mean <= 3.5 else "висока"


def option_facts(chain: dict) -> str:
    """Факти ланцюга словами: кроки, ціна й час (бали та рівень), вплив у відсотках."""
    n = len(chain["steps"])
    lines = [f"Ціна: {level(chain['cost'], n)} ({chain['cost']} б.). "
             f"Час: {'довгий' if level(chain['time'], n) == 'висока' else 'короткий' if level(chain['time'], n) == 'низька' else 'середній'} "
             f"({chain['time']} б.). Проблему буде розв'язано приблизно на "
             f"{round(chain['impact'] * 100)}%."]
    for i, step in enumerate(chain["steps"], 1):
        lines.append(f"Крок {i}: {step['label']} (ціна {step['cost']}/5, час {step['time']}/5, "
                     f"вплив {step['impact']}/5).")
    return "\n".join(lines)


# =========================================================================
#  Пункт 7: легенда простою мовою
# =========================================================================

def plain_legend(chain: dict) -> str:
    """Легенда без моделі — запасний варіант, щоб анкета працювала й без ШІ."""
    steps = [step["label"][0].lower() + step["label"][1:] for step in chain["steps"]]
    n = len(steps)
    return (f"Спочатку {'; потім '.join(steps)}. Ціна такого плану {level(chain['cost'], n)}, "
            f"часу знадобиться {'багато' if level(chain['time'], n) == 'висока' else 'небагато' if level(chain['time'], n) == 'низька' else 'помірно'}. "
            f"Проблему буде розв'язано приблизно на {round(chain['impact'] * 100)}%.")


LEGEND_SYSTEM = (
    "Ти пояснюєш план розв'язання проблеми з кібербезпеки людині БЕЗ технічної "
    "освіти (керівнику, бухгалтеру). Пиши українською, 3-5 коротких простих речень, "
    "без заголовків і списків. Не вживай абревіатур і жаргону; якщо без терміна не "
    "обійтись — поясни його в дужках звичайними словами. Можна побутові порівняння "
    "(замок, охоронець, запасний ключ). Скажи: що зробимо і в якому порядку, навіщо, "
    "чи дорого це і чи довго, наскільки це розв'яже проблему. Цифри бери ЛИШЕ з "
    "наданих і не змінюй їх; відсоток формулюй як «проблему буде розв'язано "
    "приблизно на N%». Не обіцяй того, чого немає в кроках."
)


def write_legend(generate: Generate, case: dict, chain: dict) -> str:
    user = (f"Проблема: {case['problem']}\nБажаний результат: {case['solution']}\n\n"
            f"План:\n{option_facts(chain)}")
    return generate(LEGEND_SYSTEM, user, "legend").strip()


# =========================================================================
#  Пункт 8: анкета, відповіді, підсумок
# =========================================================================

OPTION_KEYS = "АБВГДЕЖЗИК"

SURVEY_QUESTIONS = {
    "time": "Яке рішення найкраще за ЧАСОМ — найшвидше дасть результат?",
    "cost": "Яке рішення найкраще за ЦІНОЮ — найвигідніше для організації?",
    "impact": "Яке рішення найкраще за ВПЛИВОМ — найкраще розв'яже проблему?",
}

#: Ланцюги, що ділять стільки кроків (частка Жаккара), — для анкети один варіант.
NEAR_DUPLICATE = 0.6


def _step_set(chain: dict) -> frozenset:
    return frozenset(" ".join(s["label"].lower().split()) for s in chain["steps"])


def diverse_chains(chains: list[dict], limit: int = len(OPTION_KEYS),
                   max_overlap: float = NEAR_DUPLICATE) -> list[dict]:
    """Ранжовані ланцюги -> різні за суттю варіанти анкети.

    Ланцюг, чиї кроки здебільшого повторюють уже взятий кращий ланцюг, у
    анкету не йде: респондент інакше обирає між майже однаковими планами,
    а ШІ-агенти чесно пишуть «варіанти А, Б, В однакові»."""
    picked: list[dict] = []
    for chain in chains:
        steps = _step_set(chain)
        if any(len(steps & _step_set(p)) / len(steps | _step_set(p)) >= max_overlap
               for p in picked):
            continue
        picked.append(chain)
        if len(picked) >= limit:
            break
    return picked


def build_survey(case: dict, chains: list[dict], legends: dict | None = None) -> dict:
    """Ланцюги -> анкета. Ідентифікатор залежить від задачі й набору варіантів,
    тож відповіді не змішуються між різними наборами ланцюгів."""
    legends = legends or {}
    chains = chains[:len(OPTION_KEYS)]
    digest = hashlib.sha1(json.dumps(
        [case["problem"], case["solution"]] + [[s["label"] for s in c["steps"]] for c in chains],
        ensure_ascii=False).encode()).hexdigest()
    options = [{
        "key": OPTION_KEYS[i],
        "rank": chain["rank"],
        "title": " → ".join(s["label"] for s in chain["steps"]),
        "steps": [s["label"] for s in chain["steps"]],
        "legend": legends.get(chain["rank"]) or plain_legend(chain),
        "cost": chain["cost"], "time": chain["time"], "impact": chain["impact"],
        "n": len(chain["steps"]),
    } for i, chain in enumerate(chains)]
    return {"id": f"{case.get('id', 'case')}-{digest[:10]}", "problem": case["problem"],
            "solution": case["solution"], "context": case.get("context", ""),
            "question": "Як розв'язати цю проблему?",
            "questions": dict(SURVEY_QUESTIONS), "options": options}


def survey_text(survey: dict, numbers: bool = True, order: list[str] | None = None) -> str:
    """Анкета одним текстом.

    `numbers=False` — для ШІ-експерта: лише кроки, без балів і легенди (вона
    теж переказує бали). Інакше експерт просто обирає найменшу ціну й
    найбільший відсоток — повторює розрахунок замість власної думки.
    `order` — порядок, у якому показати варіанти (літери лишаються свої)."""
    by_key = {o["key"]: o for o in survey["options"]}
    keys = order or [o["key"] for o in survey["options"]]
    lines = [f"Проблема: {survey['problem']}", f"Бажаний результат: {survey['solution']}"]
    if not numbers and survey.get("context"):
        lines += ["", "Контекст мережі:", survey["context"]]
    lines += ["", survey["question"], ""]
    for key in keys:
        option = by_key[key]
        lines.append(f"Варіант {key}:")
        lines += [f"  {i}. {step}" for i, step in enumerate(option["steps"], 1)]
        if numbers:
            lines.append(f"  Ціна {option['cost']} б., час {option['time']} б. "
                         f"(кроків: {option['n']}), проблему розв'язано приблизно на "
                         f"{round(option['impact'] * 100)}%.")
            lines.append(f"  {option['legend']}")
    lines.append("")
    lines += [f"{c}: {q}" for c, q in survey["questions"].items()]
    return "\n".join(lines)


def make_response(survey: dict, respondent: str, choices: dict, comment: str = "",
                  kind: str = "human", reasons: dict | None = None,
                  estimates: dict | None = None, round_no: int = 1,
                  previous: dict | None = None) -> dict:
    """Перевірена відповідь. Невідомий варіант чи критерій — помилка, а не мовчазний пропуск.

    `reasons` — обґрунтування окремо для кожного питання; `estimates` — власні
    бали респондента (1-5) для кожного варіанта; `previous` — вибір попереднього
    раунду Дельфі, щоб було видно, хто й що змінив."""
    respondent = (respondent or "").strip()
    if not respondent:
        raise ValueError("Вкажіть, хто відповідає.")
    keys = {o["key"] for o in survey["options"]}
    clean = {}
    for criterion in SURVEY_QUESTIONS:
        choice = str(choices.get(criterion) or "").strip().upper()
        if choice not in keys:
            raise ValueError(f"Не обрано варіант для «{CRITERION_TITLE[criterion]}».")
        clean[criterion] = choice
    response = {"respondent": respondent[:80], "kind": kind, "choices": clean,
                "comment": (comment or "").strip()[:1000], "round": round_no,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if reasons:
        response["reasons"] = {c: str(reasons.get(c) or "")[:500] for c in SURVEY_QUESTIONS
                               if reasons.get(c)}
    if estimates:
        response["estimates"] = estimates
    if previous:
        response["previous"] = dict(previous)
    return response


class SurveyStore:
    """Відповіді анкет у JSON-файлах, по файлу на анкету. Файлове сховище, а не
    стан сесії: анкету заповнюють різні люди з різних браузерів."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, survey_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", survey_id)
        return self.root / f"{safe}.json"

    def load(self, survey_id: str) -> list[dict]:
        path = self._path(survey_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _write(self, survey_id: str, responses: list[dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path(survey_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(responses, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._path(survey_id))

    def add(self, survey_id: str, response: dict) -> list[dict]:
        """Повторна відповідь того самого респондента замінює попередню."""
        responses = [r for r in self.load(survey_id)
                     if r.get("respondent") != response["respondent"]]
        responses.append(response)
        self._write(survey_id, responses)
        return responses

    def remove(self, survey_id: str, respondent: str) -> list[dict]:
        responses = [r for r in self.load(survey_id) if r.get("respondent") != respondent]
        self._write(survey_id, responses)
        return responses


def tally(survey: dict, responses: list[dict], weights: dict | None = None) -> dict:
    """Голоси -> переможець за кожним критерієм і загальний.

    Загальна оцінка варіанта = зважена сума часток голосів за критеріями
    (ваги — ті самі, що в ранжуванні). Нічия називається нічиєю, а не
    вирішується порядком варіантів."""
    weights = {**{"cost": 1.0, "time": 1.0, "impact": 1.0}, **(weights or {})}
    keys = [o["key"] for o in survey["options"]]
    counts = {c: {k: 0 for k in keys} for c in SURVEY_QUESTIONS}
    for response in responses:
        for criterion, choice in (response.get("choices") or {}).items():
            if criterion in counts and choice in counts[criterion]:
                counts[criterion][choice] += 1
    n = len(responses)
    winners = {}
    for criterion, votes in counts.items():
        top = max(votes.values(), default=0)
        winners[criterion] = [k for k, v in votes.items() if v == top] if top else []
    total_w = sum(weights[c] for c in SURVEY_QUESTIONS) or 1.0
    overall = sorted(
        ({"key": k, "score": round(sum(weights[c] * counts[c][k] / n for c in counts)
                                   / total_w, 4) if n else 0.0} for k in keys),
        key=lambda item: (-item["score"], keys.index(item["key"])))
    best = [o["key"] for o in overall if n and o["score"] == overall[0]["score"]]
    return {"n": n, "counts": counts, "winners": winners, "overall": overall, "best": best}


def system_choice(survey: dict) -> dict:
    """Що обрав би сам розрахунок: найдешевший, найшвидший, найсильніший варіант.
    Показується поруч із голосами, щоб було видно, де люди не погодились із цифрами."""
    options = survey["options"]
    if not options:
        return {}
    return {
        "cost": min(options, key=lambda o: (o["cost"], o["key"]))["key"],
        "time": min(options, key=lambda o: (o["time"], o["key"]))["key"],
        "impact": sorted(options, key=lambda o: (-o["impact"], o["key"]))[0]["key"],
    }


def expert_estimates(survey: dict, responses: list[dict]) -> dict:
    """Середні власні бали експертів для кожного варіанта: {key: {criterion: mean}}."""
    sums: dict[str, dict[str, list[int]]] = {}
    for response in responses:
        for key, values in (response.get("estimates") or {}).items():
            for criterion in SURVEY_QUESTIONS:
                value = values.get(criterion) if isinstance(values, dict) else None
                if isinstance(value, int):
                    sums.setdefault(key, {}).setdefault(criterion, []).append(value)
    return {key: {c: round(sum(v) / len(v), 2) for c, v in by_c.items()}
            for key, by_c in sums.items()}


# =========================================================================
#  ШІ-експерти: незалежний перший раунд і раунд Дельфі
# =========================================================================

#: Ролі відрізняються не однією фразою, а тим, на що експерт ДИВИТЬСЯ.
#: `temperature` різна, щоб навіть на одній моделі думки не збігалися механічно.
AGENTS = [
    {"id": "L1", "name": "Л1 · ШІ-інженер ІБ", "temperature": 0.4,
     "role": "Ти — практик з інформаційної безпеки, що сам впроваджував такі заходи.",
     "lens": ["чи технічно здійсненні кроки в цій мережі й у такому порядку",
              "чи не пропущено передумову (інвентаризація, доступи, сумісність)",
              "чи справді план закриває найважливіші вузли, а не другорядні",
              "ризик зламати роботу сервісів під час впровадження"]},
    {"id": "L2", "name": "Л2 · ШІ-економіст ризиків", "temperature": 0.7,
     "role": "Ти — фінансовий аналітик з ризиків, скептичний до оптимістичних кошторисів.",
     "lens": ["приховані витрати: ліцензії, навчання людей, підтримка, простій",
              "скільки людей і якої кваліфікації потрібно",
              "що буде, якщо план затягнеться чи бюджет урізають наполовину",
              "співвідношення витрат і реального зниження збитків"]},
    {"id": "L3", "name": "Л3 · ШІ-аудитор відповідності", "temperature": 0.3,
     "role": "Ти — аудитор відповідності (ISO/IEC 27001, 27035) і керівник реагування на інциденти.",
     "lens": ["чи виконано обмеження задачі з контексту мережі",
              "чи зберігається доказова база й журнали для розслідування",
              "чи можна буде реагувати на інцидент у потрібний строк",
              "чи не лишається вузлів, за які ніхто не відповідає"]},
]

AGENT_FORMAT = (
    "Спершу сам оціни КОЖЕН варіант за шкалою 1-5: cost (1 — дешево, 5 — дорого), "
    "time (1 — дні, 5 — багато місяців), impact (1 — майже не допоможе, 5 — вирішить "
    "проблему). Потім на кожне з трьох питань обери ОДИН варіант і поясни вибір зі "
    "своєї ролі одним-двома реченнями — конкретно, з посиланням на кроки чи вузли. "
    "Якщо варіанти здаються тобі рівними, все одно обери й скажи, що вирішило. "
    "Відповідай СТРОГО одним JSON-об'єктом без тексту поза ним:\n"
    '{"estimates": {"А": {"cost": 2, "time": 3, "impact": 4}, "Б": {...}}, '
    '"time": {"choice": "А", "why": "..."}, "cost": {"choice": "Б", "why": "..."}, '
    '"impact": {"choice": "А", "why": "..."}}'
)

DELPHI_FORMAT = (
    "Це другий раунд методу Дельфі. Нижче — твоя відповідь першого раунду й анонімні "
    "відповіді інших експертів. Переглянь свій вибір: зміни його, лише якщо аргументи "
    "інших тебе переконали, і тоді скажи, чим саме; якщо лишаєшся при своїй думці — "
    "поясни, чому аргументи інших не переконали. Відповідай СТРОГО одним JSON-об'єктом:\n"
    '{"time": {"choice": "А", "why": "..."}, "cost": {"choice": "Б", "why": "..."}, '
    '"impact": {"choice": "А", "why": "..."}}'
)

# Моделі інколи відповідають латиницею, схожою на кирилицю (A замість А).
_LOOKALIKE = str.maketrans({"A": "А", "B": "В", "E": "Е", "K": "К"})


def _letter(value) -> str:
    return str(value or "").strip().upper().translate(_LOOKALIKE)[:1]


def agent_order(survey: dict, agent: dict) -> list[str]:
    """Свій порядок показу варіантів для кожного експерта (відтворюваний):
    модель схильна обирати перший показаний варіант, тож «перший» має бути різним."""
    keys = [o["key"] for o in survey["options"]]
    random.Random(f"{survey['id']}:{agent['id']}").shuffle(keys)
    return keys


def _agent_system(agent: dict, tail: str) -> str:
    lens = "\n".join(f"- {item}" for item in agent["lens"])
    return f"{agent['role']}\nНа що ти дивишся насамперед:\n{lens}\n\n{tail}"


def _parse_choices(data: dict) -> tuple[dict, dict]:
    choices, reasons = {}, {}
    for criterion in SURVEY_QUESTIONS:
        item = data.get(criterion)
        if isinstance(item, dict):
            choices[criterion] = _letter(item.get("choice"))
            reasons[criterion] = str(item.get("why") or "").strip()
        else:
            choices[criterion] = _letter(item)
    return choices, reasons


def _parse_estimates(data: dict, keys: set[str]) -> dict:
    out = {}
    raw = data.get("estimates")
    if not isinstance(raw, dict):
        return out
    for key, values in raw.items():
        key = _letter(key)
        if key not in keys or not isinstance(values, dict):
            continue
        clean = {}
        for criterion in SURVEY_QUESTIONS:
            try:
                clean[criterion] = min(max(int(round(float(values.get(criterion)))), 1), 5)
            except (TypeError, ValueError):
                continue
        if clean:
            out[key] = clean
    return out


def ask_agent(generate: Generate, agent: dict, survey: dict) -> dict:
    """Перший раунд: експерт бачить кроки без балів і в своєму порядку.
    Повертає {"response", "error", "raw"}."""
    order = agent_order(survey, agent)
    raw = generate(_agent_system(agent, AGENT_FORMAT),
                   survey_text(survey, numbers=False, order=order), agent["id"])
    try:
        data = extract_json(raw)
        choices, reasons = _parse_choices(data)
        estimates = _parse_estimates(data, {o["key"] for o in survey["options"]})
        response = make_response(survey, agent["name"], choices, kind="agent",
                                 reasons=reasons, estimates=estimates, round_no=1)
        return {"response": response, "error": "", "raw": raw}
    except ValueError as exc:
        return {"response": None, "error": str(exc), "raw": raw}


def _answer_text(response: dict) -> str:
    reasons = response.get("reasons") or {}
    return "\n".join(f"  {CRITERION_TITLE[c]}: варіант {response['choices'][c]}"
                     + (f" — {reasons[c]}" if reasons.get(c) else "")
                     for c in SURVEY_QUESTIONS)


def revise_agent(generate: Generate, agent: dict, survey: dict, own: dict,
                 others: list[dict]) -> dict:
    """Раунд Дельфі: експерт бачить анонімні відповіді інших і може змінити вибір."""
    order = agent_order(survey, agent)
    peers = "\n\n".join(f"Експерт {i}:\n{_answer_text(r)}" for i, r in enumerate(others, 1))
    user = (survey_text(survey, numbers=False, order=order)
            + f"\n\nТВОЯ ВІДПОВІДЬ (раунд 1):\n{_answer_text(own)}"
            + f"\n\nВІДПОВІДІ ІНШИХ ЕКСПЕРТІВ:\n{peers or 'немає'}")
    raw = generate(_agent_system(agent, DELPHI_FORMAT), user, f"{agent['id']}:delphi")
    try:
        choices, reasons = _parse_choices(extract_json(raw))
        response = make_response(survey, agent["name"], choices, kind="agent",
                                 reasons=reasons, estimates=own.get("estimates"),
                                 round_no=2, previous=own["choices"])
        return {"response": response, "error": "", "raw": raw}
    except ValueError as exc:
        return {"response": None, "error": str(exc), "raw": raw}


def changes(response: dict) -> dict:
    """{criterion: (було, стало)} для відповіді раунду Дельфі."""
    previous = response.get("previous") or {}
    return {c: (previous[c], response["choices"][c]) for c in SURVEY_QUESTIONS
            if previous.get(c) and previous[c] != response["choices"][c]}


SYNTH_SYSTEM = (
    "Ти підбиваєш підсумок опитування про те, як розв'язати проблему з "
    "кібербезпеки. Пиши українською для людини без технічної освіти: 4-8 коротких "
    "простих речень, без абревіатур і жаргону. Скажи, який варіант обрали за "
    "часом, за ціною і за впливом, чи були розбіжності між учасниками й чому (з їхніх "
    "обґрунтувань), чи хтось змінив думку в другому раунді, чи збігся вибір із "
    "розрахунком системи, і який варіант рекомендовано загалом. Цифри й літери "
    "варіантів бери лише з наданого. Рекомендованим називай САМЕ варіант із рядка "
    "«ЗАГАЛЬНИЙ ВИБІР» і не підміняй його своєю думкою; якщо там нічия — так і скажи."
)


def synthesize(generate: Generate, survey: dict, responses: list[dict],
               result: dict) -> str:
    calc = system_choice(survey)
    lines = [survey_text(survey), "", f"Відповідей: {result['n']}."]
    for criterion, question in survey["questions"].items():
        votes = ", ".join(f"{k}: {v}" for k, v in result["counts"][criterion].items() if v)
        winners = result["winners"][criterion]
        top = result["counts"][criterion][winners[0]] if winners else 0
        verdict = ("голосів немає" if not winners else
                   f"нічия між {', '.join(winners)}" if len(winners) > 1 else
                   f"обрано {winners[0]} " + ("одностайно" if top == result["n"]
                                              else f"({top} з {result['n']}, думки розділились)"))
        same = "збігається" if winners == [calc.get(criterion)] else "НЕ збігається"
        lines.append(f"{question} Голоси — {votes or 'немає'}; {verdict}; розрахунок "
                     f"системи — {calc.get(criterion, '—')} ({same} з вибором учасників).")
    lines.append("Загальний рейтинг: " + ", ".join(
        f"{o['key']} ({o['score']:.2f})" for o in result["overall"]))
    best = result["best"]
    lines.append("ЗАГАЛЬНИЙ ВИБІР (рекомендація): " + (
        f"варіант {best[0]}" if len(best) == 1 else
        f"нічия між {', '.join(best)}" if best else "немає відповідей"))
    for response in responses:
        lines.append(f"\nУчасник «{response['respondent']}» (раунд {response.get('round', 1)}):")
        lines.append(_answer_text(response))
        moved = changes(response)
        if moved:
            lines.append("  Змінив думку: " + "; ".join(
                f"{CRITERION_TITLE[c]} {a}→{b}" for c, (a, b) in moved.items()))
        if response.get("comment"):
            lines.append(f"  Коментар: {response['comment']}")
    return generate(SYNTH_SYSTEM, "\n".join(lines), "synth").strip()
