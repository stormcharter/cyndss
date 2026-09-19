"""Оцінка сценаріїв захисту за критеріями через мовну модель.

НАВІЩО. `qa.twin.enumerate_scenarios` виділяє допустимі шляхи S0->ST і
рахує їхню СТРУКТУРНУ оцінку (вага ребер, покриття доказами), але критерії
методу (`TwinCriterion` з фікстури - `technical_feasibility`, `risk`, `time`,
`resources`, `legal`, `consequences`, `content_fit`) вимагають судження, яке
з голого графа не витягти. Цей модуль закриває саме цю прогалину: бере
сценарій, описує його компактним текстом і питає LLM per-критерій, а
інтегральну оцінку J рахує ЧЕРЕЗ наявну `qa.twin.aggregate` - не дублюючи
арифметику benefit/cost-інверсії вдруге.

ЧЕСНІСТЬ. Це самозвіт мовної моделі, а не зовнішня верифікація (на відміну
від NVD/MITRE в `qa.twin.verify_edges`). Тому парсер НІКОЛИ не підставляє
середнє чи нуль замість відповіді, яку не вдалось розібрати - невдача
позначається явно (`parse_failed`), і хто завгодно вище по стеку бачить її,
а не гарну на вигляд, але вигадану цифру.

МЕРЕЖА - НЕ ТУТ. Модуль не імпортує httpx і нічого не знає про
OpenRouter/Hermes/ключі: `backend` - параметр із протоколом `CriterionBackend`
(один текстовий запит - один рядок відповіді). Реальний HTTP-клієнт живе в
`tools/run_12node_eval.py`, тестовий - `FakeBackend` у
`tests/test_criterion_eval.py`. Той самий поділ, що й у `qa/twin.py`:
"LLM calls ... stay in qa.agent; this module is importable from both without
cycles" - тут те саме про мережу, лише інший модуль-межа.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from qa.twin import TwinCriterion, TwinModel, aggregate, merge_panel_votes

#: Скільки разів пробуємо отримати розбірний JSON від моделі за ОДИН запит
#: критерію, перш ніж чесно здатись. Контекст моделі малий (4096 токенів) і
#: черга запитів одна (`OLLAMA_NUM_PARALLEL=1`, ~3с/запит) - нескінченні
#: ретраї тут коштують реального часу експерименту, тож межа сувора.
MAX_ATTEMPTS = 3

#: Стани доказовості, які модель МАЄ ПРАВО повернути (частина контракту
#: відповіді). "parse_failed" - НЕ серед них: це наш власний маркер відмови
#: парсера, модель його ніколи не бачить і не повертає.
ALLOWED_EVIDENCE_STATES = {"sufficient", "insufficient", "contradictory"}

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_CRITERION_SYSTEM_PROMPT = (
    "You are a cybersecurity defense scenario evaluator. "
    "You score exactly ONE criterion for ONE defense scenario at a time. "
    "Respond with a single JSON object and nothing else, in exactly this shape: "
    '{"score": <number between 0 and 1>, "confidence": <number between 0 and 1>, '
    '"evidence_state": "sufficient" | "insufficient" | "contradictory", '
    '"rationale": "<one short sentence>"}. '
    "score=0 means the criterion is fully unmet, score=1 means fully met. "
    "Do not add any text before or after the JSON object."
)


class CriterionBackend(Protocol):
    """Мінімальний контракт бекенду оцінювання: один текстовий запит - один
    рядок текстової відповіді. Реальна реалізація ходить у OpenAI-сумісний
    `/chat/completions` (як `ui.py::_llm_generate`, лише інша адреса/ключ);
    у тестах підставляється фальшивка з чергою готових відповідей, щоб жоден
    тест цього модуля не смикав мережу."""

    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        ...


def describe_scenario(model: TwinModel, path: list[str], edge_path: list[int]) -> str:
    """Компактний текстовий опис одного сценарію для запиту до LLM.

    ЛИШЕ МІТКИ, БЕЗ ВНУТРІШНІХ ID. Технічний ідентифікатор вузла
    (`ctrl-fw`, `state-vulnerable`) нічого не додає до сенсу для моделі, що
    оцінює сценарій захисту, а контекст (4096 токенів) обмежений і довший
    промпт РІЖЕТЬСЯ МОВЧКИ - витрачати його на ідентифікатори немає сенсу.
    Людський `label` несе те саме значення компактніше.
    """
    node_by_id = model.node_by_id

    # Дія (`action`), якою вхід до вузла був здійснений, якщо вона названа на
    # ребрі (типово - для доступних станів, `qa.twin.ACCESS_STATE_TYPES`).
    # Це єдине, заради чого сюди взагалі приходить `edge_path`: сам порядок
    # заходів достатньо читати з `path`.
    action_by_target: dict[str, str] = {}
    for index in edge_path:
        edge = model.edges[index]
        action = str(edge.attributes.get("action") or "").strip()
        if action:
            action_by_target[edge.target] = action

    measures: list[str] = []
    for node_id in path:
        node = node_by_id[node_id]
        if node.kind != "control":
            continue
        details: list[str] = []
        cost = node.attributes.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            details.append(f"cost={cost:g}")
        personnel = node.attributes.get("personnel")
        if isinstance(personnel, (int, float)) and not isinstance(personnel, bool):
            details.append(f"personnel={personnel:g}")
        action = action_by_target.get(node_id)
        if action:
            details.append(f"action={action}")
        suffix = f" ({', '.join(details)})" if details else ""
        measures.append(f"{node.label}{suffix}")

    states = [
        node_by_id[node_id].label for node_id in path if node_by_id[node_id].kind == "state"
    ]

    lines = [
        "States traversed: " + (" -> ".join(states) if states else "(none)"),
        "Measures in order: "
        + ("; ".join(f"{i}. {label}" for i, label in enumerate(measures, start=1))
           if measures else "(none)"),
    ]
    return "\n".join(lines)


def _build_user_prompt(scenario_text: str, criterion: TwinCriterion) -> str:
    return (
        f"Criterion: {criterion.name}\n"
        f"Criterion meaning: {criterion.description}\n\n"
        f"Defense scenario:\n{scenario_text}\n\n"
        "Score this scenario on this criterion only. Return the JSON object now."
    )


def _clip01(value: float) -> tuple[float, bool]:
    """Обрізає число в [0, 1]; повертає (значення, чи довелось обрізати)."""
    if value < 0.0:
        return 0.0, True
    if value > 1.0:
        return 1.0, True
    return value, False


def _parse_criterion_json(raw_text: str) -> dict | None:
    """Витягує з відповіді моделі один JSON-обʼєкт з оцінкою критерію.

    JSON-режим Ollama/Hermes валідний, але не гарантований (відомо з
    попередніх вимірів) - модель іноді обгортає JSON поясненням. Тому спершу
    пробуємо строгий `json.loads` усього тексту, а якщо не вийшло - шукаємо
    перший фрагмент у фігурних дужках (той самий прийом, що й
    `shipper/extractor.py`: перша `{` .. остання `}`). Повертає None, якщо
    жоден варіант не дав валідного обʼєкта з полем `score`."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    candidates = [raw_text]
    match = _JSON_OBJECT_RE.search(raw_text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(parsed, dict) or "score" not in parsed:
            continue
        try:
            score = float(parsed["score"])
        except (TypeError, ValueError):
            continue
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence_state = str(parsed.get("evidence_state") or "").strip().lower()
        if evidence_state not in ALLOWED_EVIDENCE_STATES:
            # Модель не назвала доказовість чи назвала щось поза контрактом -
            # консервативний дефолт, а не вигадана впевненість.
            evidence_state = "insufficient"
        rationale = str(parsed.get("rationale") or "").strip()
        return {
            "score": score,
            "confidence": confidence,
            "evidence_state": evidence_state,
            "rationale": rationale,
        }
    return None


def score_criterion(
    scenario_text: str,
    criterion: TwinCriterion,
    backend: CriterionBackend,
    temperature: float = 0.0,
) -> dict[str, object]:
    """Один запит до LLM на один критерій одного сценарію.

    Повертає при успіху:
        {"score": float у [0,1], "confidence": float у [0,1],
         "evidence_state": "sufficient"|"insufficient"|"contradictory",
         "rationale": str, "raw_responses": list[str]}
    (плюс "clipped": True, якщо сирий score виходив за межі [0,1]).

    Повертає при невдачі (`MAX_ATTEMPTS` спроб не дали розбірного JSON):
        {"score": None, "confidence": 0.0, "evidence_state": "parse_failed",
         "rationale": "...", "parse_failed": True, "raw_responses": [...]}
    Явний маркер, а НЕ тиха підстановка нуля чи середнього - інакше провал
    моделі виглядав би як реальна низька оцінка сценарію."""
    system_prompt = _CRITERION_SYSTEM_PROMPT
    user_prompt = _build_user_prompt(scenario_text, criterion)
    raw_responses: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        try:
            raw = backend.generate(system_prompt, user_prompt, temperature)
        except Exception as error:  # мережа флаки (429/503 тощо) - це теж невдала спроба
            raw = f"<backend error: {error}>"
        raw_responses.append(raw)
        parsed = _parse_criterion_json(raw)
        if parsed is None:
            continue
        score, clipped = _clip01(parsed["score"])
        confidence, _ = _clip01(parsed["confidence"])
        result: dict[str, object] = {
            "score": score,
            "confidence": confidence,
            "evidence_state": parsed["evidence_state"],
            "rationale": parsed["rationale"],
            "raw_responses": list(raw_responses),
        }
        if clipped:
            result["clipped"] = True
            result["rationale"] = (
                f"[score clipped from {parsed['score']!r} to {score}] {parsed['rationale']}"
            ).strip()
        return result

    return {
        "score": None,
        "confidence": 0.0,
        "evidence_state": "parse_failed",
        "rationale": f"не вдалося розібрати JSON-відповідь моделі за {MAX_ATTEMPTS} спроби",
        "parse_failed": True,
        "raw_responses": raw_responses,
    }


def evaluate_scenarios(
    model: TwinModel,
    scenarios: list[dict[str, object]],
    backend: CriterionBackend,
    repeats: int = 1,
    temperature: float = 0.0,
) -> dict[str, object]:
    """Оцінює кожен сценарій за КОЖНИМ критерієм моделі (`model.criteria`).

    Кожен критерій отримує `repeats` незалежних вимірів (`score_criterion`),
    які зводяться в один голос через `qa.twin.merge_panel_votes` (медіана,
    той самий код, що вже обслуговує панель з кількох оцінювачів - тут
    оцінювач один, але повторів кілька). Зведені голоси йдуть у
    `qa.twin.aggregate` - інтегральна оцінка J рахується НЕ вдруге, а тим
    самим кодом, що й для шляхів атаки / наборів захисту.

    ЯКЩО КРИТЕРІЙ НЕ РОЗІБРАВСЯ ЖОДНОГО РАЗУ. J сценарію стає None, а назва
    критерію потрапляє в `unresolved_criteria`. Мовчки виключити провальний
    критерій із зважування (перерахувавши J лише за рештою) означало б
    видати число, що виглядає повним, а насправді описує щось інше -
    те саме порушення чесності, що заборонене на рівні одного запиту,
    застосоване й тут."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    results: list[dict[str, object]] = []
    for scenario in scenarios:
        path = list(scenario.get("path") or [])
        edge_path = list(scenario.get("edge_path") or [])
        scenario_text = describe_scenario(model, path, edge_path)

        per_criterion: dict[str, object] = {}
        aggregate_scores: list[dict[str, object]] = []
        unresolved: list[str] = []
        for criterion in model.criteria:
            raw_votes = [
                score_criterion(scenario_text, criterion, backend, temperature=temperature)
                for _ in range(repeats)
            ]
            successful = [vote for vote in raw_votes if not vote.get("parse_failed")]
            if successful:
                merged = merge_panel_votes(successful)
                aggregate_scores.append({"criterion": criterion.name, "score": merged["score"]})
            else:
                merged = None
                unresolved.append(criterion.name)
            per_criterion[criterion.name] = {"raw_votes": raw_votes, "merged": merged}

        if unresolved:
            integral, disagreement = None, None
        else:
            integral, disagreement = aggregate(aggregate_scores, model.criteria)

        results.append(
            {
                "path": path,
                "edge_path": edge_path,
                "scenario_text": scenario_text,
                "criteria": per_criterion,
                "unresolved_criteria": unresolved,
                "J": integral,
                "disagreement": disagreement,
            }
        )

    return {"scenarios": results, "repeats": repeats, "temperature": temperature}
