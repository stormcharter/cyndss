"""Бойовий шар сценаріїв: опис інфраструктури + корпус OSINT -> сценарії.

НАВІЩО. Орієнтований типізований граф виміряний на контрольному наборі
(`experiments/htb_reconstruction`, гілка E: покриття першого шляху 0,3417 проти
0,2417 у неорієнтованої мережі; на всіх трьох розділах 0,3361 проти 0,1542), але
в UI аналітик досі отримує неорієнтовану мережу співзустрічальності, де стрілка
означає лише порядок читання. Цей модуль підключає його до бойового шляху.

ПОДІЛ ВІДПОВІДАЛЬНОСТІ — головне тут:

    структуру   дає АНАЛІТИК (опис інфраструктури: компоненти, права, вразливості);
    доказовість дає КОРПУС OSINT, зібраний системою.

Структура НЕ інферується з тексту. Витягати переходи з новин мовною моделлю —
той самий клас помилки, що й вигаданий T1190 (375 міток із 380 без опори в
тексті), тільки на рівень вище: там вигадувалась мітка, тут вигадувався б
причинний зв'язок, і помітити це було б значно важче. Тому граф будується лише
з іменованих полів опису, а корпус відповідає на окреме питання — які з цих
переходів уже згадувані в зібраних джерелах.

Що робить модуль:
  1. розбирає опис інфраструктури (JSON);
  2. будує орієнтований типізований граф (`twin_bridge`);
  3. проганяє формули (8)-(11): пряме й зворотне розширення, узгодження
     підмереж, перебір допустимих шляхів (`qa/twin.py`);
  4. ранжує альтернативи інтегральною оцінкою J (формули 12-13);
  5. прив'язує ребра до документів корпусу (`qa.twin.verify_edges`).
"""

from __future__ import annotations

import json

from qa.twin import (
    enumerate_scenarios,
    expand_backward,
    expand_forward,
    reconcile,
    verify_edges,
)
from twin_bridge import (
    EXTERNAL_ID,
    build_twin_model,
    rank_scenarios,
    score_scenario,
    transition_edge_ids,
)

#: Скільки кандидатів перебирати до ранжування. Ширше за видачу навмисно:
#: порядок перебору не є порядком якості (див. `rank_scenarios`).
CANDIDATE_LIMIT = 20


def parse_infrastructure_model(raw_text: str) -> dict:
    """Текст -> опис інфраструктури. Кидає ValueError із зрозумілою причиною.

    Формат — той самий, що в наборі експерименту (`htb.writeup.analysis.v1`):
    об'єкт із `components`, за потреби `identities`, `permission_edges`,
    `artifacts`, `vulnerabilities`."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Опис порожній.")
    try:
        model = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Це не коректний JSON: {exc.msg} (позиція {exc.pos}).") from exc
    if not isinstance(model, dict):
        raise ValueError("Очікується об'єкт опису, а не список чи скаляр.")
    if not model.get("components"):
        raise ValueError(
            "У описі немає поля `components` — без переліку компонентів немає з "
            "чого будувати переходи.")
    return model


def _steps(model, path, edge_path) -> list[dict]:
    """Шлях -> перелік ІМЕНОВАНИХ переходів.

    Саме це відрізняє результат від неорієнтованої мережі: там стрілка нічого
    не означає, тут кожен крок має назву, узяту з поля опису."""
    by_id = model.node_by_id
    steps = []
    for i, index in enumerate(edge_path or []):
        edge = model.edges[index]
        steps.append({
            # ID потрібні, щоб прив'язати позначку доказовості: `verify_edges`
            # говорить у термінах ID вузлів, а показуємо ми підписи.
            "from_id": edge.source,
            "from": by_id[edge.source].label or edge.source,
            "from_kind": by_id[edge.source].kind,
            "relation": edge.relation,
            "to_id": edge.target,
            "to": by_id[edge.target].label or edge.target,
            "to_kind": by_id[edge.target].kind,
            "order": i + 1,
        })
    return steps


def build_scenarios(machine: dict, goal: str, documents=None, k: int = 5) -> dict:
    """Опис + ціль + корпус -> ранжовані сценарії з доказовістю.

    `goal` називає аналітик (`root`, `system`, `administrator`). Якщо такого
    стану в описі немає, це кажеться прямо — ціль не добудовується."""
    documents = list(documents or [])
    goal_token = (goal or "").strip().lower()
    if not goal_token:
        return {"found": False, "scenarios": [], "evidence": [],
                "nodes": 0, "edges": 0, "transitions": 0,
                "note": "Не вказано цільовий стан. Наприклад: root, SYSTEM, administrator."}

    model = build_twin_model(machine, goal_token)
    transitions = transition_edge_ids(model)
    base = {"nodes": len(model.nodes), "edges": len(model.edges),
            "transitions": len(transitions), "goal": goal_token,
            # Модель віддаємо разом з розміром графа: захисне плече
            # (`twin_defense.rank_defenses`) рахує набори засобів САМЕ за
            # нею й за `edge_path` шляхів нижче - без моделі довелось би
            # будувати граф вдруге, а це вже друга копія логіки, яка може
            # розійтися з першою.
            "model": model}

    if not model.goal_state_id:
        return {**base, "found": False, "scenarios": [], "evidence": [],
                "note": f"Стану «{goal}» немає в описі інфраструктури. "
                        "Він має з'явитись у `resulting_access`, `runs_as` або "
                        "`security_consequence` хоча б одного запису."}

    context = reconcile(model, expand_forward(model), expand_backward(model))
    # Ідемо лише переходами: структурні відношення (вкладеність, розміщення)
    # лишаються в моделі для читання, але кроком сценарію не є.
    allowed = set(transitions)
    context["edge_ids"] = [i for i in context["edge_ids"] if i in allowed]
    found = enumerate_scenarios(model, context, matched=set(),
                                max_paths=CANDIDATE_LIMIT)
    ranked = rank_scenarios(model, found["scenarios"])[:k]

    if not ranked:
        return {**base, "found": False, "scenarios": [], "evidence": [],
                "note": "Допустимих переходів від периметра до цілі не знайдено. "
                        "Ймовірно, бракує запису, який називає дію або вразливість."}

    scenarios = []
    for rank, item in enumerate(ranked, 1):
        path, edge_path = item.get("path") or [], item.get("edge_path") or []
        scenarios.append({
            "rank": rank,
            "j": item["j"],
            "criteria": score_scenario(model, path, edge_path),
            "steps": _steps(model, path, edge_path),
            "labels": [model.node_by_id[n].label or n for n in path],
            "hops": max(len(path) - 1, 0),
            # Індекси ребер шляху - те саме подання, що очікує
            # `twin_defense.rank_defenses(model, attack_paths)`. Без цього
            # поля захисне плече довелось би збирати з `steps`, повторно
            # шукаючи індекс кожного ребра в моделі.
            "edge_path": edge_path,
        })

    # Доказовість: які переходи вже згадувані в зібраному корпусі. Вузли, що не
    # трапляються в жодному документі, до перевірки не потрапляють — інакше
    # «підтвердженням» ставав би збіг за одним загальним словом.
    touched = {nid for s in ranked for nid in (s.get("path") or [])} - {EXTERNAL_ID}
    evidence = verify_edges(model, documents, touched) if documents else []

    return {**base, "found": True, "scenarios": scenarios, "evidence": evidence,
            "candidates": found["total_found"], "note": ""}
