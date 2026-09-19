"""Експеримент: метод на 100 згенерованих корпоративних мережах проти
прихованого еталона (`experiments/corporate_networks/dataset/`).

Чотири незалежні вимірювання (див. звіт задачі для повних чисел):

  (1) Реконструкція шляху    — найкращий (за J) знайдений шлях атаки
                                проти `best_attack` з `answers.json`,
                                зіставлення за CVE.
  (2) Згода ранжування       — чи наш J узгоджується з `effectiveness`
                                еталона на ВСІХ 5 атаках кожної мережі
                                (коефіцієнт Спірмена, частка "найкращий
                                збігається").
  (3) Захисне плече           — `twin_defense.rank_defenses` на знайдених
                                шляхах.
  (4) Базова лінія             — та сама метрика (1), але на неорієнтованій
                                непозначеній проєкції графа (найкоротший
                                шлях), щоб виміряти внесок орієнтованості
                                й типізації.

Еталон (`parse_gold_attacks`, `load_answers`) читається ЛИШЕ після того, як
метод уже видав відповідь: жоден крок побудови графа чи пошуку шляхів не має
доступу до нього. Це стереже `experiments/corporate_networks/dataset.py`
(`test_gold_never_leaks_into_the_model`) — цей модуль лише СПОЖИВАЄ дві
незалежні функції парсера й ніде не змішує їхні результати ДО оцінювання.

Відтворюваність і відомий ризик
--------------------------------
`enumerate_scenarios` (`qa/twin.py`, не чіпаємо) обриває DFS-пошук на
200 кандидатах (`max_paths*10`); на щільних мережах (виміряно на net-001)
ПРОСТИХ шляхів більше за цю межу, і те, ЯКІ саме 200 потраплять у вибірку
ДО переранжування J, залежить від `PYTHONHASHSEED` — `_asset_reachability`
у `dataset.py` (теж не чіпаємо) будує частину ребер через `set` рядків.
Це властивість коду, якого це завдання прямо забороняє редагувати, а не
похибка цього модуля. `main()` тому фіксує `PYTHONHASHSEED=0` і
перезапускає процес ОДИН раз перед обчисленням — так само, як
`--check-determinism` у HTB-експерименті гарантує побайтовий повтор
результату. Наслідок для читання чисел: `found_chains`/`found_hops` та
похідні від них можуть трохи відрізнятись при запуску без цього
фіксатора (наприклад прямим імпортом у REPL) — сама КІЛЬКІСТЬ досяжних
цілей (`networks_goal_reachable`) від цього не залежить (перевірено BFS,
а не DFS-вибіркою).

Запуск:
    python experiments/corporate_networks/run_experiment.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from qa.twin import MAX_DEPTH, TwinCriterion, aggregate, enumerate_scenarios, expand_backward, expand_forward, reconcile  # noqa: E402
from twin_bridge import transition_edge_ids  # noqa: E402
import twin_defense  # noqa: E402

# Модуль має працювати і як частина пакета (`experiments.corporate_networks.
# run_experiment`), і як окремий скрипт — той самий прийом, що й у
# `experiments/htb_reconstruction/run_experiment.py`.
try:  # запуск як пакет
    from experiments.corporate_networks.dataset import (  # noqa: E402
        DEFAULT_CRITERIA, build_twin_model, load_answers, parse_gold_attacks,
        parse_network,
    )
    from experiments.htb_reconstruction.metric import (  # noqa: E402
        ordered_coverage, precision_against_gold, step_coverage,
    )
except ImportError:  # запуск як окремий скрипт
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "htb_reconstruction"))
    from dataset import (  # noqa: E402
        DEFAULT_CRITERIA, build_twin_model, load_answers, parse_gold_attacks,
        parse_network,
    )
    from metric import (  # noqa: E402
        ordered_coverage, precision_against_gold, step_coverage,
    )


# =====================================================================
#  (2) Коефіцієнт рангової кореляції Спірмена — стандартна бібліотека,
#  scipy в середовищі немає. Рахуємо як Пірсона на СЕРЕДНІХ рангах: це
#  загальний спосіб, коректний як з нічиями, так і без них (та сама
#  формула, що й `scipy.stats.spearmanr` за замовчуванням), а не
#  скорочена формула через суму квадратів різниць, яка з нічиями вже
#  неточна.
# =====================================================================

def _average_ranks(values: list[float]) -> list[float]:
    """Значення -> середні ранги (1-based); однакові значення отримують
    однаковий (середній) ранг, щоб порядок не залежав від того, яке з двох
    рівних значень трапилось першим."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Позиції i..j (0-based) -> ранги (i+1)..(j+1); нічия отримує
        # середнє цього діапазону.
        average = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman_rank_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Коефіцієнт Спірмена між `xs` і `ys`, або `None`, якщо не визначений
    (менше двох точок або нульова дисперсія рангів в одному з рядів —
    ділити в такому разі означало б удавати число там, де його немає)."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    rx = _average_ranks(list(xs))
    ry = _average_ranks(list(ys))
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    dx = [r - mean_rx for r in rx]
    dy = [r - mean_ry for r in ry]
    cov = sum(a * b for a, b in zip(dx, dy))
    var_x = sum(a * a for a in dx)
    var_y = sum(b * b for b in dy)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y) ** 0.5


# =====================================================================
#  Зіставлення за CVE: наш шлях і еталонна атака -> сумісні ланцюги для
#  `experiments.htb_reconstruction.metric` (та сама метрика, ті самі
#  функції, лише інша "мова" токенів — CVE замість вільного тексту).
# =====================================================================

def path_cves(model, path: list[str]) -> list[str]:
    """Шлях вузлами -> ланцюг CVE у порядку проходження (лише kind="vuln").

    Аналог `twin_bridge.path_concepts`, адаптований під це зіставлення:
    еталон `<attacks>` називає кроки за CVE, тож і наш шлях зводиться до
    того самого алфавіту токенів — інакше порівняння взагалі неможливе."""
    chain: list[str] = []
    for node_id in path:
        node = model.node_by_id.get(node_id)
        if node is None or node.kind != "vuln":
            continue
        cve = node.attributes.get("cve")
        if cve:
            chain.append(cve)
    return chain


def gold_attack_cve_sets(attack: dict) -> list[set]:
    """Еталонна атака (`parse_gold_attacks`) -> список множин CVE, по одній
    на крок — формат `gold`, який очікує `experiments.htb_reconstruction.
    metric`. Крок без CVE стає ПОРОЖНЬОЮ множиною (а не зникає): інакше
    довжина еталона тихо скоротилась би і покриття виглядало б кращим, ніж
    насправді."""
    return [
        {step["cve"]} if step.get("cve") else set()
        for step in attack.get("steps", [])
    ]


# =====================================================================
#  Ранжування сценаріїв інтегральною оцінкою J (формули 12-13) — аналог
#  `twin_bridge.score_scenario`/`rank_scenarios`, АЛЕ не той самий код:
#  `twin_bridge` жорстко зашиває `EXTERNAL_ID = "S_EXTERNAL"` і
#  `GENERIC_RELATIONS = {"reaches", "runs_as"}` — імена, яких у цьому графі
#  просто немає (тут периметр — "ZONE:internet", а "гола" мережева
#  досяжність виражена не окремим відношенням, а атрибутом
#  `reachability_rule="same_subnet"` на ребрі `exploits`). Застосувати
#  `twin_bridge.rank_scenarios` напряму означало б, що жоден вузол шляху
#  ніколи не дорівнює "S_EXTERNAL" — критерій "evidence" вироджується в
#  константу 1.0 і перестає щось розрізняти. Тому тут — ОКРЕМА, але
#  структурно ідентична пара функцій під власники поля цього датасету;
#  `enumerate_scenarios`/`aggregate` (сама формула J) лишаються ті самі, що
#  й у HTB-гілці E.
#
#  Означення трьох критеріїв — дослівно з `dataset.DEFAULT_CRITERIA`:
#    feasibility — частка переходів, що НАЗИВАЮТЬ конкретний механізм
#                  (вразливість, дозволений потік чи політику), а не
#                  просто мережеву досяжність. Єдиний "голий" перехід у
#                  цьому графі — `exploits` з reachability_rule="same_subnet"
#                  (L2-сумісність без жодного явного дозволу); решта
#                  (has_vuln/flow/policy_allows/yields/grants і
#                  exploits/flow/policy) називають конкретний факт.
#    evidence    — частка вузлів шляху, що є РЕАЛЬНИМИ активами мережі
#                  (kind="asset"), а не синтетичними вузлами стану,
#                  вразливості чи периметра.
#    effort      — довжина шляху, нормована на MAX_DEPTH.
# =====================================================================

def _is_bare_network_reachability(edge) -> bool:
    """Перехід, що НЕ називає жодного конкретного механізму — лише
    мовчазну L2-суміжність підмережі."""
    return edge.relation == "exploits" and edge.attributes.get("reachability_rule") == "same_subnet"


def score_scenario_corp(model, path: list[str], edge_path: list[int]) -> list[dict]:
    """Оцінки сценарію за кожним критерієм `dataset.DEFAULT_CRITERIA`."""
    edges = [model.edges[i] for i in (edge_path or [])]
    named = [e for e in edges if not _is_bare_network_reachability(e)]
    feasibility = len(named) / len(edges) if edges else 0.0

    assets_on_path = [nid for nid in path if model.node_by_id[nid].kind == "asset"]
    evidence = len(assets_on_path) / len(path) if path else 0.0

    hops = max(len(path) - 1, 0)
    effort = min(hops / MAX_DEPTH, 1.0)

    return [
        {"criterion": "feasibility", "score": round(feasibility, 6)},
        {"criterion": "evidence", "score": round(evidence, 6)},
        {"criterion": "effort", "score": round(effort, 6)},
    ]


def rank_scenarios_corp(model, scenarios) -> list[dict]:
    """Сценарії, упорядковані за спаданням J; дублі шляхів прибрані.

    Саме ця функція закриває вимогу «перший шлях не має бути алфавітним
    артефактом DFS»: `enumerate_scenarios` сортує кандидатів за кількістю
    control-вузлів на шляху (тут таких завжди 0 — control-вузли не лежать
    на переходах), тож без переранжування «перший» — це просто
    найкоротший/алфавітно менший з відвіданих DFS шлях, а не найкращий за
    J (перевірено на net-001 у `tests/test_corporate_experiment.py`)."""
    seen: set[tuple] = set()
    scored = []
    for scenario in scenarios or []:
        key = tuple(scenario.get("path") or ())
        if key in seen:
            continue
        seen.add(key)
        integral, disagreement = aggregate(
            score_scenario_corp(model, scenario.get("path") or [],
                                scenario.get("edge_path") or []),
            model.criteria,
        )
        scored.append({**scenario, "j": integral, "disagreement": disagreement})
    scored.sort(key=lambda item: (-item["j"], tuple(item.get("path") or ())))
    return scored


# =====================================================================
#  Критерій exploitability — ЩО ЦЕ ЗАКРИВАЄ. Вимірювання (2) показало
#  нульову згоду ранжування (spearman_mean=0.045, best_match_share=0.179,
#  гірше за випадкове вгадування): еталонна `effectiveness`
#  (`answers.json["scoring"]`, відтворена в `dataset.reproduce_
#  effectiveness`) — це добуток CVSS-`exploitability` по кроках атаки,
#  тобто ЛЕГКІСТЬ ЕКСПЛУАТАЦІЇ, а `score_scenario_corp` про exploitability
#  взагалі не знає — рахує лише СТРУКТУРНУ правдоподібність шляху
#  (feasibility/evidence/effort). Два різні впорядкування -> нульова
#  кореляція очікувана, а не загадкова.
#
#  Ця секція додає ЧЕТВЕРТИЙ критерій, не чіпаючи три структурні: ні
#  `score_scenario_corp`, ні `model.criteria` (=`dataset.DEFAULT_CRITERIA`)
#  не змінюються — рахунок з exploitability йде через ОКРЕМІ функції й
#  ОКРЕМУ константу ваг, щоб "structural_only" і "with_exploitability"
#  можна було порахувати незалежно на тих самих сирих кандидатах
#  `enumerate_scenarios` і показати обидва числа поруч (вимога задачі).
# =====================================================================

def path_exploitability(model, path: list[str], agg: str = "min") -> float:
    """Згортка `exploitability` вузлів-вразливостей ШЛЯХУ в одне число.
    `exploitability` — ПУБЛІЧНЕ поле XML вузла вразливості (`dataset.
    _parse_vulnerability`, `build_twin_model`), доступне методу без жодного
    погляду на `<attacks>`; це не еталон.

    ЧОМУ НЕ ДОБУТОК. Добуток по кроках — це дослівно формула еталона
    (`dataset.reproduce_effectiveness`, `answers.json["scoring"]`).
    Відтворити її тут означало б підмінити вимірювання тавтологією:
    кореляція з еталоном наблизилась би до 1.0 не тому, що метод навчився
    щось розрізняти, а тому, що ми порахували ту саму формулу вдруге під
    іншою назвою.

    ЧОМУ МІНІМУМ (обраний спосіб, `agg="min"` за замовчуванням). "Ланцюг
    не міцніший за найслабшу ланку": якщо БУДЬ-ЯКИЙ крок шляху важко
    експлуатувати, практична складність усього ланцюга визначається саме
    ним — середнє замаскувало б один важкий крок серед кількох легких і
    показало б шлях привабливішим, ніж він є насправді.
    `agg="mean"` лишений як друга, звітна конфігурація (щоб показати обидва
    числа поруч, як дозволяє умова задачі), АЛЕ не бере участі у виборі
    ваг `WITH_EXPLOITABILITY_CRITERIA` нижче — обрана згортка одна, і саме
    вона зашита у скоринг.

    Шлях без жодного вузла-вразливості (наприклад один вузол стану) ->
    чесний нуль: той самий принцип, що вже діє в `evidence`/`feasibility`
    вище ("немає даних" -> 0.0, а не нейтральне/пропущене значення)."""
    values = [
        model.node_by_id[nid].attributes.get("exploitability", 0.0)
        for nid in path
        if model.node_by_id.get(nid) is not None and model.node_by_id[nid].kind == "vuln"
    ]
    if not values:
        return 0.0
    if agg == "mean":
        return sum(values) / len(values)
    return min(values)


#: Ваги J з урахуванням exploitability — підібрані ОДИН РАЗ і зафіксовані
#: константою (вимога задачі: не підганяти під підсумкову кореляцію, тому
#: ці числа НЕ переглядались після того, як стали відомі підсумкові
#: best_match_share/spearman_mean). Обґрунтування розподілу:
#:   - структурні критерії лишаються РАЗОМ важливішими за нову ознаку
#:     (0.3+0.2+0.2=0.7 проти 0.3) — новий критерій істотний, але не
#:     підміняє собою всю оцінку;
#:   - усередині exploitability отримує ту саму вагу, що й найважливіший
#:     структурний критерій (feasibility, 0.3) — свідомий сигнал "легкість
#:     експлуатації настільки ж важлива, наскільки й названість переходу",
#:     не більше і не менше;
#:   - evidence/effort знижені з 0.3 до 0.2 порівну, щоб звільнити вагу під
#:     новий критерій, не змінюючи їхнє співвідношення одне до одного.
WITH_EXPLOITABILITY_CRITERIA = (
    TwinCriterion(
        "feasibility",
        "частка переходів, що називають конкретний механізм, а не просто "
        "мережеву досяжність",
        0.3, "benefit",
    ),
    TwinCriterion(
        "evidence",
        "частка вузлів шляху, що є реальними активами мережі",
        0.2, "benefit",
    ),
    TwinCriterion(
        "effort",
        "довжина шляху, нормована на MAX_DEPTH",
        0.2, "cost",
    ),
    TwinCriterion(
        "exploitability",
        "мінімум exploitability вразливостей шляху — практична легкість "
        "експлуатації найслабшого кроку ланцюга (публічне поле CVSS, не еталон)",
        0.3, "benefit",
    ),
)


def score_scenario_corp_with_exploitability(
    model, path: list[str], edge_path: list[int], agg: str = "min",
) -> list[dict]:
    """Три структурні критерії `score_scenario_corp` (виклик, а не
    переписана копія — щоб дві версії оцінки фізично не могли розійтися в
    тому, як рахуються спільні три критерії) + четвертий — exploitability."""
    scores = score_scenario_corp(model, path, edge_path)
    scores.append({
        "criterion": "exploitability",
        "score": round(path_exploitability(model, path, agg=agg), 6),
    })
    return scores


def rank_scenarios_corp_with_exploitability(model, scenarios, agg: str = "min") -> list[dict]:
    """Аналог `rank_scenarios_corp`, але J рахується з `WITH_EXPLOITABILITY_
    CRITERIA`, а НЕ з `model.criteria` (той лишається структурним
    `DEFAULT_CRITERIA` з `dataset.py` незмінним) — тому цю функцію й
    `rank_scenarios_corp` можна викликати на тому самому пулі кандидатів
    незалежно одна від одної й показати обидва ранжування поруч."""
    seen: set[tuple] = set()
    scored = []
    for scenario in scenarios or []:
        key = tuple(scenario.get("path") or ())
        if key in seen:
            continue
        seen.add(key)
        integral, disagreement = aggregate(
            score_scenario_corp_with_exploitability(
                model, scenario.get("path") or [], scenario.get("edge_path") or [], agg=agg,
            ),
            list(WITH_EXPLOITABILITY_CRITERIA),
        )
        scored.append({**scenario, "j": integral, "disagreement": disagreement})
    scored.sort(key=lambda item: (-item["j"], tuple(item.get("path") or ())))
    return scored


# =====================================================================
#  (2) Еталонна атака -> шлях у НАШІЙ моделі. Перетворення ЙДЕ вже
#  наявними ребрами моделі (побудованими виключно з публічних полів
#  `parse_network`); жодне нове ребро тут не додається й не вигадується.
#  Якщо на якомусь кроці потрібного ребра немає — атака НЕ конвертується
#  (`None`), а не обрізається до частини, що збіглась: часткова
#  конвертація приховала б, скільки насправді не зійшлося.
# =====================================================================

def edge_index(model) -> dict[tuple[str, str, str], int]:
    """(джерело, відношення, ціль) -> індекс ребра в `model.edges`.

    Побудова моделі не породжує двох різних ребер з однаковим ключем
    (`build_twin_model` сама дедуплікує через `seen_edges`), тож перший
    знайдений індекс — єдиний коректний."""
    index: dict[tuple[str, str, str], int] = {}
    for i, edge in enumerate(model.edges):
        index.setdefault((edge.source, edge.relation, edge.target), i)
    return index


def yields_target(model) -> dict[str, str]:
    """vuln-вузол -> умова, яку він `yields` (щонайбільше одна на
    вразливість — так побудовано в `dataset.build_twin_model`)."""
    return {edge.source: edge.target for edge in model.edges if edge.relation == "yields"}


def gold_attack_to_path(
    model, edge_index_: dict[tuple[str, str, str], int],
    yields_target_: dict[str, str], attack: dict,
) -> tuple[list[str], list[int]] | None:
    """Еталонна атака -> `(path, edge_path)` у нашій моделі, або `None`,
    якщо хоч один крок не має відповідного переходу.

    Крок називає лише `to` (актив, скомпрометований на цьому кроці) і
    `cve` — цього достатньо, щоб знайти вузол вразливості за конвенцією
    `dataset.build_twin_model` (`VULN:{asset}:{cve}`) БЕЗ жодного
    здогаду: якщо вузла чи потрібного ребра з поточного стану до нього
    немає — це чесна відмова, а не підстановка найближчого схожого."""
    path = [model.current_state_id]
    edge_path: list[int] = []
    state_id = model.current_state_id
    for step in attack.get("steps", []):
        cve = step.get("cve")
        to_asset = step.get("to")
        if not cve or not to_asset:
            return None
        vuln_id = f"VULN:{to_asset}:{cve}"
        exploit_edge = edge_index_.get((state_id, "exploits", vuln_id))
        if exploit_edge is None:
            return None
        next_state = yields_target_.get(vuln_id)
        if next_state is None:
            return None
        yield_edge = edge_index_.get((vuln_id, "yields", next_state))
        if yield_edge is None:
            return None
        path.append(vuln_id)
        edge_path.append(exploit_edge)
        path.append(next_state)
        edge_path.append(yield_edge)
        state_id = next_state
    return path, edge_path


# =====================================================================
#  (4) Базова лінія: неорієнтована непозначена проєкція + BFS найкоротшого
#  шляху. Аналог гілки-контролю A/C з HTB-експерименту: та сама МНОЖИНА
#  транзитних ребер (`twin_bridge.transition_edge_ids`), що й в
#  орієнтованому методі, але без напряму й без типу — крок нічого не
#  називає, він лише "є суміжність". Різниця з орієнтованим методом тоді
#  чесно вимірює внесок САМЕ орієнтованості й типізації, а не якоїсь
#  іншої підміни.
# =====================================================================

def baseline_shortest_path(model) -> list[str] | None:
    """Найкоротший шлях `current_state_id` -> `goal_state_id` неорієнтованим
    BFS по транзитних ребрах моделі, або `None`, якщо недосяжно.

    Суміжність будується з `model.edges` у порядку списку (жодного `set`),
    тож результат детермінований у будь-якому процесі — на відміну від
    деяких внутрішніх структур `dataset.py`/`qa.twin.py` (див. коментар у
    тесті `test_ranking_order_differs_from_dfs_order_where_scores_differ`),
    цей код я пишу сам і сам відповідаю за відтворюваність."""
    adjacency: dict[str, list[str]] = {}
    for i in transition_edge_ids(model):
        edge = model.edges[i]
        adjacency.setdefault(edge.source, []).append(edge.target)
        adjacency.setdefault(edge.target, []).append(edge.source)

    start, goal = model.current_state_id, model.goal_state_id
    if start == goal:
        return [start]
    visited = {start}
    parent: dict[str, str] = {}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = node
            if neighbor == goal:
                queue.clear()
                break
            queue.append(neighbor)
    if goal not in visited:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


# =====================================================================
#  (3) Захисне плече: `twin_defense.rank_defenses` -> компактний зведений
#  рядок на мережу. `twin_defense.rank_defenses` уже повертає повну
#  структуру (ранжовані набори, виключені бюджетом, недосяжні шляхи); ця
#  функція лише витягує ті кілька чисел, які вимагає вимірювання (3), щоб
#  їх можна було покласти в один рядок `per_case.csv`.
# =====================================================================

def defense_summary(model, attack_paths: list[list[int]]) -> dict[str, object]:
    """Зведення `twin_defense.rank_defenses` для ОДНІЄЇ мережі.

    `full_break_possible` — семантика `covers_all` із `twin_defense.cut_sets`:
    існує набір, що розриває все, що взагалі МОЖЛИВО розірвати (не
    плутати з "усі подані шляхи розірвані" — `unbreakable_path_indices`
    може лишатись непорожнім навіть тут, див. докстрінг `cut_sets`).
    `no_defense_possible` — жоден control-вузол не торкається ЖОДНОЇ
    вразливості на жодному з поданих шляхів (виміряний факт: у датасеті
    лише ~58% вразливостей мають хоч якийсь `mitigates` — повний розрив
    тоді неможливий структурно, а не через помилку пошуку).

    ТРЕТІЙ СТАН (`twin_defense.rank_defenses` тепер повертає три кошики, не
    два). `acceptable` — усе, що можна фактично порадити аналітику: набори,
    що проходять усе ОЦІНЮВАНЕ, — і ті, для яких дані сходяться (`ranked`), і
    ті, де вимога лишилась НЕПІДТВЕРДЖЕНОЮ через брак даних
    (`unevaluable_requirements`), а не через реальну розбіжність. Питання
    ПОКРИТТЯ (`full_break_possible`/`no_defense_possible`) незалежне від
    обмежень узагалі (див. докстрінг `rank_defenses`) — тому рахується по
    `all_sets` = усі три кошики РАЗОМ, а не лише `ranked`/`excluded`, як
    рахувалось до появи третього стану."""
    total_paths = len(attack_paths)
    result = twin_defense.rank_defenses(model, attack_paths)
    ranked = result["ranked"]
    excluded = result["excluded_by_constraints"]
    unevaluable = result["unevaluable_requirements"]
    acceptable = ranked + unevaluable
    all_sets = acceptable + excluded
    best = max(acceptable, key=lambda e: e["j"]) if acceptable else None
    best_overall = max(all_sets, key=lambda e: e["j"]) if all_sets else None
    return {
        "total_paths": total_paths,
        "full_break_possible": any(entry["covers_all"] for entry in all_sets),
        "no_defense_possible": not all_sets,
        "best_size": best["size"] if best else None,
        "best_cost": (
            sum(twin_defense._cost_of(model.node_by_id[cid]) for cid in best["controls"])
            if best else None
        ),
        "best_unbroken_paths": (
            len(best["uncovered_path_indices"]) if best else total_paths
        ),
        # Найкращий-УЗАГАЛІ (за J, ігноруючи обмеження) опинився серед
        # виключених бюджетом -> те, що ми фактично можемо порадити
        # аналітику, гірше за те, що технічно існує.
        "best_violates_constraints": bool(
            best_overall is not None and best_overall in excluded
        ),
        # Найкращий-УЗАГАЛІ пройшов усе ОЦІНЮВАНЕ, але модель має вимогу
        # (наприклад regulatory), якої жоден вузол не здатен підтвердити —
        # не порушення, а брак даних; аналітик має вирішити сам.
        "best_has_unevaluable_requirements": bool(
            best_overall is not None and best_overall in unevaluable
        ),
        "truncated": result["truncated"],
        "unbreakable_path_indices": result["unbreakable_path_indices"],
    }


# =====================================================================
#  Наскрізна оцінка ОДНІЄЇ мережі: усі чотири вимірювання в один рядок.
#
#  ПОРЯДОК ВИКЛИКІВ НАВМИСНИЙ і є головною гарантією модуля: `parse_network`
#  (сліпий до `<attacks>`) і `build_twin_model` йдуть ПЕРШИМИ; `parse_gold_
#  attacks` викликається ЛИШЕ ПІСЛЯ того, як пошук (forward/backward/
#  reconcile/enumerate/ранжування) уже відпрацював і дав відповідь.
#  Жодна змінна, обчислена ДО читання еталона, потім не перераховується
#  з його участю — це стереже `test_evaluate_network_does_not_leak_gold_
#  into_structural_fields`.
# =====================================================================

#: Розмір пулу кандидатів перед переранжуванням J — той самий, що й у
#: HTB-гілці E (`experiments/htb_reconstruction/run_experiment.py`).
SCENARIO_POOL = 20
#: Скільки найкращих (за J) шляхів лишаємо для coverage_top_k і захисного
#: плеча — той самий k=5, що й прод (`ui.py`) та HTB-експеримент.
TOP_K = 5


def evaluate_network(xml_text: str, manifest_entry: dict, answers_entry: dict) -> dict:
    """Один рядок `per_case.csv`: усі чотири вимірювання для ОДНІЄЇ мережі."""
    network = parse_network(xml_text)  # ЕТАЛОН СЮДИ НЕ ПОТРАПЛЯЄ (dataset.py)
    model = build_twin_model(network)   # модель побудована НАОСЛІП

    row: dict[str, object] = {
        "network": network.get("id", ""),
        "profile": manifest_entry.get("profile", ""),
        "difficulty": manifest_entry.get("difficulty", ""),
        "nodes": len(model.nodes),
        "edges": len(model.edges),
    }

    # --- (1) орієнтований типізований граф: forward/backward/reconcile/enumerate ---
    forward = expand_forward(model)
    backward = expand_backward(model)
    gc = reconcile(model, forward, backward)
    allowed = set(transition_edge_ids(model))
    gc = {**gc, "edge_ids": [i for i in gc["edge_ids"] if i in allowed]}
    found = enumerate_scenarios(model, gc, matched=set(), max_paths=SCENARIO_POOL)
    ranked = rank_scenarios_corp(model, found["scenarios"])
    top_paths = ranked[:TOP_K]
    row["goal_reachable"] = bool(top_paths)
    row["found_chains"] = len(top_paths)
    row["found_hops"] = max(len(top_paths[0]["path"]) - 1, 0) if top_paths else 0

    # --- те саме переранжування, але з урахуванням exploitability (той
    #     самий пул кандидатів `found["scenarios"]`, публічне поле XML, а
    #     не еталон) — незалежний "другий" набір топ-шляхів для (1). ---
    ranked_expl = rank_scenarios_corp_with_exploitability(model, found["scenarios"])
    top_paths_expl = ranked_expl[:TOP_K]

    # --- (4) базова лінія: та сама множина ребер, неорієнтована проєкція ---
    baseline_path = baseline_shortest_path(model)
    row["baseline_reachable"] = baseline_path is not None
    row["baseline_hops"] = max(len(baseline_path) - 1, 0) if baseline_path else 0

    # =================================================================
    #  Межа: усе, що ВИЩЕ, обчислено БЕЗ жодного звернення до еталона.
    #  Усе, що НИЖЧЕ, читає `<attacks>`/`answers.json` — лише щоб ОЦІНИТИ
    #  вже готову відповідь, а не вплинути на неї.
    # =================================================================
    gold_attacks = parse_gold_attacks(xml_text)
    best_attack_id = answers_entry.get("best_attack", "")
    best_gold = next((a for a in gold_attacks if a["id"] == best_attack_id), None)
    gold_cve_sets = gold_attack_cve_sets(best_gold) if best_gold else []

    if top_paths:
        top_chain = path_cves(model, top_paths[0]["path"])
        union_chain: list[str] = []
        for scenario in top_paths:
            for cve in path_cves(model, scenario["path"]):
                if cve not in union_chain:
                    union_chain.append(cve)
        row.update({
            "path_coverage": step_coverage(top_chain, gold_cve_sets),
            "path_ordered_coverage": ordered_coverage(top_chain, gold_cve_sets),
            "path_precision": precision_against_gold(top_chain, gold_cve_sets),
            "path_coverage_top_k": step_coverage(union_chain, gold_cve_sets),
        })
    else:
        row.update({
            "path_coverage": 0.0, "path_ordered_coverage": 0.0,
            "path_precision": 0.0, "path_coverage_top_k": 0.0,
        })

    # --- (1) той самий показник, але на топ-шляхах з урахуванням
    #     exploitability — окремий набір "with_exploitability", що
    #     дозволяє побачити різницю зі "structural_only" вище. ---
    if top_paths_expl:
        expl_chain = path_cves(model, top_paths_expl[0]["path"])
        expl_union_chain: list[str] = []
        for scenario in top_paths_expl:
            for cve in path_cves(model, scenario["path"]):
                if cve not in expl_union_chain:
                    expl_union_chain.append(cve)
        row.update({
            "expl_path_coverage": step_coverage(expl_chain, gold_cve_sets),
            "expl_path_ordered_coverage": ordered_coverage(expl_chain, gold_cve_sets),
            "expl_path_precision": precision_against_gold(expl_chain, gold_cve_sets),
            "expl_path_coverage_top_k": step_coverage(expl_union_chain, gold_cve_sets),
        })
    else:
        row.update({
            "expl_path_coverage": 0.0, "expl_path_ordered_coverage": 0.0,
            "expl_path_precision": 0.0, "expl_path_coverage_top_k": 0.0,
        })

    if baseline_path:
        baseline_chain = path_cves(model, baseline_path)
        row.update({
            "baseline_coverage": step_coverage(baseline_chain, gold_cve_sets),
            "baseline_ordered_coverage": ordered_coverage(baseline_chain, gold_cve_sets),
            "baseline_precision": precision_against_gold(baseline_chain, gold_cve_sets),
        })
    else:
        row.update({
            "baseline_coverage": 0.0, "baseline_ordered_coverage": 0.0,
            "baseline_precision": 0.0,
        })

    # --- (2) згода ранжування: усі 5 еталонних атак -> шлях -> наш J ---
    ei = edge_index(model)
    yt = yields_target(model)
    effectiveness = answers_entry.get("effectiveness", {})
    attack_j: dict[str, float] = {}
    # "with_exploitability" — той самий конвертований шлях еталонної атаки,
    # але J рахований `WITH_EXPLOITABILITY_CRITERIA`; конвертація
    # (`gold_attack_to_path`) спільна для обох наборів — вона визначається
    # лише реальними ребрами моделі, не оцінкою J, тож не може сама по собі
    # стати джерелом різниці між двома рядками.
    attack_j_expl: dict[str, float] = {}
    for attack in gold_attacks:
        converted = gold_attack_to_path(model, ei, yt, attack)
        if converted is None:
            continue
        path, edge_path = converted
        integral, _ = aggregate(score_scenario_corp(model, path, edge_path), model.criteria)
        attack_j[attack["id"]] = integral
        integral_expl, _ = aggregate(
            score_scenario_corp_with_exploitability(model, path, edge_path),
            list(WITH_EXPLOITABILITY_CRITERIA),
        )
        attack_j_expl[attack["id"]] = integral_expl

    row["gold_attacks_total"] = len(gold_attacks)
    row["gold_attacks_convertible"] = len(attack_j)
    row["gold_attacks_inconvertible"] = len(gold_attacks) - len(attack_j)

    if attack_j:
        our_best = max(sorted(attack_j), key=lambda aid: attack_j[aid])
        row["ranking_best_match"] = our_best == best_attack_id
    else:
        row["ranking_best_match"] = ""

    if attack_j_expl:
        our_best_expl = max(sorted(attack_j_expl), key=lambda aid: attack_j_expl[aid])
        row["expl_ranking_best_match"] = our_best_expl == best_attack_id
    else:
        row["expl_ranking_best_match"] = ""

    common_ids = [aid for aid in attack_j if aid in effectiveness]
    if len(common_ids) >= 2:
        xs = [attack_j[aid] for aid in common_ids]
        ys = [effectiveness[aid] for aid in common_ids]
        row["spearman"] = spearman_rank_correlation(xs, ys)
    else:
        row["spearman"] = ""

    common_ids_expl = [aid for aid in attack_j_expl if aid in effectiveness]
    if len(common_ids_expl) >= 2:
        xs = [attack_j_expl[aid] for aid in common_ids_expl]
        ys = [effectiveness[aid] for aid in common_ids_expl]
        row["expl_spearman"] = spearman_rank_correlation(xs, ys)
    else:
        row["expl_spearman"] = ""

    # --- (3) захисне плече: лише для мереж, де метод узагалі щось знайшов ---
    if top_paths:
        attack_paths = [scenario["edge_path"] for scenario in top_paths]
        defense = defense_summary(model, attack_paths)
        row.update({
            "defense_total_paths": defense["total_paths"],
            "defense_full_break_possible": defense["full_break_possible"],
            "defense_no_defense_possible": defense["no_defense_possible"],
            "defense_best_size": defense["best_size"],
            "defense_best_cost": defense["best_cost"],
            "defense_best_unbroken_paths": defense["best_unbroken_paths"],
            "defense_best_violates_constraints": defense["best_violates_constraints"],
            "defense_best_has_unevaluable_requirements": (
                defense["best_has_unevaluable_requirements"]
            ),
            "defense_truncated": defense["truncated"],
        })
    else:
        row.update({
            "defense_total_paths": 0, "defense_full_break_possible": "",
            "defense_no_defense_possible": "", "defense_best_size": "",
            "defense_best_cost": "", "defense_best_unbroken_paths": "",
            "defense_best_violates_constraints": "",
            "defense_best_has_unevaluable_requirements": "",
            "defense_truncated": "",
        })

    return row


FIELDS = [
    "network", "profile", "difficulty", "nodes", "edges",
    "goal_reachable", "found_chains", "found_hops",
    "path_coverage", "path_ordered_coverage", "path_precision", "path_coverage_top_k",
    "expl_path_coverage", "expl_path_ordered_coverage",
    "expl_path_precision", "expl_path_coverage_top_k",
    "baseline_reachable", "baseline_hops",
    "baseline_coverage", "baseline_ordered_coverage", "baseline_precision",
    "gold_attacks_total", "gold_attacks_convertible", "gold_attacks_inconvertible",
    "ranking_best_match", "spearman", "expl_ranking_best_match", "expl_spearman",
    "defense_total_paths", "defense_full_break_possible", "defense_no_defense_possible",
    "defense_best_size", "defense_best_cost", "defense_best_unbroken_paths",
    "defense_best_violates_constraints", "defense_best_has_unevaluable_requirements",
    "defense_truncated",
]


# =====================================================================
#  Агрегація: 100 рядків `evaluate_network` -> одна структура для
#  `summary.json`. Та сама дисципліна, що й `summarise()` у HTB-
#  експерименті: недосяжні мережі НЕ викидаються із загальних середніх
#  (їхній показник — чесний нуль), а окремо наводиться підрахунок лише по
#  тих мережах, де метод узагалі щось знайшов, — щоб було видно обидві
#  картини одразу.
# =====================================================================

def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "sd": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


#: Спільні префікси полів `FIELDS`, які прибираються з імені секції в
#: summary.json (`path_coverage` -> `coverage`, `expl_path_coverage` ->
#: `coverage` у СВОЇЙ секції `with_exploitability`) — щоб JSON лишався
#: коротким і однаково читався для обох наборів чисел.
_RECONSTRUCTION_LABEL_PREFIXES = ("expl_path_", "path_", "baseline_")


def _reconstruction_stats(rows: list[dict], keys: tuple[str, ...],
                          only: str | None = None) -> dict[str, dict]:
    """`keys` — які поля `FIELDS` усереднити (напр. `path_*`/`baseline_*`/
    `expl_path_*`). `only` — назва булевого поля-фільтра
    (`"goal_reachable"`/`"baseline_reachable"`) для підмножини "лише
    досяжні"; `None` — усі рядки, недосяжні зі своїми чесними нулями."""
    selected = [r for r in rows if not only or r.get(only)]
    result = {}
    for key in keys:
        label = key
        for prefix in _RECONSTRUCTION_LABEL_PREFIXES:
            if key.startswith(prefix):
                label = key[len(prefix):]
                break
        result[label] = _stats([r[key] for r in selected])
    return result


def _ranking_agreement_stats(
    rows: list[dict],
    best_match_field: str = "ranking_best_match",
    spearman_field: str = "spearman",
) -> dict[str, object]:
    """`best_match_field`/`spearman_field` — які поля рядка означають
    "найкращий збігається"/"Спірмен" для ЦЬОГО набору. За замовчуванням —
    структурний (`ranking_best_match`/`spearman`); виклик з
    `expl_ranking_best_match`/`expl_spearman` рахує ТОЙ САМИЙ показник, але
    незалежно, на іншому наборі полів рядка — секції `with_exploitability`
    і структурна фізично не можуть переплутати джерела чисел."""
    best_matches = [r[best_match_field] for r in rows if r.get(best_match_field) != ""]
    spearmans = [r[spearman_field] for r in rows
                if r.get(spearman_field) not in ("", None)]
    return {
        "networks": len(rows),
        "gold_attacks_total": sum(r.get("gold_attacks_total", 0) for r in rows),
        "gold_attacks_convertible": sum(r.get("gold_attacks_convertible", 0) for r in rows),
        "gold_attacks_inconvertible": sum(r.get("gold_attacks_inconvertible", 0) for r in rows),
        "networks_with_best_match_defined": len(best_matches),
        "best_match_share": (
            round(sum(1 for m in best_matches if m) / len(best_matches), 4)
            if best_matches else 0.0
        ),
        "networks_with_spearman_defined": len(spearmans),
        "spearman_mean": round(statistics.fmean(spearmans), 4) if spearmans else 0.0,
    }


def _sliced_ranking_agreement(
    rows: list[dict],
    best_match_field: str = "ranking_best_match",
    spearman_field: str = "spearman",
) -> dict[str, object]:
    """`_ranking_agreement_stats` на всіх рядках + розрізи `by_difficulty`/
    `by_profile` — той самий прийом для структурного й
    `with_exploitability` наборів, лише різні поля-джерела."""
    agreement = _ranking_agreement_stats(rows, best_match_field, spearman_field)
    agreement["by_difficulty"] = {
        value: _ranking_agreement_stats(
            [r for r in rows if r.get("difficulty") == value], best_match_field, spearman_field,
        )
        for value in sorted({r.get("difficulty", "") for r in rows} - {""})
    }
    agreement["by_profile"] = {
        value: _ranking_agreement_stats(
            [r for r in rows if r.get("profile") == value], best_match_field, spearman_field,
        )
        for value in sorted({r.get("profile", "") for r in rows} - {""})
    }
    return agreement


def _defense_stats(rows: list[dict]) -> dict[str, object]:
    scored = [r for r in rows if r.get("defense_full_break_possible") != ""]
    sizes = [r["defense_best_size"] for r in scored if r.get("defense_best_size") not in ("", None)]
    costs = [r["defense_best_cost"] for r in scored if r.get("defense_best_cost") not in ("", None)]
    unbroken = [r["defense_best_unbroken_paths"] for r in scored
                if r.get("defense_best_unbroken_paths") not in ("", None)]
    # `mean_best_unbroken_paths` (нижче) — середня КІЛЬКІСТЬ нерозірваних
    # шляхів НА МЕРЕЖУ, а не частка від усіх знайдених шляхів по датасету —
    # звіт раніше плутав ці два числа (`README.md`, "Виправлення помилки
    # попередньої редакції"). Тут рахуємо ОБИДВА напряму по рядках, щоб
    # звіт більше не діставав частку арифметикою в тексті (25/325 вручну).
    total_paths_all = sum(r.get("defense_total_paths") or 0 for r in scored)
    total_unbroken_all = sum(unbroken)
    return {
        "networks_with_reachable_target": len(scored),
        "networks_with_full_break_possible": sum(1 for r in scored if r["defense_full_break_possible"]),
        "networks_with_no_defense_possible": sum(1 for r in scored if r["defense_no_defense_possible"]),
        # Два НЕЗАЛЕЖНІ лічильники (задача 3 цього завдання): "порушує" (є
        # дані, і вони не сходяться) і "має неоцінювану вимогу" (даних нема
        # взагалі — підтвердити нічим). Раніше друге стану не існувало, і
        # брак даних про regulatory-затвердження мовчки зараховувався в
        # перше — див. README, розділ «Захисне плече».
        "networks_best_violates_constraints": sum(
            1 for r in scored if r["defense_best_violates_constraints"]
        ),
        "networks_best_has_unevaluable_requirements": sum(
            1 for r in scored if r["defense_best_has_unevaluable_requirements"]
        ),
        "mean_best_size": round(statistics.fmean(sizes), 4) if sizes else 0.0,
        "mean_best_cost": round(statistics.fmean(costs), 4) if costs else 0.0,
        "mean_best_unbroken_paths": round(statistics.fmean(unbroken), 4) if unbroken else 0.0,
        "total_paths_all_networks": total_paths_all,
        "total_unbroken_paths_all_networks": total_unbroken_all,
        "unbroken_path_share": (
            round(total_unbroken_all / total_paths_all, 4) if total_paths_all else 0.0
        ),
    }


def _qualitative_examples(rows: list[dict]) -> dict[str, list[str]]:
    """Поіменні приклади мереж для якісних заувваг звіту (README раніше
    називав їх руками — `net-002, net-004, …`). Списки відсортовані за
    `network`, щоб порядок не залежав від порядку рядків `per_case.csv`."""
    return {
        # Ціль досяжна, але топ-1 знайдений шлях не має ЖОДНОГО спільного
        # CVE з еталонною атакою (`path_coverage == 0.0`) — саме так
        # рахується "0 спільних кроків" у `step_coverage`.
        "zero_cve_overlap_networks": sorted(
            r["network"] for r in rows
            if r.get("goal_reachable") and r.get("path_coverage") == 0.0
        ),
        # ЖОДНА з 5 еталонних атак мережі не конвертувалась у шлях моделі.
        "fully_inconvertible_attack_networks": sorted(
            r["network"] for r in rows if r.get("gold_attacks_convertible") == 0
        ),
        # Жоден наявний засіб захисту взагалі не торкається жодної
        # вразливості на знайдених шляхах атаки цієї мережі.
        "no_defense_possible_networks": sorted(
            r["network"] for r in rows if r.get("defense_no_defense_possible") is True
        ),
    }


def summarise(rows: list[dict]) -> dict[str, object]:
    """"structural_only" (усі поля, що були й раніше — незмінна поведінка,
    перевірена регресом на замороженому `results/summary.json`) +
    "with_exploitability" (додаткова секція, рахована з `expl_*` полів
    рядка, НЕЗАЛЕЖНО від структурної — обидва набори чисел мають лишитись
    поруч, щоб різницю можна було побачити)."""
    # `found_hops`/`baseline_hops` додані до тих самих ключів статистики,
    # що й coverage/ordered_coverage/precision — README раніше рахував
    # середню довжину шляху окремо руками з `per_case.csv`; тепер це число
    # (за префіксом `_RECONSTRUCTION_LABEL_PREFIXES` жодного немає, тож
    # ярлик лишається "found_hops" дослівно) живе поруч у `summary.json`.
    reconstruction_keys = ("path_coverage", "path_ordered_coverage",
                           "path_precision", "path_coverage_top_k", "found_hops")
    expl_reconstruction_keys = ("expl_path_coverage", "expl_path_ordered_coverage",
                                "expl_path_precision", "expl_path_coverage_top_k")
    baseline_keys = ("baseline_coverage", "baseline_ordered_coverage",
                     "baseline_precision", "baseline_hops")

    agreement = _sliced_ranking_agreement(rows)
    expl_agreement = _sliced_ranking_agreement(
        rows, "expl_ranking_best_match", "expl_spearman",
    )

    return {
        "networks_total": len(rows),
        "networks_goal_reachable": sum(1 for r in rows if r.get("goal_reachable")),
        "reconstruction": {
            "all": _reconstruction_stats(rows, reconstruction_keys),
            "reachable_only": _reconstruction_stats(rows, reconstruction_keys, only="goal_reachable"),
        },
        "baseline": {
            "networks_reachable": sum(1 for r in rows if r.get("baseline_reachable")),
            "all": _reconstruction_stats(rows, baseline_keys),
            # "де досяжно" — так само, як `reconstruction.reachable_only`,
            # лише фільтр за `baseline_reachable`, а не `goal_reachable`:
            # база досяжна майже завжди там, де орієнтований метод — ні.
            "reachable_only": _reconstruction_stats(rows, baseline_keys, only="baseline_reachable"),
        },
        "ranking_agreement": agreement,
        "defense": _defense_stats(rows),
        "qualitative_examples": _qualitative_examples(rows),
        "with_exploitability": {
            "reconstruction": {
                "all": _reconstruction_stats(rows, expl_reconstruction_keys),
                "reachable_only": _reconstruction_stats(
                    rows, expl_reconstruction_keys, only="goal_reachable",
                ),
            },
            "ranking_agreement": expl_agreement,
        },
    }


# =====================================================================
#  ЗАДАЧА 1: `results/README.md` генерується ЦІЛКОМ із порахованого
#  `summary` — жодне число в тексті не константа. Розбито на дрібні
#  функції по секціях (заголовок / визначення критеріїв / таблиця
#  реконструкції / згода ранжування / захисне плече / база / якісні
#  заувваги / відтворюваність): кожна повертає свій фрагмент markdown,
#  `render_readme` лише склеює їх у фіксованому порядку. Це той самий
#  документ, що раніше підтримувався руками (звідси й зміст пояснень —
#  вони НЕ переписані, лише переведені в шаблон із підстановкою значень).
# =====================================================================

def _fmt_ratio(value: object) -> str:
    """[0,1]-показник -> рядок з 3 знаками після коми (стиль звіту)."""
    return f"{float(value or 0.0):.3f}"


def _fmt2(value: object) -> str:
    """Показник довжини/розміру/вартості -> 2 знаки після коми."""
    return f"{float(value or 0.0):.2f}"


def _fmt_pct(numerator: object, denominator: object) -> str:
    """Частка `numerator/denominator` -> рядок з відсотком (1 знак)."""
    n, d = float(numerator or 0), float(denominator or 0)
    return f"{(n / d * 100 if d else 0.0):.1f}%"


def _fmt_names(names: list[str], limit: int = 5) -> str:
    """Список id мереж -> " (net-a, net-b, …)" або "" для порожнього списку —
    та сама якісна ілюстрація, що раніше вписувалась руками."""
    if not names:
        return ""
    shown = ", ".join(names[:limit])
    return f" ({shown}{', …' if len(names) > limit else ''})"


def _readme_header(config: dict) -> str:
    seed = config.get("pythonhashseed", "?")
    return (
        "# Результати: 100 корпоративних мереж проти прихованого еталона\n\n"
        "Файл згенеровано автоматично (`render_readme` у "
        "`experiments/corporate_networks/run_experiment.py`) одразу після "
        "запису `summary.json`/`per_case.csv` — жодне число тут не вписане "
        "руками, усі беруться з порахованого `summary`. Отримано запуском "
        "`python experiments/corporate_networks/run_experiment.py "
        "--check-determinism` (детермінізм повторного запуску підтверджено "
        f"побайтово, `PYTHONHASHSEED` зафіксовано на `{seed}` — див. "
        "докстрінг `run_experiment.py`, розділ «Відтворюваність і відомий "
        "ризик»). Файли: `summary.json` (усі агреговані числа), "
        "`per_case.csv` (рядок на мережу, 100 рядків).\n\n"
        "Еталон (`<attacks>` у кожному `net-*.xml`, `answers.json`) "
        "читається у `evaluate_network` ЛИШЕ ПІСЛЯ того, як пошук уже дав "
        "відповідь — гарантія перевірена тестом "
        "`tests/test_corporate_experiment.py::"
        "test_evaluate_network_does_not_leak_gold_into_structural_fields` "
        "(підміна CVE всередині `<attacks>` не змінює жодного структурного "
        "поля результату)."
    )


def _readme_determinism_fix_section(summary: dict) -> str:
    """Абзац про щойно виправлену недетермінованість побудови графа
    (причина/фікс/виміряний ДО і ПІСЛЯ) — вимога задачі 1."""
    coverage_mean = summary["reconstruction"]["all"]["coverage"]["mean"]
    return (
        "## Виправлена недетермінованість побудови графа\n\n"
        "До цієї редакції частина ребер моделі (переходи `exploits`, "
        "`dataset._asset_reachability`) збиралась через `set` рядків, а "
        "порядок обходу множини рядків у CPython залежить від "
        "`PYTHONHASHSEED` процесу. Сам НАБІР ребер від цього не змінювався, "
        "але змінювалась їхня ПОСЛІДОВНІСТЬ, а `qa.twin.enumerate_scenarios` "
        "обриває пошук у глибину на межі кандидатів (`max_paths*10`) — тож "
        "із різним порядком у вибірку потрапляли РІЗНІ шляхи ДО "
        "переранжування J. Виміряно напряму на 100 мережах ДО фікса: середнє "
        "покриття (1) гуляло 0,2900 / 0,3047 / 0,3247 / 0,3142 на сідах "
        "0 / 1 / 42 / 777. Виправлено закріпленням КАНОНІЧНОГО (сортованого) "
        "порядку переходів перед додаванням ребер моделі "
        "(`dataset.build_twin_model`, збірка `transitions` перед `link(...)`) "
        "— після фікса середнє покриття однакове на всіх чотирьох сідах і "
        f"дорівнює {_fmt_ratio(coverage_mean)} (перевірено тестом "
        "`tests/test_corporate_dataset.py::"
        "test_transition_edges_are_built_in_a_deterministic_order`)."
    )


def _readme_two_number_sets_section(summary: dict) -> str:
    struct = summary["ranking_agreement"]
    expl = summary["with_exploitability"]["ranking_agreement"]
    expl_weight = next(c.weight for c in WITH_EXPLOITABILITY_CRITERIA if c.name == "exploitability")
    structural_weight = round(1.0 - expl_weight, 4)
    return (
        "## Два набори чисел: `structural_only` і `with_exploitability`\n\n"
        "Перша редакція цього звіту виміряла **нульову згоду** нашого "
        "ранжування J з еталонним `effectiveness`: "
        f"`best_match_share={_fmt_ratio(struct['best_match_share'])}` "
        "(гірше за випадкове вгадування з 5 варіантів, 0.20), "
        f"`spearman_mean={_fmt_ratio(struct['spearman_mean'])}`. Причина "
        "встановлена й не загадкова: J рахував ЛИШЕ структурну "
        "правдоподібність шляху (`feasibility`/`evidence`/`effort` — "
        "`dataset.DEFAULT_CRITERIA`), а еталонна `effectiveness` "
        "(`answers.json[\"scoring\"]`) — це добуток CVSS-`exploitability` "
        "кроків, тобто ЛЕГКІСТЬ ЕКСПЛУАТАЦІЇ. Два різні впорядкування, і "
        "нульова кореляція між ними була очікуваним наслідком.\n\n"
        "Ця редакція додає до J четвертий критерій — `exploitability` "
        "(докладніше нижче) — і рахує ВЕСЬ експеримент ДВІЧІ: "
        "`summary.json`/`per_case.csv` містять і структурний варіант (усі "
        "поля без префікса `expl_`, той самий код, що й раніше — регрес "
        "`tests/test_corporate_experiment.py::"
        "test_structural_only_ranking_agreement_is_unchanged_by_adding_the_exploitability_branch`), "
        "і варіант `with_exploitability` (поля з префіксом `expl_`, окрема "
        "секція `summary[\"with_exploitability\"]`) — щоб різницю можна "
        "було побачити напряму, а не повірити на слово.\n\n"
        "**Чому це лише частково заслуга методу — чесно, наперед.** Ми "
        "додали критерій, побудований на ТОМУ САМОМУ полі "
        "(`exploitability` вузла вразливості), яке еталонна "
        "`effectiveness` перемножує по кроках. Будь-яке використання "
        "цього поля наближує наше ранжування до еталона майже за "
        "побудовою — це не те саме, що метод «навчився» щось "
        "розпізнавати з чистої структури графа. Ми свідомо НЕ відтворили "
        "формулу еталона дослівно (не беремо добуток по кроках — див. "
        f"нижче), і вага exploitability в J — {expl_weight:g} з 1.0, тобто "
        f"структурні критерії й далі важать разом більше ({structural_weight:g}). "
        "Але сам факт, що зростання є, треба читати як «ми перестали "
        "ігнорувати дані, які еталон використовує», а не як незалежне "
        "підтвердження, що метод правильно моделює складність атаки."
    )


def _readme_exploitability_methodology_section() -> str:
    """Як побудований критерій `exploitability` — ваги беруться напряму з
    `WITH_EXPLOITABILITY_CRITERIA`/`DEFAULT_CRITERIA` (код), не переписані
    вручну другий раз."""
    default_by_name = {c.name: c for c in DEFAULT_CRITERIA}
    expl_by_name = {c.name: c for c in WITH_EXPLOITABILITY_CRITERIA}
    weight_rows = "\n".join(
        f"  | {name} | {default_by_name[name].weight if name in default_by_name else '—'} "
        f"| {expl_by_name[name].weight} |"
        for name in ("feasibility", "evidence", "effort", "exploitability")
    )
    return (
        "## Критерій `exploitability`: як побудований\n\n"
        "`path_exploitability(model, path, agg=\"min\")` (`run_experiment.py`) "
        "бере `exploitability` (0..1, публічне поле XML вузлів-вразливостей "
        "— не еталон) з усіх вузлів kind=\"vuln\" на шляху й згортає в одне "
        "число.\n\n"
        "- **Згортка — МІНІМУМ, обраний спосіб.** «Ланцюг не міцніший за "
        "найслабшу ланку»: якщо БУДЬ-ЯКИЙ крок шляху важко експлуатувати, "
        "практична складність усього ланцюга визначається саме ним. "
        "Середнє замаскувало б один важкий крок серед кількох легких. "
        "`agg=\"mean\"` теж реалізований і протестований "
        "(`tests/test_corporate_experiment.py::"
        "test_path_exploitability_mean_is_the_secondary_reported_variant`) "
        "як друга, звітна конфігурація, але НЕ бере участі у виборі ваг "
        "нижче й НЕ використовується для чисел цього звіту.\n"
        "- **НЕ добуток.** Добуток по кроках — дослівно формула еталона "
        "(`dataset.reproduce_effectiveness`). Відтворити її означало б "
        "підмінити вимірювання тавтологією: кореляція наблизилась би до "
        "~1.0 не тому, що метод навчився щось розрізняти, а тому, що ми "
        "порахували ту саму формулу вдруге під іншою назвою.\n"
        "- **Ваги J** (`WITH_EXPLOITABILITY_CRITERIA`) підібрані ОДИН РАЗ і "
        "зафіксовані константою, ДО того, як стали відомі підсумкові "
        "best_match_share/spearman_mean (тест `test_with_exploitability_"
        "criteria_weights_are_fixed_and_sum_to_one` стереже саме ці числа):"
        "\n\n"
        "  | критерій | вага structural_only | вага with_exploitability |\n"
        "  |---|---|---|\n"
        f"{weight_rows}\n\n"
        "  Обґрунтування: структурні критерії лишаються РАЗОМ важливішими "
        "за нову ознаку; усередині exploitability отримує ту саму вагу, що "
        "й найважливіший структурний критерій (feasibility) — істотний "
        "сигнал, але не домінує. evidence/effort знижені порівну, щоб "
        "звільнити вагу, не змінюючи їхнє співвідношення одне до одного.\n"
        "- **Критерій реально змінює порядок**, а не додає мертве число: "
        "на синтетичній фікстурі з двома структурно ІДЕНТИЧНИМИ шляхами "
        "(однакові feasibility/evidence/effort, тай-брейк структурного J — "
        "суто алфавітний) шлях із вищою exploitability виходить першим "
        "лише у варіанті `with_exploitability` "
        "(`test_exploitability_criterion_reorders_structurally_tied_"
        "paths_by_merit`)."
    )


def _readme_reconstruction_table(summary: dict) -> str:
    s_all = summary["reconstruction"]["all"]
    s_reach = summary["reconstruction"]["reachable_only"]
    e_all = summary["with_exploitability"]["reconstruction"]["all"]
    e_reach = summary["with_exploitability"]["reconstruction"]["reachable_only"]

    def row(label: str, key: str) -> str:
        return (
            f"| {label} | {_fmt_ratio(s_all[key]['mean'])} | "
            f"{_fmt_ratio(e_all[key]['mean'])} | {_fmt_ratio(s_reach[key]['mean'])} | "
            f"{_fmt_ratio(e_reach[key]['mean'])} |"
        )

    reach = summary["networks_goal_reachable"]
    total = summary["networks_total"]
    return (
        "### (1) Реконструкція шляху — найкращий (за J) знайдений шлях "
        "проти `best_attack`\n\n"
        "| Показник | structural_only, усі 100 | with_exploitability, усі "
        "100 | structural_only, лише досяжні | with_exploitability, лише "
        "досяжні |\n"
        "|---|---|---|---|---|\n"
        f"{row('coverage', 'coverage')}\n"
        f"{row('ordered_coverage', 'ordered_coverage')}\n"
        f"{row('precision', 'precision')}\n"
        f"{row('coverage_top_k (union до ' + str(TOP_K) + ' шляхів)', 'coverage_top_k')}\n\n"
        "На відміну від (2), тут зміна незначна й неоднозначна за напрямом: "
        "(1) порівнює НАШ знайдений шлях із конкретним `best_attack` за "
        "СПІВПАДІННЯМ CVE, а не зі складністю експлуатації — переранжування "
        "топ-K кандидатів J-ом, що тепер враховує exploitability, змінює, "
        "ЯКИЙ САМЕ шлях потрапляє в top-1, але не обов'язково ЗБЛИЖУЄ його "
        "з еталонним ланцюгом CVE.\n\n"
        f"Ціль досяжна орієнтованим методом (`expand_forward`, повна BFS, "
        f"не DFS-вибірка) у **{reach}/{total}** мереж — не залежить від "
        "переранжування J, те саме число для обох наборів (зафіксовано "
        "тестом `tests/test_corporate_dataset.py::"
        "test_goal_state_reachability_across_the_dataset`)."
    )


def _readme_ranking_agreement_section(summary: dict) -> str:
    struct = summary["ranking_agreement"]
    expl = summary["with_exploitability"]["ranking_agreement"]
    lines = [
        "### (2) Згода ранжування — головний результат",
        "",
        "| Показник | structural_only | with_exploitability |",
        "|---|---|---|",
        (
            f"| best_match_share ({struct['networks_with_best_match_defined']} "
            "мереж, де є з чим порівнювати) | "
            f"**{_fmt_ratio(struct['best_match_share'])}** | "
            f"**{_fmt_ratio(expl['best_match_share'])}** |"
        ),
        (
            f"| spearman_mean ({struct['networks_with_spearman_defined']} "
            f"мереж) | **{_fmt_ratio(struct['spearman_mean'])}** | "
            f"**{_fmt_ratio(expl['spearman_mean'])}** ("
            f"{expl['networks_with_spearman_defined']} мереж) |"
        ),
        (
            f"| конвертовано атак | {struct['gold_attacks_convertible']}/"
            f"{struct['gold_attacks_total']} "
            f"({_fmt_pct(struct['gold_attacks_convertible'], struct['gold_attacks_total'])}) "
            f"| {expl['gold_attacks_convertible']}/{expl['gold_attacks_total']} "
            "(те саме — конвертація не залежить від скорингу) |"
        ),
        "",
        "Розрізи (мереж; best_match_share structural_only / "
        "with_exploitability; spearman_mean structural_only / "
        "with_exploitability):",
        "",
        "| | мереж | best_match_share (S / E) | spearman_mean (S / E) |",
        "|---|---|---|---|",
    ]
    for key, values in struct["by_difficulty"].items():
        expl_values = expl["by_difficulty"][key]
        lines.append(
            f"| {key} | {values['networks']} | "
            f"{_fmt_ratio(values['best_match_share'])} / "
            f"{_fmt_ratio(expl_values['best_match_share'])} | "
            f"{_fmt_ratio(values['spearman_mean'])} / "
            f"{_fmt_ratio(expl_values['spearman_mean'])} |"
        )
    for key, values in struct["by_profile"].items():
        expl_values = expl["by_profile"][key]
        lines.append(
            f"| {key} | {values['networks']} | "
            f"{_fmt_ratio(values['best_match_share'])} / "
            f"{_fmt_ratio(expl_values['best_match_share'])} | "
            f"{_fmt_ratio(values['spearman_mean'])} / "
            f"{_fmt_ratio(expl_values['spearman_mean'])} |"
        )
    lines.extend([
        "",
        (
            f"`best_match_share` йде від {_fmt_ratio(struct['best_match_share'])} "
            f"до {_fmt_ratio(expl['best_match_share'])}, `spearman_mean` — від "
            f"{_fmt_ratio(struct['spearman_mean'])} до "
            f"{_fmt_ratio(expl['spearman_mean'])} після додавання "
            "exploitability. Жоден зріз не досягає ідеальної згоди (1.0) — "
            "очікувано, бо згортка МІНІМУМОМ не відтворює точну формулу "
            "добутку еталона, лише наближається до неї через спільне поле.\n\n"
            "**Ще раз, чесно.** Цей стрибок — здебільшого прямий наслідок "
            "того, що `effectiveness` еталона й наш новий критерій "
            "рахуються з ОДНОГО й того самого поля `exploitability`. Це не "
            "доказ, що метод «навчився» оцінювати складність експлуатації "
            "— це доказ, що ігнорувати наявне поле було методологічною "
            "прогалиною першої редакції. Зростання реальне (виміряне на "
            "всіх 100 мережах, `tests/test_corporate_experiment.py::"
            "test_with_exploitability_ranking_agreement_measured_on_the_full_dataset`), "
            "але його не можна подавати як \"метод покращився сам по собі\"."
        ),
    ])
    return "\n".join(lines)


def _readme_defense_section(summary: dict) -> str:
    d = summary["defense"]
    violates = d["networks_best_violates_constraints"]
    unevaluable = d["networks_best_has_unevaluable_requirements"]
    violation_note = (
        (
            f"**Обмеження моделі РЕАЛЬНО впливають на рекомендацію**: у "
            f"{violates} мереж(і) найкращий за J набір control-вузлів РЕАЛЬНО "
            "не проходить оцінюване обмеження `qa.twin.ALLOWED_CONSTRAINTS` "
            "— дані про нього Є, і вони не сходяться (наприклад перевищений "
            "бюджет чи термін). Технічно найкраща за J рекомендація тоді — "
            "не та, яку можна дати аналітику як є; йому доведеться або "
            "узгодити виняток, або взяти гірший, але допустимий набір."
        )
        if violates
        else (
            "Найкращий набір ЖОДНОГО разу не порушує ОЦІНЮВАНЕ обмеження "
            "моделі — або тому, що обмежень немає (порожній `constraints`), "
            "або тому, що для всіх обмежень, які можна звірити з даними, "
            "найкращий за J набір і так укладається в задані ліміти."
        )
    )
    unevaluable_note = (
        (
            f"**Третій стан: вимогу нічим підтвердити.** У {unevaluable} "
            "мереж(ах) найкращий за J набір проходить УСЕ, що можна "
            "перевірити, але модель містить активну вимогу "
            "(`qa.twin.ALLOWED_CONSTRAINTS`; тут — regulatory-прапорець, "
            "зведений із текстової вимоги `<task><constraints>` "
            "«відповідність ISO/IEC 27035», `experiments/corporate_networks/"
            "dataset.py`), для якої в моделі НЕМАЄ жодного вузла, здатного її "
            "підтвердити: у джерельних XML немає поля, що позначало б засіб "
            "захисту як регуляторно затверджений (`dataset._parse_defense` "
            "знає лише kind/product/state), тож `regulatory_approved` не "
            "виставлено НІ ОДНОМУ control-вузлу цих мереж. "
            "`twin_defense.unevaluable_constraints` визнає вимогу "
            "неоцінюваною, лише коли ЖОДЕН control-вузол УСІЄЇ моделі "
            "(не лише розглянутого набору) не несе потрібного атрибута — "
            "це НЕ доведена невідповідність, а чесно позначений брак даних: "
            "рекомендація лишається найкращою серед перевіреного, але "
            "відповідність цій вимозі підтвердити нічим — рішення про "
            "прийнятність лишається за аналітиком, система його не приймає "
            "автоматично ні в той, ні в інший бік."
        )
        if unevaluable
        else (
            "Жоден набір не має вимог, яких не можна оцінити наявними "
            "даними моделі."
        )
    )
    return (
        "### (3) Захисне плече\n\n"
        "| Показник | значення |\n"
        "|---|---|\n"
        f"| мереж з досяжною ціллю (рахували захист) | {d['networks_with_reachable_target']} |\n"
        f"| повний розрив можливий | {d['networks_with_full_break_possible']} "
        f"({_fmt_pct(d['networks_with_full_break_possible'], d['networks_with_reachable_target'])}) |\n"
        "| захисту НЕМАЄ взагалі (жоден control не торкається жодної "
        f"вразливості на шляхах) | {d['networks_with_no_defense_possible']} |\n"
        f"| середній розмір найкращого набору | {_fmt2(d['mean_best_size'])} control |\n"
        f"| середня вартість найкращого набору | {_fmt2(d['mean_best_cost'])} |\n"
        "| середня КІЛЬКІСТЬ нерозірваних шляхів найкращим набором (з до "
        f"{TOP_K} знайдених на мережу) | {_fmt2(d['mean_best_unbroken_paths'])} |\n"
        "| частка НЕРОЗІРВАНИХ шляхів від усіх знайдених "
        f"({d['total_unbroken_paths_all_networks']} з "
        f"{d['total_paths_all_networks']}) | "
        f"**{float(d['unbroken_path_share'] or 0.0) * 100:.1f}%** |\n"
        f"| найкращий набір ПОРУШУЄ оцінюване обмеження | {violates} |\n"
        f"| найкращий набір має вимогу, якої НІЧИМ підтвердити | {unevaluable} |\n\n"
        f"{violation_note}\n\n"
        f"{unevaluable_note}\n\n"
        "**Про `mean_best_unbroken_paths`.** Це СЕРЕДНЯ КІЛЬКІСТЬ "
        "нерозірваних шляхів на мережу (з до "
        f"{TOP_K} знайдених), а НЕ частка — попередня редакція цього звіту "
        "плутала ці два числа (підписувала середню кількість як відсоток). "
        "Частка нерозірваних шляхів від УСІХ знайдених по датасету — рядок "
        "вище, порахований напряму сумою по мережах, а не переплутаним "
        "середнім.\n\n"
        "\"Повний розрив\" (`covers_all`) стосується лише шляхів, які "
        "взагалі МОЖНА розірвати (`twin_defense.cut_sets`) — навіть у "
        "мережах із \"повним розривом\" можуть лишатись нерозірвані шляхи: "
        "це ті, де ЖОДНА вразливість на маршруті не має власного "
        "`mitigates` (виміряний факт: близько 58% вразливостей датасету "
        "взагалі без mitigates-засобу)."
    )


def _readme_baseline_section(summary: dict) -> str:
    b = summary["baseline"]
    r = summary["reconstruction"]["all"]
    total = summary["networks_total"]
    return (
        "### (4) Базова лінія — неорієнтована непозначена проєкція\n\n"
        "Та сама МНОЖИНА транзитних ребер (`twin_bridge.transition_edge_"
        "ids`), що й в орієнтованому методі, але БЕЗ напряму й типу — крок "
        "нічого не називає, він лише \"є суміжність\". Різниця з "
        "орієнтованим методом тоді чесно вимірює внесок САМЕ орієнтованості "
        "й типізації.\n\n"
        "| Показник | орієнтований метод (усі 100) | база (усі 100) |\n"
        "|---|---|---|\n"
        f"| ціль досяжна | {summary['networks_goal_reachable']}/{total} | "
        f"**{b['networks_reachable']}/{total}** |\n"
        f"| coverage | {_fmt_ratio(r['coverage']['mean'])} | "
        f"{_fmt_ratio(b['all']['coverage']['mean'])} |\n"
        f"| ordered_coverage | {_fmt_ratio(r['ordered_coverage']['mean'])} | "
        f"{_fmt_ratio(b['all']['ordered_coverage']['mean'])} |\n"
        f"| precision | {_fmt_ratio(r['precision']['mean'])} | "
        f"{_fmt_ratio(b['all']['precision']['mean'])} |\n"
        "| середня довжина шляху (де досяжно) | "
        f"{_fmt2(summary['reconstruction']['reachable_only']['found_hops']['mean'])} "
        f"переходів | {_fmt2(b['reachable_only']['hops']['mean'])} переходи |\n\n"
        "База «розв'язує» досяжність тривіально (ігнорування напряму лише "
        "розширює зв'язність — доведено включенням на всіх 100 мережах, "
        "`tests/test_corporate_experiment.py::"
        "test_baseline_reaches_strictly_more_networks_than_the_directed_method`), "
        "але знаходить набагато КОРОТШІ шляхи — переважно прямі \"сусід "
        "через сусіда\" стрибки, не повний ланцюг ескалації. Це і є "
        "вимірюваний внесок орієнтованості й типізації: без них метод "
        "«встигає до цілі» частіше й «дешевше», але відтворює РЕАЛЬНИЙ "
        "ланцюг атаки суттєво гірше."
    )


def _readme_qualitative_section(summary: dict) -> str:
    ex = summary["qualitative_examples"]
    ra = summary["ranking_agreement"]
    total = summary["networks_total"]
    reach = summary["networks_goal_reachable"]
    zero_overlap = ex["zero_cve_overlap_networks"]
    inconvertible = ex["fully_inconvertible_attack_networks"]
    no_defense = ex["no_defense_possible_networks"]
    unreachable = total - reach
    return (
        "## Де метод показав себе погано (не приховуємо)\n\n"
        "- **Згода ранжування покращилась, але лишається далекою від "
        f"ідеальної.** `best_match_share={_fmt_ratio(ra['best_match_share'])}` "
        "— краще за випадкове вгадування (0.20), але все ще означає, що в "
        "суттєвій частці мереж наш \"найкращий\" не збігається з еталонним. "
        f"`spearman_mean={_fmt_ratio(ra['spearman_mean'])}` — помірний, не "
        "сильний зв'язок.\n"
        f"- **{len(zero_overlap)} з {reach} досяжних мереж**"
        f"{_fmt_names(zero_overlap)} дали шлях до цілі, що НЕ МАЄ жодного "
        "спільного CVE з `best_attack`.\n"
        f"- **{ra['gold_attacks_inconvertible']} з {ra['gold_attacks_total']} "
        f"({_fmt_pct(ra['gold_attacks_inconvertible'], ra['gold_attacks_total'])}) "
        "еталонних атак** узагалі не конвертуються в шлях нашої моделі "
        f"(мереж{_fmt_names(inconvertible)} — 0 конвертованих атак з 5 кожна) "
        "— модельне припущення `_exploit_class_condition` не завжди "
        "збігається з тим, що насправді описує конкретний крок еталона.\n"
        + (
            f"- **{len(no_defense)} з {reach} досяжних мереж**"
            f"{_fmt_names(no_defense)} — жоден наявний засіб захисту "
            "взагалі не торкається жодної вразливості на знайдених шляхах "
            "атаки: захисної рекомендації дати неможливо структурно, а не "
            "через слабкість пошуку.\n"
            if no_defense else ""
        )
        + f"- **{unreachable} з {total} мереж** ціль недосяжна орієнтованим "
        "методом узагалі (лише трьома публічними правилами досяжності — "
        "`same_subnet`/`flow`/`policy` — не завжди можна пояснити те, що "
        "еталон описує)."
    )


def _readme_todo_section(summary: dict) -> str:
    ra = summary["ranking_agreement"]
    return (
        "## Що лишилось / ризики\n\n"
        "- Слабка згода ранжування БУЛА головним кандидатом на подальшу "
        "роботу в попередній редакції — цю роботу зроблено (критерій "
        f"`exploitability`), і згода зросла до `spearman_mean="
        f"{_fmt_ratio(summary['with_exploitability']['ranking_agreement']['spearman_mean'])}` "
        "— усе ще не сильний зв'язок, подальше покращення, ймовірно, "
        "впреться в те саме модельне припущення `_exploit_class_condition`, "
        "що вже заважає (1) і конвертованості атак.\n"
        "- Зростання згоди (2) **частково є артефактом вимірювання, а не "
        "чистим покращенням методу** — обидва числа (наш критерій і "
        "еталонна метрика) побудовані на тому самому полі `exploitability` "
        "(розгорнуто вище, розділ «Два набори чисел»).\n"
        "- Модельні припущення `dataset.build_twin_model` "
        "(`EXPLOIT_CLASS_CONDITION`, `_privilege_escalation_condition`, "
        "`_DEFENSE_RELATION`) — джерело як недосяжності частини мереж, так "
        f"і неконвертованості {ra['gold_attacks_inconvertible']} атак — не "
        "помилка ЦЬОГО модуля.\n"
        "- `agg=\"mean\"` (альтернативна згортка exploitability) "
        "реалізована й протестована, але не використана для жодного з "
        "наведених у цьому звіті чисел — обраний спосіб один (мінімум)."
    )


def render_readme(summary: dict, config: dict) -> str:
    """Повний текст `results/README.md`, зібраний з дрібних секцій вище.

    ЄДИНЕ джерело чисел — переданий `summary` (той самий об'єкт, що йде в
    `summary.json`) і `config`; жодна секція не читає файли з диска й не
    містить константи, підібрані заднім числом. Порядок секцій — той
    самий, що в попередній (ручній) редакції звіту, лише переведений у
    підстановку значень."""
    sections = [
        _readme_header(config),
        _readme_determinism_fix_section(summary),
        _readme_two_number_sets_section(summary),
        _readme_exploitability_methodology_section(),
        "## Числа",
        _readme_ranking_agreement_section(summary),
        _readme_reconstruction_table(summary),
        _readme_defense_section(summary),
        _readme_baseline_section(summary),
        _readme_qualitative_section(summary),
        _readme_todo_section(summary),
    ]
    return "\n\n".join(sections) + "\n"


# =====================================================================
#  Введення/виведення: манифест, запуск на всьому датасеті, запис
#  результатів у форматі `experiments/htb_reconstruction/results/`.
# =====================================================================

def load_manifest(path: str) -> dict[str, dict]:
    """`manifest.json` (плоский список) -> мережа -> її запис."""
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["id"]: entry for entry in entries}


def run(dataset_dir: str) -> tuple[list[dict], dict]:
    """Прогін методу на всіх `net-*.xml` каталогу `dataset_dir`."""
    manifest = load_manifest(os.path.join(dataset_dir, "manifest.json"))
    answers = load_answers(os.path.join(dataset_dir, "answers.json"))
    rows = []
    for xml_path in sorted(glob.glob(os.path.join(dataset_dir, "net-*.xml"))):
        network_id = os.path.splitext(os.path.basename(xml_path))[0]
        with open(xml_path, "r", encoding="utf-8") as f:
            xml_text = f.read()
        rows.append(evaluate_network(
            xml_text, manifest.get(network_id, {}), answers.get(network_id, {}),
        ))
    return rows, summarise(rows)


def write_results(rows: list[dict], summary: dict, config: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": config, "summary": summary}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    with (out_dir / "README.md").open("w", encoding="utf-8") as fh:
        fh.write(render_readme(summary, config))


def main() -> int:
    # Фіксуємо PYTHONHASHSEED ДО імпорту решти й перезапускаємось один раз —
    # рядковий хеш впливає на `_asset_reachability` (dataset.py) вже на
    # етапі побудови моделі, тож встановлювати `os.environ` без перезапуску
    # процесу запізно (значення діє лише з наступного старту інтерпретатора).
    # Пояснення — у докстрінгу модуля («Відтворюваність і відомий ризик»).
    if os.environ.get("PYTHONHASHSEED") is None:
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(HERE / "dataset"),
                        help="каталог з net-*.xml, manifest.json, answers.json")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--check-determinism", action="store_true",
                        help="прогнати двічі й звірити результат побайтово")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover — старі інтерпретатори
        pass

    rows, summary = run(args.dataset)

    if args.check_determinism:
        rows2, summary2 = run(args.dataset)
        identical = rows == rows2 and summary == summary2
        print(f"детермінізм повторного запуску: {'ТАК' if identical else 'НІ'}")
        if not identical:
            return 1

    config = {
        "dataset": args.dataset, "scenario_pool": SCENARIO_POOL, "top_k": TOP_K,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "uses_llm": False, "uses_network": False,
    }
    write_results(rows, summary, config, Path(args.out))

    reach = summary["networks_goal_reachable"]
    base_reach = summary["baseline"]["networks_reachable"]
    print(f"\nМереж усього: {summary['networks_total']}; "
          f"ціль досяжна (орієнтований метод): {reach}; "
          f"ціль досяжна (база, неорієнтована): {base_reach}")

    r_all = summary["reconstruction"]["all"]
    r_reach = summary["reconstruction"]["reachable_only"]
    b_all = summary["baseline"]["all"]
    print(f"{'':38}{'coverage':>10}{'ordered':>10}{'precision':>10}")
    print(f"  (1) реконструкція, усі 100     "
          f"{r_all['coverage']['mean']:>10.3f}{r_all['ordered_coverage']['mean']:>10.3f}"
          f"{r_all['precision']['mean']:>10.3f}")
    print(f"      реконструкція, лише досяжні "
          f"{r_reach['coverage']['mean']:>10.3f}{r_reach['ordered_coverage']['mean']:>10.3f}"
          f"{r_reach['precision']['mean']:>10.3f}")
    print(f"  (4) база, усі 100               "
          f"{b_all['coverage']['mean']:>10.3f}{b_all['ordered_coverage']['mean']:>10.3f}"
          f"{b_all['precision']['mean']:>10.3f}")

    a = summary["ranking_agreement"]
    print(f"\n(2) згода ранжування, structural_only: найкращий збігається у "
          f"{a['best_match_share']:.3f} з {a['networks_with_best_match_defined']} мереж "
          f"(усього {a['networks']}); Спірмен середній={a['spearman_mean']:.3f} "
          f"по {a['networks_with_spearman_defined']} мережах; "
          f"конвертовано атак {a['gold_attacks_convertible']}/{a['gold_attacks_total']}")

    ae = summary["with_exploitability"]["ranking_agreement"]
    print(f"    згода ранжування, with_exploitability: найкращий збігається у "
          f"{ae['best_match_share']:.3f} з {ae['networks_with_best_match_defined']} мереж; "
          f"Спірмен середній={ae['spearman_mean']:.3f} по "
          f"{ae['networks_with_spearman_defined']} мережах")

    d = summary["defense"]
    print(f"(3) захист: повний розрив у {d['networks_with_full_break_possible']}/"
          f"{d['networks_with_reachable_target']} мереж; "
          f"жодного захисту в {d['networks_with_no_defense_possible']}; "
          f"середній розмір найкращого набору={d['mean_best_size']:.2f}, "
          f"середня вартість={d['mean_best_cost']:.2f}, "
          f"середньо нерозірваних шляхів={d['mean_best_unbroken_paths']:.2f}")

    print(f"\nЗаписано: {Path(args.out) / 'per_case.csv'}, {Path(args.out) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
