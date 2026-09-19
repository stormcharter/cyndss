"""Експеримент на 100 корпоративних мережах: тести вимірювань (1)-(4).

Головна гарантія, яку тут стережемо, — ПРИХОВАНИЙ `<attacks>` не потрапляє в
жоден крок побудови графа чи пошуку шляхів; еталон (`parse_gold_attacks`,
`load_answers`) читається лише ПІСЛЯ того, як метод уже видав відповідь.

Ідуть і під pytest, і напряму: `python tests/test_corporate_experiment.py`.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa.twin import TwinConstraint, TwinCriterion, TwinEdge, TwinModel, TwinNode
from experiments.corporate_networks.dataset import (
    build_twin_model, load_answers, parse_gold_attacks, parse_network,
)
from experiments.htb_reconstruction.metric import (
    ordered_coverage, precision_against_gold, step_coverage,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_DIR = os.path.join(_REPO_ROOT, "experiments", "corporate_networks", "dataset")


def _read(network_id: str) -> str:
    path = os.path.join(_DATASET_DIR, f"{network_id}.xml")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# =====================================================================
#  (2) Спірмен — коефіцієнт рангової кореляції, перевірений на відомих
#  прикладах ДО того, як ним щось вимірюється на справжніх даних.
# =====================================================================

def test_spearman_matches_hand_computed_example_without_ties():
    """Класичний приклад без нічиїх — два судді ранжують 5 претендентів:
    A=[1,2,3,4,5], B=[2,1,4,3,5]. За скороченою формулою
    rho = 1 - 6*sum(d^2)/(n*(n^2-1)): d=[-1,1,-1,1,0], sum(d^2)=4,
    n=5 -> rho = 1 - 24/120 = 0.8. Числа порахував уручну, а не взяв з коду."""
    from experiments.corporate_networks.run_experiment import spearman_rank_correlation
    got = spearman_rank_correlation([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
    assert abs(got - 0.8) < 1e-9


def test_spearman_handles_ties_via_average_ranks():
    """x=[1,2,2,4], y=[1,3,2,4] — у x нічия на місці 2/3. Середні ранги:
    rx=[1, 2.5, 2.5, 4], ry=[1, 3, 2, 4]. Пірсон на рангах (уручну):
    mean_rx=2.5, mean_ry=2.5; dx=[-1.5,0,0,1.5]; dy=[-1.5,0.5,-0.5,1.5];
    cov=sum(dx*dy)=4.5; var_x=4.5; var_y=5.0; rho=4.5/sqrt(4.5*5.0)=0.94868...
    Це стандартний спосіб рахувати Спірмена з нічиями (Пірсон на середніх
    рангах) — той самий, що дає scipy.stats.spearmanr, якого в середовищі
    немає, тож перевіряємо власну реалізацію вручну порахованим числом."""
    from experiments.corporate_networks.run_experiment import spearman_rank_correlation
    got = spearman_rank_correlation([1, 2, 2, 4], [1, 3, 2, 4])
    assert abs(got - 0.9486832980505138) < 1e-9


def test_spearman_perfect_agreement_is_one():
    from experiments.corporate_networks.run_experiment import spearman_rank_correlation
    assert spearman_rank_correlation([1, 2, 3], [10, 20, 30]) == 1.0


def test_spearman_perfect_disagreement_is_minus_one():
    from experiments.corporate_networks.run_experiment import spearman_rank_correlation
    assert spearman_rank_correlation([1, 2, 3], [30, 20, 10]) == -1.0


def test_spearman_is_undefined_for_constant_values():
    """Усі ранги однакові -> дисперсія рангів нульова -> кореляція не
    визначена. Повертаємо `None`, а не ділимо на нуль і не вдаємо 0.0 —
    0.0 означало б «немає зв'язку», а не «не порахувати»."""
    from experiments.corporate_networks.run_experiment import spearman_rank_correlation
    assert spearman_rank_correlation([1, 1, 1], [1, 2, 3]) is None


def test_spearman_is_undefined_below_two_points():
    from experiments.corporate_networks.run_experiment import spearman_rank_correlation
    assert spearman_rank_correlation([1], [2]) is None
    assert spearman_rank_correlation([], []) is None


# =====================================================================
#  Синтетична модель для маленьких, повністю контрольованих перевірок
#  (той самий прийом, що й `MACHINE` у test_htb_experiment.py).
# =====================================================================

def _tiny_model() -> TwinModel:
    """S0 -[exploits, same_subnet]-> VULN:A1:CVE-1111-0001 -[yields]-> S1 -[exploits, flow]->
    VULN:A2:CVE-2222-0002 -[yields]-> ST (ціль). Два вузли-активи (A1, A2), навмисно різних
    типів ребер `reachability_rule`, щоб перевірити `feasibility`."""
    nodes = [
        TwinNode(id="S0", kind="state", label="S0",
                 attributes={"state_type": "security_state"}),
        TwinNode(id="A1", kind="asset", label="A1"),
        TwinNode(id="VULN:A1:CVE-1111-0001", kind="vuln", label="VULN:A1:CVE-1111-0001",
                 attributes={"cve": "CVE-1111-0001"}),
        TwinNode(id="S1", kind="state", label="S1",
                 attributes={"state_type": "security_state"}),
        TwinNode(id="A2", kind="asset", label="A2"),
        TwinNode(id="VULN:A2:CVE-2222-0002", kind="vuln", label="VULN:A2:CVE-2222-0002",
                 attributes={"cve": "CVE-2222-0002"}),
        TwinNode(id="ST", kind="state", label="ST",
                 attributes={"state_type": "security_state"}),
    ]
    edges = [
        TwinEdge(source="S0", target="VULN:A1:CVE-1111-0001", relation="exploits",
                 attributes={"weight": 1.0, "transition": True,
                             "reachability_rule": "same_subnet"}),
        TwinEdge(source="VULN:A1:CVE-1111-0001", target="S1", relation="yields",
                 attributes={"weight": 1.0, "transition": True}),
        TwinEdge(source="S1", target="VULN:A2:CVE-2222-0002", relation="exploits",
                 attributes={"weight": 1.0, "transition": True,
                             "reachability_rule": "flow"}),
        TwinEdge(source="VULN:A2:CVE-2222-0002", target="ST", relation="yields",
                 attributes={"weight": 1.0, "transition": True}),
    ]
    return TwinModel(
        name="tiny", nodes=nodes, edges=edges,
        criteria=[
            TwinCriterion("feasibility", "", 0.4, "benefit"),
            TwinCriterion("evidence", "", 0.3, "benefit"),
            TwinCriterion("effort", "", 0.3, "cost"),
        ],
        constraints=[], current_state_id="S0", goal_state_id="ST",
    )


# =====================================================================
#  (1)/(2) Зіставлення за CVE: шлях -> ланцюг CVE, атака -> ланцюг CVE
# =====================================================================

def test_path_cves_takes_only_vuln_nodes_in_traversal_order():
    from experiments.corporate_networks.run_experiment import path_cves
    model = _tiny_model()
    path = ["S0", "VULN:A1:CVE-1111-0001", "S1", "VULN:A2:CVE-2222-0002", "ST"]
    assert path_cves(model, path) == ["CVE-1111-0001", "CVE-2222-0002"]


def test_path_cves_of_empty_path_is_empty():
    from experiments.corporate_networks.run_experiment import path_cves
    assert path_cves(_tiny_model(), []) == []


def test_gold_attack_cve_sets_builds_one_singleton_set_per_step():
    from experiments.corporate_networks.run_experiment import gold_attack_cve_sets
    attack = {"id": "a1", "steps": [
        {"n": 1, "from": "x", "to": "y", "cve": "CVE-1111-0001"},
        {"n": 2, "from": "y", "to": "z", "cve": "CVE-2222-0002"},
    ]}
    assert gold_attack_cve_sets(attack) == [{"CVE-1111-0001"}, {"CVE-2222-0002"}]


def test_gold_attack_cve_sets_step_without_cve_is_an_empty_set_not_dropped():
    """Крок без CVE не зникає з еталона (це змінило б довжину еталона й
    завищило б покриття) — стає порожньою множиною, яку жоден вузол не
    покриє."""
    from experiments.corporate_networks.run_experiment import gold_attack_cve_sets
    attack = {"steps": [{"cve": "", "to": "y"}, {"cve": "CVE-9", "to": "z"}]}
    assert gold_attack_cve_sets(attack) == [set(), {"CVE-9"}]


def test_cve_matching_on_a_small_synthetic_reconstruction_case():
    """Маленький повністю контрольований випадок вимірювання (1): наш
    знайдений шлях покриває обидва кроки еталона в правильному порядку і
    без домішок -> coverage=ordered=precision=1.0."""
    from experiments.corporate_networks.run_experiment import (
        gold_attack_cve_sets, path_cves,
    )
    model = _tiny_model()
    found_chain = path_cves(model, ["S0", "VULN:A1:CVE-1111-0001", "S1", "VULN:A2:CVE-2222-0002", "ST"])
    gold = gold_attack_cve_sets({"steps": [
        {"cve": "CVE-1111-0001", "to": "A1"},
        {"cve": "CVE-2222-0002", "to": "A2"},
    ]})
    assert step_coverage(found_chain, gold) == 1.0
    assert ordered_coverage(found_chain, gold) == 1.0
    assert precision_against_gold(found_chain, gold) == 1.0


def test_cve_matching_flags_partial_coverage_honestly():
    """Той самий синтетичний випадок, але еталон називає третій крок,
    якого в знайденому шляху немає — покриття має впасти, а не
    округлитись до 1.0."""
    from experiments.corporate_networks.run_experiment import (
        gold_attack_cve_sets, path_cves,
    )
    model = _tiny_model()
    found_chain = path_cves(model, ["S0", "VULN:A1:CVE-1111-0001", "S1", "VULN:A2:CVE-2222-0002", "ST"])
    gold = gold_attack_cve_sets({"steps": [
        {"cve": "CVE-1111-0001", "to": "A1"},
        {"cve": "CVE-2222-0002", "to": "A2"},
        {"cve": "CVE-3333-0003", "to": "A3"},
    ]})
    assert step_coverage(found_chain, gold) == 2 / 3


# =====================================================================
#  (1) Ранжування сценаріїв інтегральною оцінкою J — НЕ порядок DFS
# =====================================================================

def test_score_scenario_feasibility_penalises_bare_same_subnet_reachability():
    """`exploits` за правилом `same_subnet` — це "просто мережева
    досяжність", а не названий механізм (див. докстрінг `DEFAULT_CRITERIA`
    у dataset.py: "…а не просто мережеву досяжність"). `flow` — навпаки,
    явно заявлений дозвіл, тож зараховується як механізм."""
    from experiments.corporate_networks.run_experiment import score_scenario_corp
    model = _tiny_model()
    path = ["S0", "VULN:A1:CVE-1111-0001", "S1", "VULN:A2:CVE-2222-0002", "ST"]
    edge_path = [0, 1, 2, 3]  # exploits(same_subnet), yields, exploits(flow), yields
    scores = {s["criterion"]: s["score"] for s in score_scenario_corp(model, path, edge_path)}
    # 3 з 4 переходів названі (yields завжди механізм, exploits/flow теж);
    # лише exploits/same_subnet — "гола" досяжність.
    assert scores["feasibility"] == 3 / 4


def test_score_scenario_evidence_counts_only_asset_kind_nodes():
    """"реальні активи" — це вузли kind="asset"; стани й вразливості —
    ні (docstring DEFAULT_CRITERIA: "…а не синтетичними вузлами периметра
    чи стану"). У `_tiny_model` шлях S0->VULN:A1:CVE-1111-0001->S1->VULN:A2:CVE-2222-0002->ST не містить ЖОДНОГО
    вузла kind="asset" (A1/A2 не на шляху) -> evidence=0."""
    from experiments.corporate_networks.run_experiment import score_scenario_corp
    model = _tiny_model()
    path = ["S0", "VULN:A1:CVE-1111-0001", "S1", "VULN:A2:CVE-2222-0002", "ST"]
    scores = {s["criterion"]: s["score"] for s in score_scenario_corp(model, path, [0, 1, 2, 3])}
    assert scores["evidence"] == 0.0


def test_rank_scenarios_corp_orders_by_descending_j_not_input_order():
    from experiments.corporate_networks.run_experiment import rank_scenarios_corp
    model = _tiny_model()
    short = {"path": ["S0", "VULN:A1:CVE-1111-0001", "S1", "VULN:A2:CVE-2222-0002", "ST"], "edge_path": [0, 1, 2, 3]}
    # Довший, штучно повторений сценарій мав би менший J (ефорт вищий) —
    # подаємо його ПЕРШИМ у списку, щоб перевірити саме переранжування,
    # а не проходження вхідного порядку.
    scored = rank_scenarios_corp(model, [short, short])
    assert len(scored) == 1  # дублі шляхів прибираються (як у twin_bridge)
    assert "j" in scored[0]


def _two_path_model() -> TwinModel:
    """Синтетична модель із РІВНО двома простими шляхами S0->ST:

      P1 (коротший, якісніший): S0 -[exploits/flow]-> ASSET1(asset) -[grants]-> ST
         2 переходи, обидва НАЗВАНІ (жоден не "гола" same_subot-досяжність),
         один вузол-актив на шляху -> вищі feasibility/evidence, менший effort.
      P2 (довший, гірший): S0 -[exploits/same_subnet]-> AAA_V1(vuln) -[yields]->
         AAA_S1(state) -[exploits/same_subnet]-> AAA_V2(vuln) -[yields]-> ST
         4 переходи, half "гола" досяжність, жодного активу на шляху.

    Вузол `AAA_V1` лексикографічно МЕНШИЙ за `ASSET1` ('AAA' < 'ASS'), тож
    сирий порядок `enumerate_scenarios` (алфавітний тай-брейк — control-
    вузлів kind="control" на жодному з двох шляхів немає, точно як і в
    корпоративному датасеті, тож перший ключ сортування завжди 0) ставить
    ГІРШИЙ P2 першим. Це не залежить від PYTHONHASHSEED (жодного `set` тут
    немає — вузли й ребра задані як звичайний список), тож тест
    відтворюваний побайтово в будь-якому процесі."""
    nodes = [
        TwinNode(id="S0", kind="state", label="S0",
                 attributes={"state_type": "security_state"}),
        TwinNode(id="ASSET1", kind="asset", label="ASSET1"),
        TwinNode(id="ST", kind="state", label="ST",
                 attributes={"state_type": "security_state"}),
        TwinNode(id="AAA_V1", kind="vuln", label="AAA_V1", attributes={"cve": "CVE-A"}),
        TwinNode(id="AAA_S1", kind="state", label="AAA_S1",
                 attributes={"state_type": "security_state"}),
        TwinNode(id="AAA_V2", kind="vuln", label="AAA_V2", attributes={"cve": "CVE-B"}),
    ]
    edges = [
        TwinEdge(source="S0", target="ASSET1", relation="exploits",
                 attributes={"weight": 1.0, "transition": True, "reachability_rule": "flow"}),
        TwinEdge(source="ASSET1", target="ST", relation="grants",
                 attributes={"weight": 1.0, "transition": True}),
        TwinEdge(source="S0", target="AAA_V1", relation="exploits",
                 attributes={"weight": 1.0, "transition": True, "reachability_rule": "same_subnet"}),
        TwinEdge(source="AAA_V1", target="AAA_S1", relation="yields",
                 attributes={"weight": 1.0, "transition": True}),
        TwinEdge(source="AAA_S1", target="AAA_V2", relation="exploits",
                 attributes={"weight": 1.0, "transition": True, "reachability_rule": "same_subnet"}),
        TwinEdge(source="AAA_V2", target="ST", relation="yields",
                 attributes={"weight": 1.0, "transition": True}),
    ]
    return TwinModel(
        name="two-path", nodes=nodes, edges=edges,
        criteria=[
            TwinCriterion("feasibility", "", 0.4, "benefit"),
            TwinCriterion("evidence", "", 0.3, "benefit"),
            TwinCriterion("effort", "", 0.3, "cost"),
        ],
        constraints=[], current_state_id="S0", goal_state_id="ST",
    )


def test_ranking_order_differs_from_dfs_order_where_scores_differ():
    """Вимога (1): «перший шлях» не має бути алфавітним артефактом
    DFS-обходу `enumerate_scenarios` — та сама помилка, що вже трапилась
    в HTB-експерименті («перший шлях» виявився алфавітним артефактом).
    Перевірено на КОНТРОЛЬОВАНІЙ синтетичній моделі (детерміновано в
    будь-якому процесі), а не на знімку реальної мережі: незалежне
    вимірювання показало, що `enumerate_scenarios` на net-001 впирається
    в межу `max_paths*10=200` кандидатів і те, ЯКІ саме 200 з тисяч
    простих шляхів буде відкинуто, залежить від `PYTHONHASHSEED`
    (`_asset_reachability` у dataset.py будує пари через `set`) — це
    властивість коду, якого не можна чіпати (`qa/twin.py`, `dataset.py`),
    тож знімок реальних даних для ТОЧНОГО порядку був би тестом, що
    зрідка й невмотивовано падає. Синтетична модель цієї проблеми
    позбавлена (жодного `set` у побудові) і доводить ту саму властивість
    надійно."""
    from experiments.corporate_networks.run_experiment import rank_scenarios_corp
    from qa.twin import enumerate_scenarios, expand_backward, expand_forward, reconcile

    model = _two_path_model()
    forward = expand_forward(model)
    backward = expand_backward(model)
    gc = reconcile(model, forward, backward)
    found = enumerate_scenarios(model, gc, matched=set(), max_paths=5)
    assert len(found["scenarios"]) == 2, "фікстура мала дати рівно два прості шляхи"

    dfs_order = [tuple(sc["path"]) for sc in found["scenarios"]]
    assert dfs_order[0][1] == "AAA_V1", "сирий порядок мав поставити довший шлях першим (алфавітний тай-брейк)"

    ranked = rank_scenarios_corp(model, found["scenarios"])
    ranked_order = [tuple(sc["path"]) for sc in ranked]
    assert ranked_order[0][1] == "ASSET1", "J-ранжування мало поставити коротший/якісніший шлях першим"
    assert ranked_order != dfs_order


def _tiny_model_with_exploitability() -> TwinModel:
    """Той самий `_tiny_model`, лише з доданим публічним полем
    `exploitability` на вузлах-вразливостях (0.9 і 0.2) — саме те поле,
    яке в `answers.json["scoring"]` еталон перемножує по кроках
    (`dataset.reproduce_effectiveness`), тут воно ЖИВЕ на вузлі як
    структурний атрибут моделі (`dataset.build_twin_model`), доступний
    методу без жодного погляду на `<attacks>`."""
    model = _tiny_model()
    nodes = list(model.nodes)
    for i, node in enumerate(nodes):
        if node.id == "VULN:A1:CVE-1111-0001":
            nodes[i] = TwinNode(id=node.id, kind=node.kind, label=node.label,
                                attributes={**node.attributes, "exploitability": 0.9})
        elif node.id == "VULN:A2:CVE-2222-0002":
            nodes[i] = TwinNode(id=node.id, kind=node.kind, label=node.label,
                                attributes={**node.attributes, "exploitability": 0.2})
    return TwinModel(
        name=model.name, nodes=nodes, edges=model.edges, criteria=model.criteria,
        constraints=model.constraints, current_state_id=model.current_state_id,
        goal_state_id=model.goal_state_id,
    )


def test_path_exploitability_min_takes_the_weakest_link_by_default():
    """Обраний спосіб згортки — МІНІМУМ (докстрінг `path_exploitability`:
    "ланцюг не міцніший за найслабшу ланку"). Шлях має вузли з
    exploitability 0.9 і 0.2 -> результат 0.2, а не добуток (0.18,
    формула еталона) і не середнє (0.55)."""
    from experiments.corporate_networks.run_experiment import path_exploitability
    model = _tiny_model_with_exploitability()
    path = ["S0", "VULN:A1:CVE-1111-0001", "S1", "VULN:A2:CVE-2222-0002", "ST"]
    assert path_exploitability(model, path) == 0.2
    assert path_exploitability(model, path, agg="min") == 0.2


def test_path_exploitability_mean_is_the_secondary_reported_variant():
    """`agg="mean"` — друга, звітна конфігурація згортки (не бере участі
    у виборі ваг): (0.9 + 0.2) / 2 = 0.55."""
    from experiments.corporate_networks.run_experiment import path_exploitability
    model = _tiny_model_with_exploitability()
    path = ["S0", "VULN:A1:CVE-1111-0001", "S1", "VULN:A2:CVE-2222-0002", "ST"]
    assert abs(path_exploitability(model, path, agg="mean") - 0.55) < 1e-9


def test_path_exploitability_of_a_path_without_vuln_nodes_is_an_honest_zero():
    """Той самий принцип чесного нуля, що й у `evidence`/`feasibility`:
    "немає даних" -> 0.0, а не нейтральне/пропущене значення."""
    from experiments.corporate_networks.run_experiment import path_exploitability
    model = _tiny_model_with_exploitability()
    assert path_exploitability(model, ["S0"]) == 0.0
    assert path_exploitability(model, []) == 0.0


def test_score_scenario_corp_with_exploitability_adds_a_fourth_criterion():
    """Три структурні критерії лишаються ТИМИ САМИМИ (той самий виклик
    `score_scenario_corp` усередині, а не переписана копія), плюс четвертий
    — exploitability за мінімумом шляху."""
    from experiments.corporate_networks.run_experiment import (
        score_scenario_corp_with_exploitability,
    )
    model = _tiny_model_with_exploitability()
    path = ["S0", "VULN:A1:CVE-1111-0001", "S1", "VULN:A2:CVE-2222-0002", "ST"]
    edge_path = [0, 1, 2, 3]
    scores = {s["criterion"]: s["score"] for s in
             score_scenario_corp_with_exploitability(model, path, edge_path)}
    assert set(scores) == {"feasibility", "evidence", "effort", "exploitability"}
    assert scores["feasibility"] == 3 / 4  # незмінно відносно структурного тесту
    assert scores["evidence"] == 0.0       # незмінно відносно структурного тесту
    assert scores["exploitability"] == 0.2


def test_with_exploitability_criteria_weights_are_fixed_and_sum_to_one():
    """Ваги підібрані ОДИН РАЗ і зафіксовані константою (вимога задачі:
    не підганяти під підсумкову кореляцію) — тест стереже саме ці числа,
    а не якісь інші, що можуть з'явитись після майбутнього "покращення"."""
    from experiments.corporate_networks.run_experiment import WITH_EXPLOITABILITY_CRITERIA
    weights = {c.name: c.weight for c in WITH_EXPLOITABILITY_CRITERIA}
    assert weights == {"feasibility": 0.3, "evidence": 0.2, "effort": 0.2, "exploitability": 0.3}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def _tied_structural_paths_model() -> TwinModel:
    """Дві стежки S0->ST, структурно ІДЕНТИЧНІ за feasibility/evidence/
    effort (обидві: один перехід "flow" -> yields, жодного вузла-активу,
    однакова довжина), що різняться ЛИШЕ exploitability вузла-вразливості
    на шляху (A=0.1, B=0.9). Потрібно, щоб довести вимогу задачі: критерій
    exploitability має РЕАЛЬНО міняти порядок там, де структурні оцінки
    однакові, а не бути мертвим числом."""
    nodes = [
        TwinNode(id="S0", kind="state", label="S0", attributes={"state_type": "security_state"}),
        TwinNode(id="ST", kind="state", label="ST", attributes={"state_type": "security_state"}),
        TwinNode(id="VULN:A:CVE-A", kind="vuln", label="A",
                 attributes={"cve": "CVE-A", "exploitability": 0.1}),
        TwinNode(id="VULN:B:CVE-B", kind="vuln", label="B",
                 attributes={"cve": "CVE-B", "exploitability": 0.9}),
    ]
    edges = [
        TwinEdge(source="S0", target="VULN:A:CVE-A", relation="exploits",
                 attributes={"weight": 1.0, "transition": True, "reachability_rule": "flow"}),
        TwinEdge(source="VULN:A:CVE-A", target="ST", relation="yields",
                 attributes={"weight": 1.0, "transition": True}),
        TwinEdge(source="S0", target="VULN:B:CVE-B", relation="exploits",
                 attributes={"weight": 1.0, "transition": True, "reachability_rule": "flow"}),
        TwinEdge(source="VULN:B:CVE-B", target="ST", relation="yields",
                 attributes={"weight": 1.0, "transition": True}),
    ]
    return TwinModel(
        name="tied-structural", nodes=nodes, edges=edges,
        criteria=[
            TwinCriterion("feasibility", "", 0.4, "benefit"),
            TwinCriterion("evidence", "", 0.3, "benefit"),
            TwinCriterion("effort", "", 0.3, "cost"),
        ],
        constraints=[], current_state_id="S0", goal_state_id="ST",
    )


def test_structural_ranking_ties_break_alphabetically_not_by_merit():
    """Контрольна перевірка ПЕРЕД головним тестом: на цій фікстурі
    структурний J справді однаковий для обох стежок, і "перша" стежка —
    лише алфавітний тай-брейк (`VULN:A...` < `VULN:B...`), а не заслуга
    exploitability, якого структурний J узагалі не бачить."""
    from experiments.corporate_networks.run_experiment import rank_scenarios_corp
    model = _tied_structural_paths_model()
    scen_a = {"path": ["S0", "VULN:A:CVE-A", "ST"], "edge_path": [0, 1]}
    scen_b = {"path": ["S0", "VULN:B:CVE-B", "ST"], "edge_path": [2, 3]}
    ranked = rank_scenarios_corp(model, [scen_a, scen_b])
    assert ranked[0]["j"] == ranked[1]["j"], "фікстура мала дати однаковий структурний J"
    assert ranked[0]["path"][1] == "VULN:A:CVE-A", "тай-брейк мав поставити алфавітно меншу вершину першою"


def test_exploitability_criterion_reorders_structurally_tied_paths_by_merit():
    """ГОЛОВНА вимога задачі: критерій exploitability реально змінює
    порядок там, де структурні оцінки однакові. Стежка через B
    (exploitability=0.9, легша) має вийти вище стежки через A
    (exploitability=0.1, важча), хоча структурно вони нерозрізнювані."""
    from experiments.corporate_networks.run_experiment import (
        rank_scenarios_corp_with_exploitability,
    )
    model = _tied_structural_paths_model()
    scen_a = {"path": ["S0", "VULN:A:CVE-A", "ST"], "edge_path": [0, 1]}
    scen_b = {"path": ["S0", "VULN:B:CVE-B", "ST"], "edge_path": [2, 3]}
    ranked = rank_scenarios_corp_with_exploitability(model, [scen_a, scen_b])
    assert ranked[0]["j"] > ranked[1]["j"]
    assert ranked[0]["path"][1] == "VULN:B:CVE-B", "легша для експлуатації стежка мала вийти першою"


def test_ranked_top_scenario_matches_dataset_criteria_on_net_001():
    """Слабший, але відтворюваний наскрізний прогін на СПРАВЖНІХ даних:
    без прив'язки до конкретного порядку (той нестабільний через
    PYTHONHASHSEED — див. тест вище), лише структурні гарантії: список
    непорожній, J відсортовано за спаданням, і кожен шлях справді веде
    з поточного стану в цільовий."""
    from experiments.corporate_networks.run_experiment import rank_scenarios_corp
    from qa.twin import enumerate_scenarios, expand_backward, expand_forward, reconcile
    from twin_bridge import transition_edge_ids

    model = build_twin_model(parse_network(_read("net-001")))
    forward = expand_forward(model)
    backward = expand_backward(model)
    gc = reconcile(model, forward, backward)
    allowed = set(transition_edge_ids(model))
    gc = {**gc, "edge_ids": [i for i in gc["edge_ids"] if i in allowed]}
    found = enumerate_scenarios(model, gc, matched=set(), max_paths=20)
    ranked = rank_scenarios_corp(model, found["scenarios"])

    assert ranked, "очікували хоча б один сценарій на net-001"
    js = [sc["j"] for sc in ranked]
    assert js == sorted(js, reverse=True)
    for sc in ranked:
        assert sc["path"][0] == model.current_state_id
        assert sc["path"][-1] == model.goal_state_id


# =====================================================================
#  (2) Еталонна атака -> шлях у НАШІЙ моделі (лише за реально наявними
#  ребрами; жодного нового ребра тут не вигадується).
# =====================================================================

def test_gold_attack_to_path_walks_real_exploits_and_yields_edges():
    """`_tiny_model`: атака з двома кроками (A1/CVE-1111-0001,
    A2/CVE-2222-0002) точно відповідає наявним ребрам -> повний шлях
    S0->VULN:A1:CVE-1111-0001->S1->VULN:A2:CVE-2222-0002->ST."""
    from experiments.corporate_networks.run_experiment import (
        edge_index, gold_attack_to_path, yields_target,
    )
    model = _tiny_model()
    ei = edge_index(model)
    yt = yields_target(model)
    attack = {"steps": [
        {"from": "S0", "to": "A1", "cve": "CVE-1111-0001"},
        {"from": "A1", "to": "A2", "cve": "CVE-2222-0002"},
    ]}
    result = gold_attack_to_path(model, ei, yt, attack)
    assert result is not None
    path, edge_path = result
    assert path == ["S0", "VULN:A1:CVE-1111-0001", "S1", "VULN:A2:CVE-2222-0002", "ST"]
    assert edge_path == [0, 1, 2, 3]


def test_gold_attack_to_path_returns_none_when_a_step_has_no_matching_edge():
    """Крок називає CVE, якого на цільовому активі в моделі немає (або
    перехід із поточного стану на нього не існує) -> вся атака НЕ
    конвертується, а не обрізається до частини, що збіглась."""
    from experiments.corporate_networks.run_experiment import (
        edge_index, gold_attack_to_path, yields_target,
    )
    model = _tiny_model()
    ei = edge_index(model)
    yt = yields_target(model)
    attack = {"steps": [
        {"from": "S0", "to": "A1", "cve": "CVE-NOT-IN-MODEL"},
    ]}
    assert gold_attack_to_path(model, ei, yt, attack) is None


def test_gold_attack_to_path_returns_none_for_a_step_without_a_cve():
    from experiments.corporate_networks.run_experiment import (
        edge_index, gold_attack_to_path, yields_target,
    )
    model = _tiny_model()
    ei = edge_index(model)
    yt = yields_target(model)
    attack = {"steps": [{"from": "S0", "to": "A1", "cve": ""}]}
    assert gold_attack_to_path(model, ei, yt, attack) is None


def test_gold_attack_to_path_on_a_real_network_matches_a_known_case():
    """net-001, атака a2, крок 1: web-01 -> db-01 через CVE-2020-1472 —
    той самий факт, що вже зафіксований у `tests/test_corporate_dataset.py`
    (`test_exploits_web01_to_dc01_via_the_dmz_to_server_policy_rule` та
    сусідні тести). Перший вузол зконвертованого шляху — вузол
    вразливості на db-01."""
    from experiments.corporate_networks.run_experiment import (
        edge_index, gold_attack_to_path, yields_target,
    )
    model = build_twin_model(parse_network(_read("net-001")))
    gold = parse_gold_attacks(_read("net-001"))
    a2 = next(a for a in gold if a["id"] == "a2")
    ei = edge_index(model)
    yt = yields_target(model)
    result = gold_attack_to_path(model, ei, yt, a2)
    assert result is not None
    path, edge_path = result
    assert path[0] == model.current_state_id
    assert "VULN:db-01:CVE-2020-1472" in path


def test_gold_attack_to_path_convertibility_measured_across_the_dataset():
    """Скільки зі 500 еталонних атак (100 мереж x 5) конвертуються в
    реальний шлях нашої моделі — вимірюємо чесно на ВСЬОМУ датасеті, як і
    `test_goal_state_reachability_across_the_dataset` у
    tests/test_corporate_dataset.py. Число зафіксоване ПІСЛЯ чесного
    вимірювання (453/500), а не підібране заздалегідь."""
    from experiments.corporate_networks.run_experiment import (
        edge_index, gold_attack_to_path, yields_target,
    )
    xml_paths = sorted(glob.glob(os.path.join(_DATASET_DIR, "net-*.xml")))
    assert len(xml_paths) == 100
    total = ok = 0
    for xml_path in xml_paths:
        with open(xml_path, "r", encoding="utf-8") as f:
            xml_text = f.read()
        model = build_twin_model(parse_network(xml_text))
        gold = parse_gold_attacks(xml_text)
        ei = edge_index(model)
        yt = yields_target(model)
        for attack in gold:
            total += 1
            if gold_attack_to_path(model, ei, yt, attack) is not None:
                ok += 1
    assert total == 500, total
    assert ok == 453, ok


# =====================================================================
#  (4) Базова лінія: неорієнтована непозначена проєкція + найкоротший шлях
# =====================================================================

def test_baseline_shortest_path_ignores_direction():
    """Модель, де ЄДИНЕ транзитне ребро веде у "зворотному" напрямі
    (ST -> S0) — орієнтований метод не пройде ним НІКУДИ (з S0 виходу
    немає), а неорієнтована проєкція має знайти шлях, бо напрям
    ігнорується. Це і є вимірюваний внесок орієнтованості: базова лінія
    зобов'язана бути СХІДНОЮ або кращою за зв'язністю, гіршою за точністю."""
    nodes = [
        TwinNode(id="S0", kind="state", label="S0", attributes={"state_type": "security_state"}),
        TwinNode(id="ST", kind="state", label="ST", attributes={"state_type": "security_state"}),
    ]
    edges = [
        TwinEdge(source="ST", target="S0", relation="exploits",
                 attributes={"weight": 1.0, "transition": True, "reachability_rule": "flow"}),
    ]
    model = TwinModel(name="reverse", nodes=nodes, edges=edges, criteria=[],
                      constraints=[], current_state_id="S0", goal_state_id="ST")
    from experiments.corporate_networks.run_experiment import baseline_shortest_path
    assert baseline_shortest_path(model) == ["S0", "ST"]


def test_baseline_shortest_path_returns_none_when_disconnected():
    nodes = [
        TwinNode(id="S0", kind="state", label="S0", attributes={"state_type": "security_state"}),
        TwinNode(id="ST", kind="state", label="ST", attributes={"state_type": "security_state"}),
    ]
    model = TwinModel(name="disconnected", nodes=nodes, edges=[], criteria=[],
                      constraints=[], current_state_id="S0", goal_state_id="ST")
    from experiments.corporate_networks.run_experiment import baseline_shortest_path
    assert baseline_shortest_path(model) is None


def test_baseline_shortest_path_picks_the_shortest_of_several_routes():
    """S0 -> A -> ST (2 переходи) і S0 -> B -> C -> ST (3 переходи) —
    baseline має повернути КОРОТШИЙ (через A), як звичайний BFS."""
    nodes = [
        TwinNode(id="S0", kind="state", label="S0", attributes={"state_type": "security_state"}),
        TwinNode(id="A", kind="asset", label="A"),
        TwinNode(id="B", kind="asset", label="B"),
        TwinNode(id="C", kind="asset", label="C"),
        TwinNode(id="ST", kind="state", label="ST", attributes={"state_type": "security_state"}),
    ]
    def e(src, tgt):
        return TwinEdge(source=src, target=tgt, relation="flow",
                        attributes={"weight": 1.0, "transition": True})
    edges = [e("S0", "B"), e("B", "C"), e("C", "ST"), e("S0", "A"), e("A", "ST")]
    model = TwinModel(name="two-routes", nodes=nodes, edges=edges, criteria=[],
                      constraints=[], current_state_id="S0", goal_state_id="ST")
    from experiments.corporate_networks.run_experiment import baseline_shortest_path
    assert baseline_shortest_path(model) == ["S0", "A", "ST"]


def test_baseline_reaches_strictly_more_networks_than_the_directed_method():
    """Емпіричний факт про весь датасет: неорієнтована проєкція досяжна
    щонайменше там, де досяжний орієнтований метод (ігнорування напряму
    ніколи не звужує зв'язність) — перевіряємо це включення на всіх 100
    мережах, а не стверджуємо число, яке не міряли."""
    from experiments.corporate_networks.run_experiment import baseline_shortest_path
    from qa.twin import expand_forward
    xml_paths = sorted(glob.glob(os.path.join(_DATASET_DIR, "net-*.xml")))
    directed_reachable = baseline_reachable = 0
    for xml_path in xml_paths:
        with open(xml_path, "r", encoding="utf-8") as f:
            model = build_twin_model(parse_network(f.read()))
        forward = expand_forward(model)
        is_directed = model.goal_state_id in forward["node_ids"]
        is_baseline = baseline_shortest_path(model) is not None
        directed_reachable += is_directed
        baseline_reachable += is_baseline
        assert is_baseline or not is_directed, (
            "орієнтований метод знайшов ціль там, де неорієнтована проєкція — ні; "
            "це неможливо, бо остання є надмножиною зв'язності"
        )
    assert directed_reachable == 65, directed_reachable
    assert baseline_reachable >= directed_reachable
    print(f"\nбазова лінія: ціль досяжна у {baseline_reachable}/100 мереж "
          f"(орієнтований метод: {directed_reachable}/100)")


# =====================================================================
#  (3) Захисне плече: `twin_defense.rank_defenses` -> компактний зведений
#  рядок на мережу (той самий прийом, що й `_synthetic_model` у
#  tests/test_twin_defense.py).
# =====================================================================

def _defense_model(nodes, edges, constraints=()):
    return TwinModel(
        name="defense-synthetic", nodes=list(nodes), edges=list(edges),
        criteria=[TwinCriterion("risk", "", 1.0, "benefit")],
        constraints=list(constraints), current_state_id="", goal_state_id="",
    )


def test_defense_summary_reports_a_full_break_when_a_control_covers_everything():
    from experiments.corporate_networks.run_experiment import defense_summary
    model = _defense_model(
        nodes=[
            TwinNode("a1", "asset", "A1"),
            TwinNode("v1", "vuln", "V1"),
            TwinNode("ctrl", "control", "Control", attributes={"cost": 3.0}),
        ],
        edges=[
            TwinEdge("a1", "v1", "has_vuln"),
            TwinEdge("ctrl", "v1", "mitigates"),
        ],
    )
    summary = defense_summary(model, [[0]])
    assert summary["full_break_possible"] is True
    assert summary["no_defense_possible"] is False
    assert summary["best_size"] == 1
    assert summary["best_cost"] == 3.0
    assert summary["best_unbroken_paths"] == 0
    assert summary["best_violates_constraints"] is False


def test_defense_summary_reports_no_defense_possible_when_nothing_mitigates():
    """Жодного `mitigates`-ребра на шлях -> жоден control взагалі не
    претендент — «58% вразливостей без mitigates» з умови задачі саме про
    це: не помилка коду, а чесний вимірюваний факт датасету."""
    from experiments.corporate_networks.run_experiment import defense_summary
    model = _defense_model(
        nodes=[TwinNode("a1", "asset", "A1"), TwinNode("v1", "vuln", "V1")],
        edges=[TwinEdge("a1", "v1", "has_vuln")],
    )
    summary = defense_summary(model, [[0]])
    assert summary["no_defense_possible"] is True
    assert summary["full_break_possible"] is False
    assert summary["best_size"] is None
    assert summary["best_unbroken_paths"] == 1  # єдиний поданий шлях лишається нерозірваним


def test_defense_summary_flags_when_the_best_set_violates_the_budget():
    """Той самий крайовий випадок, що й `_tight_budget_model` у
    test_twin_defense.py: покриття є, бюджету - немає. Єдиний кандидат
    опиняється у `excluded_by_constraints`, тож найкращий (і єдиний)
    набір не проходить обмеження."""
    from experiments.corporate_networks.run_experiment import defense_summary
    model = _defense_model(
        nodes=[
            TwinNode("a1", "asset", "A1"),
            TwinNode("v1", "vuln", "V1"),
            TwinNode("ctrl", "control", "Expensive", attributes={"cost": 100.0}),
        ],
        edges=[
            TwinEdge("a1", "v1", "has_vuln"),
            TwinEdge("ctrl", "v1", "mitigates"),
        ],
        constraints=[TwinConstraint("budget", 10.0)],
    )
    summary = defense_summary(model, [[0]])
    assert summary["best_violates_constraints"] is True
    assert summary["no_defense_possible"] is False


def test_defense_summary_flags_when_the_best_set_has_an_unevaluable_requirement():
    """ЗАДАЧА 3 цього завдання: регуляторна вимога активна (`limit=1`), але
    ЖОДЕН control-вузол моделі не має атрибута `regulatory_approved` —
    підтвердити відповідність нічим. Це НЕ порушення (даних, які не
    сходяться, немає — немає й самих даних) і НЕ "просто проходить" (вимога
    лишається непідтвердженою) — третій, окремий прапорець.

    `full_break_possible` має лишитись True: питання ПОКРИТТЯ шляху атаки і
    питання ОЦІНЮВАНОСТІ вимоги — незалежні (той самий принцип, що вже
    діє для `excluded_by_constraints` — обмеження НЕ звужують пошук
    покриття, `twin_defense.rank_defenses`)."""
    from experiments.corporate_networks.run_experiment import defense_summary
    model = _defense_model(
        nodes=[
            TwinNode("a1", "asset", "A1"),
            TwinNode("v1", "vuln", "V1"),
            TwinNode("ctrl", "control", "Control", attributes={"cost": 3.0}),
        ],
        edges=[
            TwinEdge("a1", "v1", "has_vuln"),
            TwinEdge("ctrl", "v1", "mitigates"),
        ],
        constraints=[TwinConstraint("regulatory", 1.0)],
    )
    summary = defense_summary(model, [[0]])
    assert summary["full_break_possible"] is True
    assert summary["no_defense_possible"] is False
    assert summary["best_size"] == 1
    assert summary["best_unbroken_paths"] == 0
    assert summary["best_violates_constraints"] is False
    assert summary["best_has_unevaluable_requirements"] is True


def test_defense_summary_with_no_attack_paths_is_a_clean_no_op():
    from experiments.corporate_networks.run_experiment import defense_summary
    model = _defense_model(nodes=[], edges=[])
    summary = defense_summary(model, [])
    assert summary["total_paths"] == 0
    assert summary["full_break_possible"] is False
    assert summary["no_defense_possible"] is True
    assert summary["best_unbroken_paths"] == 0


# =====================================================================
#  Наскрізна інтеграція: `evaluate_network` — один рядок на мережу,
#  усі чотири вимірювання разом.
# =====================================================================

def _manifest_and_answers():
    import json
    with open(os.path.join(_DATASET_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = {entry["id"]: entry for entry in json.load(f)}
    answers = load_answers(os.path.join(_DATASET_DIR, "answers.json"))
    return manifest, answers


def test_evaluate_network_on_a_reachable_network_has_sane_values():
    from experiments.corporate_networks.run_experiment import evaluate_network
    manifest, answers = _manifest_and_answers()
    row = evaluate_network(_read("net-001"), manifest["net-001"], answers["net-001"])

    assert row["network"] == "net-001"
    assert row["profile"] == "small"
    assert row["difficulty"] == "easy"
    assert row["goal_reachable"] is True
    assert row["found_chains"] > 0
    assert row["found_hops"] > 0
    for key in ("path_coverage", "path_ordered_coverage", "path_precision", "path_coverage_top_k"):
        assert 0.0 <= row[key] <= 1.0, (key, row[key])
    assert row["gold_attacks_total"] == 5
    assert row["gold_attacks_convertible"] + row["gold_attacks_inconvertible"] == 5
    assert row["ranking_best_match"] in (True, False)


def test_evaluate_network_on_an_unreachable_network_zeroes_out_not_raises():
    """net-003 — одна з 35 мереж, де ціль недосяжна за трьома публічними
    правилами (виміряно в `tests/test_corporate_dataset.py::
    test_goal_state_reachability_across_the_dataset`). Функція має
    повернути нулі, а не впасти й не пропустити мережу мовчки."""
    from experiments.corporate_networks.run_experiment import evaluate_network
    manifest, answers = _manifest_and_answers()
    row = evaluate_network(_read("net-003"), manifest["net-003"], answers["net-003"])

    assert row["goal_reachable"] is False
    assert row["found_chains"] == 0
    assert row["found_hops"] == 0
    for key in ("path_coverage", "path_ordered_coverage", "path_precision", "path_coverage_top_k"):
        assert row[key] == 0.0, (key, row[key])
    # Захисне плече рахується ЛИШЕ для мереж з досяжною ціллю — тут
    # порожньо, а не помилково нульове (це різні речі: "не рахували" й
    # "порахували нуль").
    assert row["defense_full_break_possible"] == ""


def test_evaluate_network_does_not_leak_gold_into_structural_fields():
    """Підміняємо CVE всередині `<attacks>` на явно фальшиві — структурні
    поля (граф, досяжність, довжина знайденого шляху, база) НЕ мають
    змінитись: вони обчислюються з моделі, яку `<attacks>` фізично не
    torkається (`parse_network` вирізає його ДО будь-якого обходу)."""
    import re
    from experiments.corporate_networks.run_experiment import evaluate_network
    manifest, answers = _manifest_and_answers()
    xml_text = _read("net-001")

    def fake_attacks(text):
        def swap(match):
            return re.sub(r'cve="CVE-\d{4}-\d+"', 'cve="CVE-0000-0000"', match.group(0))
        return re.sub(r"<attacks>.*</attacks>", swap, text, flags=re.S)

    tampered = fake_attacks(xml_text)
    assert tampered != xml_text
    assert 'cve="CVE-0000-0000"' in tampered

    row_real = evaluate_network(xml_text, manifest["net-001"], answers["net-001"])
    row_tampered = evaluate_network(tampered, manifest["net-001"], answers["net-001"])

    for key in ("nodes", "edges", "goal_reachable", "found_chains", "found_hops",
                "baseline_reachable", "baseline_hops"):
        assert row_real[key] == row_tampered[key], key


def test_evaluate_network_adds_an_independent_exploitability_variant():
    """Вимога задачі: другий, незалежний набір чисел `with_exploitability`
    з'являється поряд зі структурним, у межах [0,1], і зіставляється з тим
    самим `best_attack`/`effectiveness`, що й структурний."""
    from experiments.corporate_networks.run_experiment import evaluate_network
    manifest, answers = _manifest_and_answers()
    row = evaluate_network(_read("net-001"), manifest["net-001"], answers["net-001"])
    for key in ("expl_path_coverage", "expl_path_ordered_coverage",
                "expl_path_precision", "expl_path_coverage_top_k"):
        assert 0.0 <= row[key] <= 1.0, (key, row[key])
    assert row["expl_ranking_best_match"] in (True, False, "")
    assert row["expl_spearman"] == "" or isinstance(row["expl_spearman"], float)


def test_evaluate_network_exploitability_variant_does_not_leak_gold_either():
    """Та сама гарантія, що й `test_evaluate_network_does_not_leak_gold_
    into_structural_fields`, поширена на нову гілку: `expl_path_coverage`
    тощо ПОРІВНЮЮТЬ з еталоном (тому МАЮТЬ змінитись при підміні CVE), але
    саме РАНЖУВАННЯ (`rank_scenarios_corp_with_exploitability`, звідки
    беруться шляхи) рахується ще до читання `<attacks>` — підміна еталона
    не повинна зламати функцію чи дати неможливе значення."""
    import re
    from experiments.corporate_networks.run_experiment import evaluate_network
    manifest, answers = _manifest_and_answers()
    xml_text = _read("net-001")

    def fake_attacks(text):
        def swap(match):
            return re.sub(r'cve="CVE-\d{4}-\d+"', 'cve="CVE-0000-0000"', match.group(0))
        return re.sub(r"<attacks>.*</attacks>", swap, text, flags=re.S)

    tampered = fake_attacks(xml_text)
    row_tampered = evaluate_network(tampered, manifest["net-001"], answers["net-001"])
    for key in ("expl_path_coverage", "expl_path_ordered_coverage",
                "expl_path_precision", "expl_path_coverage_top_k"):
        assert 0.0 <= row_tampered[key] <= 1.0, (key, row_tampered[key])


def test_fields_lists_every_column_evaluate_network_produces():
    from experiments.corporate_networks.run_experiment import FIELDS, evaluate_network
    manifest, answers = _manifest_and_answers()
    row = evaluate_network(_read("net-001"), manifest["net-001"], answers["net-001"])
    assert set(FIELDS) <= set(row), sorted(set(FIELDS) - set(row))


# =====================================================================
#  Агрегація: `summarise(rows)` — та сама дисципліна "не викидати
#  недосяжне з середніх", що й `summarise()` у HTB-експерименті.
# =====================================================================

def _reconstruction_row(network="n1", profile="small", difficulty="easy",
                        reachable=True, coverage=1.0, baseline_reachable=True,
                        baseline_coverage=1.0, best_match="", spearman="",
                        gold_total=5, gold_convertible=5,
                        defense_reachable=None,
                        expl_coverage=None, expl_best_match="", expl_spearman="",
                        defense_total_paths=1, defense_best_unbroken_paths=0,
                        defense_full_break_possible=True,
                        defense_no_defense_possible=False,
                        defense_best_violates_constraints=False,
                        defense_best_has_unevaluable_requirements=False):
    """Мінімальний рядок з УСІМА полями `FIELDS`, потрібними `summarise`.
    Незгадані поля отримують нейтральні значення — тест підставляє лише
    те, що перевіряє, а решта не заважає (той самий прийом, що й `_row`
    у test_htb_experiment.py).

    `expl_*` — незалежний набір "with_exploitability" (вимога задачі:
    обидва набори чисел рахуються незалежно). За замовчуванням дублює
    структурний `coverage`/`best_match`/`spearman`, щоб тести, яким різниця
    байдужа, не мусили її вказувати явно; тести незалежності передають
    свої `expl_*` значення напряму.

    `defense_total_paths`/`defense_best_unbroken_paths`/`defense_full_break_
    possible`/`defense_no_defense_possible`/`defense_best_violates_
    constraints`/`defense_best_has_unevaluable_requirements` — за
    замовчуванням відтворюють те, що раніше було жорстко зашите в
    `row.update(...)` нижче (1 шлях, 0 нерозірваних, повний розрив
    можливий, жодних порушень чи неоцінюваних вимог), тож наявні виклики без
    цих аргументів поводяться так само, як і до розширення фікстури."""
    if defense_reachable is None:
        defense_reachable = reachable
    if expl_coverage is None:
        expl_coverage = coverage
    row = {
        "network": network, "profile": profile, "difficulty": difficulty,
        "nodes": 1, "edges": 1,
        "goal_reachable": reachable, "found_chains": 1 if reachable else 0,
        "found_hops": 1 if reachable else 0,
        "path_coverage": coverage if reachable else 0.0,
        "path_ordered_coverage": coverage if reachable else 0.0,
        "path_precision": coverage if reachable else 0.0,
        "path_coverage_top_k": coverage if reachable else 0.0,
        "expl_path_coverage": expl_coverage if reachable else 0.0,
        "expl_path_ordered_coverage": expl_coverage if reachable else 0.0,
        "expl_path_precision": expl_coverage if reachable else 0.0,
        "expl_path_coverage_top_k": expl_coverage if reachable else 0.0,
        "baseline_reachable": baseline_reachable,
        "baseline_hops": 1 if baseline_reachable else 0,
        "baseline_coverage": baseline_coverage if baseline_reachable else 0.0,
        "baseline_ordered_coverage": baseline_coverage if baseline_reachable else 0.0,
        "baseline_precision": baseline_coverage if baseline_reachable else 0.0,
        "gold_attacks_total": gold_total,
        "gold_attacks_convertible": gold_convertible,
        "gold_attacks_inconvertible": gold_total - gold_convertible,
        "ranking_best_match": best_match,
        "spearman": spearman,
        "expl_ranking_best_match": expl_best_match,
        "expl_spearman": expl_spearman,
    }
    if defense_reachable:
        row.update({
            "defense_total_paths": defense_total_paths,
            "defense_full_break_possible": defense_full_break_possible,
            "defense_no_defense_possible": defense_no_defense_possible,
            "defense_best_size": 1,
            "defense_best_cost": 0.0,
            "defense_best_unbroken_paths": defense_best_unbroken_paths,
            "defense_best_violates_constraints": defense_best_violates_constraints,
            "defense_best_has_unevaluable_requirements": defense_best_has_unevaluable_requirements,
            "defense_truncated": False,
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


def test_summarise_reconstruction_all_includes_zeros_for_unreachable():
    """Вимога задачі: недосяжні мережі НЕ викидаються із середніх по всіх
    100 — їхній показник дорівнює нулю і тягне середнє вниз чесно."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", reachable=True, coverage=1.0),
        _reconstruction_row("n2", reachable=False),
    ]
    summary = summarise(rows)
    assert summary["reconstruction"]["all"]["coverage"]["mean"] == 0.5
    # А окремо по ДОСЯЖНИХ — те саме, що бачив би аналітик там, де метод
    # взагалі спробував: 1.0, а не занижене мовчазним нулем.
    assert summary["reconstruction"]["reachable_only"]["coverage"]["mean"] == 1.0


def test_summarise_reports_reachable_network_counts():
    from experiments.corporate_networks.run_experiment import summarise
    rows = [_reconstruction_row("n1", reachable=True), _reconstruction_row("n2", reachable=False)]
    summary = summarise(rows)
    assert summary["networks_total"] == 2
    assert summary["networks_goal_reachable"] == 1


def test_summarise_baseline_is_reported_alongside_the_directed_method():
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", reachable=False, baseline_reachable=True, baseline_coverage=0.4),
    ]
    summary = summarise(rows)
    assert summary["baseline"]["networks_reachable"] == 1
    assert summary["baseline"]["all"]["coverage"]["mean"] == 0.4
    # На тому самому рядку орієнтований метод дав 0 — контраст і є суттю
    # вимірювання (4).
    assert summary["reconstruction"]["all"]["coverage"]["mean"] == 0.0


def test_summarise_ranking_agreement_ignores_undefined_entries():
    """`ranking_best_match=""`/`spearman=""` — «не порахувати» (0
    конвертованих атак чи <2 спільних точок), а не «false»/0.0. Такі
    рядки не повинні тягнути частку/середнє вниз."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", best_match=True, spearman=0.5),
        _reconstruction_row("n2", best_match=False, spearman=-0.5),
        _reconstruction_row("n3", best_match="", spearman="", gold_convertible=0),
    ]
    summary = summarise(rows)
    agreement = summary["ranking_agreement"]
    assert agreement["networks_with_best_match_defined"] == 2
    assert agreement["best_match_share"] == 0.5
    assert agreement["networks_with_spearman_defined"] == 2
    assert abs(agreement["spearman_mean"] - 0.0) < 1e-9


def test_summarise_ranking_agreement_counts_gold_attack_convertibility():
    from experiments.corporate_networks.run_experiment import summarise
    rows = [_reconstruction_row("n1", gold_total=5, gold_convertible=3)]
    summary = summarise(rows)
    agreement = summary["ranking_agreement"]
    assert agreement["gold_attacks_total"] == 5
    assert agreement["gold_attacks_convertible"] == 3
    assert agreement["gold_attacks_inconvertible"] == 2


def test_summarise_ranking_agreement_has_difficulty_and_profile_slices():
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", difficulty="easy", profile="small", best_match=True, spearman=1.0),
        _reconstruction_row("n2", difficulty="hard", profile="large", best_match=False, spearman=-1.0),
    ]
    summary = summarise(rows)
    by_difficulty = summary["ranking_agreement"]["by_difficulty"]
    by_profile = summary["ranking_agreement"]["by_profile"]
    assert by_difficulty["easy"]["best_match_share"] == 1.0
    assert by_difficulty["hard"]["best_match_share"] == 0.0
    assert by_profile["small"]["best_match_share"] == 1.0
    assert by_profile["large"]["best_match_share"] == 0.0


def test_summarise_defense_aggregates_only_networks_with_a_reachable_target():
    """`defense_*` порожні («не рахували») для недосяжних мереж — вони не
    повинні брати участь у середньому розмірі/вартості набору."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", reachable=True, defense_reachable=True),
        _reconstruction_row("n2", reachable=False, defense_reachable=False),
    ]
    summary = summarise(rows)
    defense = summary["defense"]
    assert defense["networks_with_reachable_target"] == 1
    assert defense["networks_with_full_break_possible"] == 1
    assert defense["mean_best_size"] == 1.0


def test_summarise_defense_separates_violations_from_unevaluable_requirements():
    """`networks_best_violates_constraints` (є дані, і вони не сходяться) і
    `networks_best_has_unevaluable_requirements` (даних нема — підтвердити
    нічим) — ДВА незалежні числа, а не одне на двох: мережа з порушенням не
    повинна тягнути другий лічильник, і навпаки."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", reachable=True, defense_reachable=True,
                            defense_best_violates_constraints=True,
                            defense_best_has_unevaluable_requirements=False),
        _reconstruction_row("n2", reachable=True, defense_reachable=True,
                            defense_best_violates_constraints=False,
                            defense_best_has_unevaluable_requirements=True),
        _reconstruction_row("n3", reachable=True, defense_reachable=True,
                            defense_best_violates_constraints=False,
                            defense_best_has_unevaluable_requirements=False),
    ]
    summary = summarise(rows)
    defense = summary["defense"]
    assert defense["networks_best_violates_constraints"] == 1
    assert defense["networks_best_has_unevaluable_requirements"] == 1


# =====================================================================
#  Розширення `summarise()` під ЗАДАЧУ 1: `results/README.md` генерується
#  з `summary`, а не підтримується руками, тож усі числа, які раніше жили
#  ЛИШЕ в тексті README (середня довжина шляху, частка нерозірваних
#  шляхів "від усіх знайдених", поіменні приклади мереж), тепер мають
#  потрапляти в `summary.json` — інакше шаблону нема звідки їх узяти.
# =====================================================================

def test_summarise_reconstruction_reports_mean_found_hops():
    """README називає середню довжину знайденого шляху (`found_hops`) —
    раніше це число рахувалось руками з `per_case.csv`; тепер має бути
    поруч із coverage/ordered_coverage/precision у `summary.json`."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", reachable=True),   # found_hops=1 (фікстура)
        _reconstruction_row("n2", reachable=False),  # found_hops=0
    ]
    summary = summarise(rows)
    assert summary["reconstruction"]["all"]["found_hops"]["mean"] == 0.5
    assert summary["reconstruction"]["reachable_only"]["found_hops"]["mean"] == 1.0


def test_summarise_baseline_reports_reachable_only_hops_stats():
    """Той самий показник для бази, і окремо — лише серед мереж, де база
    досяжна (README порівнює «середню довжину шляху (де досяжно)» для
    обох методів, а не змішує з нульовими недосяжними)."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", baseline_reachable=True),   # baseline_hops=1
        _reconstruction_row("n2", baseline_reachable=False),  # baseline_hops=0
    ]
    summary = summarise(rows)
    assert summary["baseline"]["all"]["hops"]["mean"] == 0.5
    assert summary["baseline"]["reachable_only"]["hops"]["mean"] == 1.0


def test_summarise_defense_reports_total_paths_and_unbroken_share_across_networks():
    """`mean_best_unbroken_paths` — середня КІЛЬКІСТЬ нерозірваних шляхів
    НА МЕРЕЖУ (не частка!, README раніше плутав ці два числа). Частку
    "нерозірваних від усіх знайдених" README рахував руками (25/325);
    тепер це готове поле `summary`, а не арифметика в тексті."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", reachable=True, defense_reachable=True,
                            defense_total_paths=5, defense_best_unbroken_paths=2),
        _reconstruction_row("n2", reachable=True, defense_reachable=True,
                            defense_total_paths=3, defense_best_unbroken_paths=0),
    ]
    summary = summarise(rows)
    defense = summary["defense"]
    assert defense["total_paths_all_networks"] == 8
    assert defense["total_unbroken_paths_all_networks"] == 2
    assert defense["unbroken_path_share"] == 0.25


def test_summarise_defense_unbroken_share_is_zero_when_nothing_was_scored():
    """Чесний нуль (не ділення на нуль), коли жодна мережа не досяжна —
    той самий принцип, що вже діє для `spearman_mean`/`best_match_share`."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [_reconstruction_row("n1", reachable=False, defense_reachable=False)]
    summary = summarise(rows)
    assert summary["defense"]["unbroken_path_share"] == 0.0


def test_summarise_lists_reachable_networks_with_zero_cve_overlap():
    """Мережі, де ціль досяжна, але топ-1 шлях не має ЖОДНОГО спільного
    CVE з еталонною атакою (`path_coverage == 0.0`) — README раніше
    називав такі мережі поіменно вручну (`net-002, net-004, …`); тепер
    список рахується з `rows`, а не переписується руками."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", reachable=True, coverage=0.0),
        _reconstruction_row("n2", reachable=True, coverage=0.5),
        # недосяжна мережа з "нульовим" покриттям не рахується — там
        # взагалі немає знайденого шляху, який міг би не мати спільних CVE.
        _reconstruction_row("n3", reachable=False, coverage=0.0),
    ]
    summary = summarise(rows)
    assert summary["qualitative_examples"]["zero_cve_overlap_networks"] == ["n1"]


def test_summarise_lists_networks_with_no_convertible_gold_attacks():
    """Мережі, де ЖОДНА з еталонних атак не конвертувалась у шлях нашої
    моделі (`gold_attacks_convertible == 0`) — теж був ручний перелік."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", gold_total=5, gold_convertible=0),
        _reconstruction_row("n2", gold_total=5, gold_convertible=3),
    ]
    summary = summarise(rows)
    assert summary["qualitative_examples"]["fully_inconvertible_attack_networks"] == ["n1"]


def test_summarise_lists_networks_with_no_defense_possible():
    """Мережі, де жоден наявний засіб захисту взагалі не торкається
    жодної вразливості на знайдених шляхах (`defense_no_defense_possible`)."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", reachable=True, defense_reachable=True,
                            defense_no_defense_possible=True,
                            defense_full_break_possible=False),
        _reconstruction_row("n2", reachable=True, defense_reachable=True),
    ]
    summary = summarise(rows)
    assert summary["qualitative_examples"]["no_defense_possible_networks"] == ["n1"]


# =====================================================================
#  `summarise()["with_exploitability"]` — другий, НЕЗАЛЕЖНИЙ набір чисел
#  (вимога задачі: "structural_only" і "with_exploitability" рахуються
#  незалежно й обидва лишаються видимими поруч у `summary.json`).
# =====================================================================

def test_summarise_reports_a_with_exploitability_section_alongside_structural():
    """Секція `with_exploitability` з'являється поряд зі структурною, не
    замінюючи її."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [_reconstruction_row("n1")]
    summary = summarise(rows)
    assert "reconstruction" in summary  # structural_only лишається на місці
    assert "with_exploitability" in summary
    assert "reconstruction" in summary["with_exploitability"]
    assert "ranking_agreement" in summary["with_exploitability"]


def test_summarise_with_exploitability_reconstruction_reads_its_own_fields():
    """`with_exploitability.reconstruction` рахується з `expl_path_*`, а
    НЕ з `path_*` — підміна лише `expl_coverage` (структурний `coverage`
    лишається як був) має змінити ЛИШЕ expl-секцію."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [_reconstruction_row("n1", coverage=0.2, expl_coverage=0.9)]
    summary = summarise(rows)
    assert summary["reconstruction"]["all"]["coverage"]["mean"] == 0.2
    assert summary["with_exploitability"]["reconstruction"]["all"]["coverage"]["mean"] == 0.9


def test_summarise_with_exploitability_ranking_agreement_reads_its_own_fields():
    """`with_exploitability.ranking_agreement` рахується з `expl_ranking_
    best_match`/`expl_spearman`, а НЕ зі структурних `ranking_best_match`/
    `spearman` — два рядки, де структурний і expl-варіант розходяться в
    ПРОТИЛЕЖНІ боки, мають дати РІЗНІ `best_match_share` для двох секцій."""
    from experiments.corporate_networks.run_experiment import summarise
    rows = [
        _reconstruction_row("n1", best_match=False, expl_best_match=True),
        _reconstruction_row("n2", best_match=True, expl_best_match=False),
    ]
    summary = summarise(rows)
    assert summary["ranking_agreement"]["best_match_share"] == 0.5
    assert summary["with_exploitability"]["ranking_agreement"]["best_match_share"] == 0.5
    # Тепер розхитуємо симетрію: обидва рядки з expl_best_match=True.
    # Структурна частка МАЄ лишитись 0.5 (вона не читає expl-поля), а
    # expl-частка мусить піднятись до 1.0 — інакше секції рахуються не
    # незалежно, а з одного й того самого джерела.
    rows[1]["expl_ranking_best_match"] = True
    summary2 = summarise(rows)
    assert summary2["ranking_agreement"]["best_match_share"] == 0.5
    assert summary2["with_exploitability"]["ranking_agreement"]["best_match_share"] == 1.0


def test_fields_and_summarise_are_importable_together():
    from experiments.corporate_networks.run_experiment import FIELDS, summarise
    row = _reconstruction_row()
    assert set(FIELDS) <= set(row)
    summarise([row])  # не падає на мінімальному коректному рядку


# =====================================================================
#  Наскрізний запуск на ВСЬОМУ датасеті + запис результатів
# =====================================================================

def test_load_manifest_returns_id_keyed_dict():
    from experiments.corporate_networks.run_experiment import load_manifest
    manifest = load_manifest(os.path.join(_DATASET_DIR, "manifest.json"))
    assert len(manifest) == 100
    assert manifest["net-001"]["profile"] == "small"
    assert manifest["net-001"]["difficulty"] == "easy"


def test_run_produces_one_row_per_network_with_known_totals():
    """Наскрізний прогін на всіх 100 мережах. Числа зафіксовані ПІСЛЯ
    чесного вимірювання (той самий прийом, що й `reachable == 65` у
    tests/test_corporate_dataset.py) — якщо метод чи датасет зміняться,
    цей тест мусить впасти й привернути увагу, а не тихо змінити сенс."""
    from experiments.corporate_networks.run_experiment import run
    rows, summary = run(_DATASET_DIR)
    assert len(rows) == 100
    assert summary["networks_total"] == 100
    assert summary["networks_goal_reachable"] == 65
    assert summary["ranking_agreement"]["gold_attacks_total"] == 500
    assert summary["ranking_agreement"]["gold_attacks_convertible"] == 453


def test_structural_only_ranking_agreement_is_unchanged_by_adding_the_exploitability_branch():
    """Регрес (вимога задачі): додавання гілки `with_exploitability` НЕ
    повинно зрушити жодне вже задокументоване структурне число. Значення —
    дослівна копія `experiments/corporate_networks/results/summary.json`,
    записаного ДО цієї роботи (комміт `491a178`), а не підібране заднім
    числом.

    Пінуємо саме `ranking_agreement` (а не `reconstruction`/`baseline.
    coverage`): цей показник рахується напряму з конвертованих еталонних
    шляхів (`gold_attack_to_path`), без участі DFS-вибірки
    `enumerate_scenarios` — а отже, доведено НЕ залежить від
    `PYTHONHASHSEED` поза `main()` (перевірено прямим повторним запуском
    із трьома різними хеш-сідами: число не рухається). `networks_goal_
    reachable`/`baseline.networks_reachable` — БУЛЕВА досяжність через
    повний BFS, теж перевірено стабільна (порядок ребер впливає лише на
    ТЕ, ЯКИЙ найкоротший шлях знайде BFS серед кількох рівних, а не на
    факт досяжності) — на відміну від `baseline.all.coverage.mean`, яка
    саме тому свідомо НЕ пінується тут: вона залежить від ДОВЖИНИ й
    СКЛАДУ знайденого найкоротшого шляху серед кількох рівних (перевірено
    прямим вимірюванням: 0.1395/0.1478/0.1568/0.1618 на різних сідах)."""
    from experiments.corporate_networks.run_experiment import run
    rows, summary = run(_DATASET_DIR)
    assert summary["networks_goal_reachable"] == 65
    assert summary["baseline"]["networks_reachable"] == 100
    assert summary["ranking_agreement"]["gold_attacks_total"] == 500
    assert summary["ranking_agreement"]["gold_attacks_convertible"] == 453
    assert summary["ranking_agreement"]["best_match_share"] == 0.1789
    assert summary["ranking_agreement"]["spearman_mean"] == 0.0452


def test_with_exploitability_ranking_agreement_measured_on_the_full_dataset():
    """Головний результат цієї роботи, чесно виміряний на всьому датасеті
    (не підібраний заднім числом, як і `453` вище): додавання
    exploitability піднімає згоду ранжування з еталоном. Числа зафіксовані
    ПІСЛЯ вимірювання й НЕ залежать від `PYTHONHASHSEED` (те саме
    обґрунтування, що й у сусідньому тесті на структурний варіант —
    `ranking_agreement` не використовує DFS-вибірку)."""
    from experiments.corporate_networks.run_experiment import run
    rows, summary = run(_DATASET_DIR)
    expl = summary["with_exploitability"]["ranking_agreement"]
    assert expl["networks_with_best_match_defined"] == 95
    assert expl["networks_with_spearman_defined"] == 93
    assert expl["best_match_share"] == 0.3895
    assert expl["spearman_mean"] == 0.3253
    # Порівняння МАЄ бути прямим (>), а не просто "не дорівнює" — інакше
    # погіршення теж вважалося б "успіхом".
    assert expl["best_match_share"] > summary["ranking_agreement"]["best_match_share"]
    assert expl["spearman_mean"] > summary["ranking_agreement"]["spearman_mean"]


# =====================================================================
#  ЗАДАЧА 2 (перша редакція): обмеження задачі (`<task><constraints>` ->
#  `classify_task_constraints` -> `TwinConstraint`) БУЛИ порожні для всіх
#  100 мереж, тож перевірка обмежень у захисному плечі
#  (`twin_defense.rank_defenses`) працювала вхолосту. Після появи
#  regulatory-прапорця для мереж із «відповідність ISO/IEC 27035» цей тест
#  зафіксував число 21 як `networks_best_violates_constraints` — і саме тут
#  ховалась ЗАДАЧА 3 (це завдання): 21 не було доведеним порушенням.
#  У джерельних XML НЕМАЄ поля, що позначало б control-вузол як
#  `regulatory_approved` (`dataset._parse_defense` знає лише
#  kind/product/state), тож стара двійкова перевірка трактувала ВІДСУТНІСТЬ
#  даних як доведену НЕВІДПОВІДНІСТЬ — будь-який непорожній набір заходів
#  автоматично "порушував" вимогу, якої нічим підтвердити. Це саме зіпсувало
#  й сусідні числа: `mean_best_unbroken_paths` (0.37 -> 1.97) і частку
#  нерозірваних шляхів (7.7% -> 39.4%), бо `defense_summary` не бачив
#  ЖОДНОГО прийнятного набору для цих 21 мережі (усі відкидались як
#  "порушники") і рахував ВЕСЬ пул шляхів нерозірваним.
#
#  `twin_defense.unevaluable_constraints`/`rank_defenses` (третій кошик,
#  `unevaluable_requirements`) виправляють це: тепер тест ЧЕСНО фіксує ОБИДВА
#  числа нарізно — `networks_best_violates_constraints` (є дані, і вони не
#  сходяться) і `networks_best_has_unevaluable_requirements` (даних нема
#  взагалі). Це СВІДОМА правка попереднього assert-у 21 на нове число: стара
#  перевірка була написана до виправлення саме тієї помилки, яку описує ця
#  задача, і буквально стверджувала бракований результат як факт.
# =====================================================================

def test_defense_constraints_distinguish_violations_from_unevaluable_requirements():
    """Головний вимірюваний ефект задачі 3: 21 з 65 досяжних мереж (ті, де є
    регуляторна вимога ISO/IEC 27035 і технічно можливий захист) МАЮТЬ
    найкращий-за-J набір із вимогою, якої нічим підтвердити — але це більше
    НЕ порушення (`networks_best_violates_constraints` тепер 0, бо жодна
    мережа датасету не дає жодному control-вузлу `regulatory_approved`, а
    отже жодна перевірка регуляторності НЕ є оцінюваною на цих даних)."""
    from experiments.corporate_networks.run_experiment import run
    rows, summary = run(_DATASET_DIR)
    assert summary["defense"]["networks_best_violates_constraints"] == 0
    assert summary["defense"]["networks_best_has_unevaluable_requirements"] == 21
    # Відновлені (не завищені відсутністю рекомендації) числа захисного
    # плеча - той самий факт, що описаний вище: коли 21 мережа мала best=None
    # через хибне "порушення", ВЕСЬ їхній пул шляхів рахувався нерозірваним.
    assert summary["defense"]["mean_best_unbroken_paths"] == 0.3692
    assert summary["defense"]["unbroken_path_share"] == 0.0738
    # І це не побічний ефект випадкової зміни складу шляхів: недосяжність/
    # реконструкція/база не зрушились (той самий регрес, що вже стереже
    # `test_structural_only_ranking_agreement_is_unchanged_by_adding_the_
    # exploitability_branch`) — зміна ізольована рівно в захисному плечі.
    assert summary["networks_goal_reachable"] == 65
    assert summary["baseline"]["networks_reachable"] == 100


def test_write_results_creates_csv_and_summary_json(tmp_path):
    from experiments.corporate_networks.run_experiment import (
        FIELDS, summarise, write_results,
    )
    rows = [_reconstruction_row("n1"), _reconstruction_row("n2", reachable=False)]
    summary = summarise(rows)
    write_results(rows, summary, {"note": "test"}, tmp_path)

    csv_path = tmp_path / "per_case.csv"
    summary_path = tmp_path / "summary.json"
    assert csv_path.exists()
    assert summary_path.exists()

    import csv as csv_module
    with csv_path.open(encoding="utf-8") as fh:
        written = list(csv_module.DictReader(fh))
    assert len(written) == 2
    assert set(FIELDS) <= set(written[0])

    import json as json_module
    with summary_path.open(encoding="utf-8") as fh:
        payload = json_module.load(fh)
    assert payload["config"] == {"note": "test"}
    assert payload["summary"]["networks_total"] == 2


# =====================================================================
#  ЗАДАЧА 1: `results/README.md` генерується з `summary`, а не
#  підтримується руками. Перевіряємо ПІДСТАНОВКОЮ підроблених значень у
#  `summary` (а не порівнянням з числами, вписаними в шаблон константами):
#  якщо підроблене число дослівно з'являється в тексті README, підстановка
#  справді читає `summary`, а не повторює застиглий текст попередньої
#  редакції.
# =====================================================================

def test_write_results_creates_a_readme_file(tmp_path):
    from experiments.corporate_networks.run_experiment import write_results
    rows = [_reconstruction_row("n1"), _reconstruction_row("n2", reachable=False)]
    from experiments.corporate_networks.run_experiment import summarise
    summary = summarise(rows)
    write_results(rows, summary, {"note": "test"}, tmp_path)
    assert (tmp_path / "README.md").exists()


def test_readme_substitutes_fake_reconstruction_and_ranking_numbers_from_summary():
    """Підміняємо `reconstruction`/`ranking_agreement`/`with_exploitability`
    на легко впізнавані унікальні значення — жодне з них не збігається з
    жодним числом, яке могло б випадково трапитись у статичному тексті
    шаблону (вагах критеріїв, відсотках методології тощо)."""
    from experiments.corporate_networks.run_experiment import render_readme
    rows = [_reconstruction_row("n1"), _reconstruction_row("n2", reachable=False)]
    from experiments.corporate_networks.run_experiment import summarise
    summary = summarise(rows)

    summary["reconstruction"]["all"]["coverage"]["mean"] = 0.123456
    summary["reconstruction"]["reachable_only"]["coverage"]["mean"] = 0.654321
    summary["with_exploitability"]["reconstruction"]["all"]["coverage"]["mean"] = 0.234567
    summary["ranking_agreement"]["best_match_share"] = 0.918273
    summary["ranking_agreement"]["spearman_mean"] = 0.019283
    summary["with_exploitability"]["ranking_agreement"]["best_match_share"] = 0.847362
    summary["with_exploitability"]["ranking_agreement"]["spearman_mean"] = 0.736251

    text = render_readme(summary, {"pythonhashseed": "0"})

    assert "0.123" in text   # reconstruction.all.coverage.mean
    assert "0.654" in text   # reconstruction.reachable_only.coverage.mean
    assert "0.235" in text   # with_exploitability.reconstruction.all.coverage.mean
    assert "0.918" in text   # ranking_agreement.best_match_share
    assert "0.019" in text   # ranking_agreement.spearman_mean
    assert "0.847" in text   # with_exploitability.ranking_agreement.best_match_share
    assert "0.736" in text   # with_exploitability.ranking_agreement.spearman_mean


def test_readme_substitutes_fake_defense_numbers_from_summary():
    from experiments.corporate_networks.run_experiment import render_readme, summarise
    rows = [_reconstruction_row("n1")]
    summary = summarise(rows)

    summary["defense"]["mean_best_size"] = 7.7777
    summary["defense"]["mean_best_cost"] = 3.3344
    summary["defense"]["unbroken_path_share"] = 0.445566
    summary["defense"]["networks_with_full_break_possible"] = 17
    summary["defense"]["networks_best_violates_constraints"] = 9

    text = render_readme(summary, {"pythonhashseed": "0"})

    assert "7.78" in text
    assert "3.33" in text
    # Частка нерозірваних шляхів рендериться відсотком (стиль оригінального
    # звіту — "7.7%"), а не сирим коефіцієнтом [0,1].
    assert "44.6%" in text
    assert "17" in text
    assert "9" in text


def test_readme_substitutes_fake_unevaluable_requirements_count_from_summary():
    """ЗАДАЧА 3: `networks_best_has_unevaluable_requirements` — окреме число
    від `networks_best_violates_constraints`, і воно теж має братись із
    `summary`, а не бути застиглим текстом."""
    from experiments.corporate_networks.run_experiment import render_readme, summarise
    rows = [_reconstruction_row("n1")]
    summary = summarise(rows)

    summary["defense"]["networks_best_violates_constraints"] = 0
    summary["defense"]["networks_best_has_unevaluable_requirements"] = 13

    text = render_readme(summary, {"pythonhashseed": "0"})

    assert "13" in text
    assert "нічим підтвердити" in text


def test_readme_reflects_whether_any_defense_set_has_unevaluable_requirements():
    """Той самий принцип, що й для `networks_best_violates_constraints`
    (сусідній тест): текст про неоцінювані вимоги має РЕАЛЬНО залежати від
    `summary`, а не бути однаковим незалежно від числа."""
    from experiments.corporate_networks.run_experiment import render_readme, summarise
    rows = [_reconstruction_row("n1")]
    summary = summarise(rows)

    summary["defense"]["networks_best_has_unevaluable_requirements"] = 0
    text_without_unevaluable = render_readme(summary, {"pythonhashseed": "0"})

    summary["defense"]["networks_best_has_unevaluable_requirements"] = 5
    text_with_unevaluable = render_readme(summary, {"pythonhashseed": "0"})

    assert text_without_unevaluable != text_with_unevaluable


def test_readme_distinguishes_violations_from_unevaluable_requirements():
    """Задача 3, головна гарантія тексту: «порушено» і «нічим підтвердити» —
    не одне й те саме речення з підставленим числом. Мережа БЕЗ жодного
    порушення, але З неоцінюваними вимогами, не повинна читатись так, ніби
    рекомендація «не проходить обмеження»."""
    from experiments.corporate_networks.run_experiment import render_readme, summarise
    rows = [_reconstruction_row("n1")]
    summary = summarise(rows)

    summary["defense"]["networks_best_violates_constraints"] = 0
    summary["defense"]["networks_best_has_unevaluable_requirements"] = 21
    text = render_readme(summary, {"pythonhashseed": "0"})

    # Абзац про порушення явно каже "жодного разу не порушує" — брак даних
    # не має підмінятись твердженням про порушення.
    assert "ЖОДНОГО разу не порушує" in text
    assert "нічим підтвердити" in text


def test_readme_lists_qualitative_example_networks_from_summary():
    """Поіменні приклади мереж (раніше вписані вручну) теж мають виходити
    з `summary["qualitative_examples"]`, а не бути застиглим текстом."""
    from experiments.corporate_networks.run_experiment import render_readme, summarise
    rows = [_reconstruction_row("n1")]
    summary = summarise(rows)
    summary["qualitative_examples"] = {
        "zero_cve_overlap_networks": ["net-fake-001", "net-fake-002"],
        "fully_inconvertible_attack_networks": ["net-fake-003"],
        "no_defense_possible_networks": ["net-fake-004"],
    }
    text = render_readme(summary, {"pythonhashseed": "0"})
    assert "net-fake-001" in text
    assert "net-fake-003" in text
    assert "net-fake-004" in text


def test_readme_reflects_whether_defense_constraints_were_ever_violated():
    """Раніше речення "обмеження ніколи не порушуються" було статичним
    текстом; тепер воно має описувати те, що РЕАЛЬНО каже `summary` —
    інакше після появи регуляторного обмеження (задача 2) текст мовчки
    збреше."""
    from experiments.corporate_networks.run_experiment import render_readme, summarise
    rows = [_reconstruction_row("n1")]
    summary = summarise(rows)

    summary["defense"]["networks_best_violates_constraints"] = 0
    text_without_violations = render_readme(summary, {"pythonhashseed": "0"})

    summary["defense"]["networks_best_violates_constraints"] = 4
    text_with_violations = render_readme(summary, {"pythonhashseed": "0"})

    assert text_without_violations != text_with_violations


def test_write_results_readme_file_matches_render_readme_output(tmp_path):
    """`write_results` не повинен мати власної, другої копії шаблону —
    файл на диску має бути РІВНО тим, що повертає `render_readme`."""
    from experiments.corporate_networks.run_experiment import (
        render_readme, summarise, write_results,
    )
    rows = [_reconstruction_row("n1"), _reconstruction_row("n2", reachable=False)]
    summary = summarise(rows)
    config = {"note": "test"}
    write_results(rows, summary, config, tmp_path)
    on_disk = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert on_disk == render_readme(summary, config)


if __name__ == "__main__":
    import pathlib
    import tempfile
    import traceback
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(pathlib.Path(td))
            else:
                fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
