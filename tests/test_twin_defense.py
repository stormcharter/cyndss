"""
Захисне плече: шляхи атаки (`qa.twin.enumerate_scenarios`) -> набори засобів
захисту, що їх розривають (`twin_defense.py`).

НАВІЩО ОКРЕМИЙ ФАЙЛ. `test_twin_model.py` і `test_twin_bridge.py` перевіряють
побудову й перебір ШЛЯХІВ АТАКИ. Тут - зворотне плече методу: за знайденими
шляхами підібрати НАБІР засобів захисту (мінімальна трансверсаль/hitting set)
і оцінити його інтегральною оцінкою J. Формули й крайові випадки інші, тож і
файл окремий.

Запуск:
    python tests/test_twin_defense.py
або pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa import twin  # noqa: E402
from twin_defense import (  # noqa: E402
    DEFAULT_DEFENSE_CRITERIA,
    coverage,
    cut_sets,
    defense_within_constraints,
    rank_defenses,
    unevaluable_constraints,
    vulnerabilities_on_path,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "qa", "fixtures", "corporate_12node.yaml",
)


def _load_corporate_fixture() -> "twin.TwinModel":
    with open(FIXTURE_PATH, encoding="utf-8") as handle:
        return twin.parse_twin_model(handle.read())


# Фікстура з 12 активами, 5 вразливостями й 8 засобами захисту - основний
# реалістичний випадок для цього модуля (те саме призначення, для якого вона
# й з'явилась у репо: планування засобів захисту, а не лише перебір атак).
CORP = _load_corporate_fixture()


def _edge_index(model: "twin.TwinModel", source: str, relation: str, target: str) -> int:
    for i, edge in enumerate(model.edges):
        if edge.source == source and edge.relation == relation and edge.target == target:
            return i
    raise AssertionError(f"фікстура не містить ребра {source} -{relation}-> {target}")


def _synthetic_model(nodes, edges, constraints=()):
    """Мінімальна модель для крайових випадків: будуємо `TwinModel` напряму,
    без `parse_twin_model` - тут важлива лише форма даних, яку читають функції
    `twin_defense`, а не повна схема документа (current_state/goal_state цим
    функціям узагалі не потрібні)."""
    return twin.TwinModel(
        name="synthetic",
        nodes=list(nodes),
        edges=list(edges),
        criteria=[twin.TwinCriterion("risk", "residual risk", 1.0, "benefit")],
        constraints=list(constraints),
        current_state_id="",
        goal_state_id="",
    )


# ---------------------------------------------------------------------------
# coverage(): control-вузол -> вразливості, які він РОЗРИВАЄ
# ---------------------------------------------------------------------------

def test_coverage_maps_a_control_to_the_vulnerabilities_it_mitigates():
    cov = coverage(CORP)
    assert cov["ctrl-fw"] == frozenset({"vuln-web"})
    assert cov["ctrl-patch"] == frozenset({"vuln-web", "vuln-db"})
    assert cov["ctrl-auth"] == frozenset({"vuln-ad"})


def test_coverage_excludes_detects_and_recovers_from():
    """ctrl-siem лише ДЕТЕКТУЄ vuln-ad, ctrl-backup лише ВІДНОВЛЮЄ після
    vuln-web. Жодне з двох не унеможливлює перехід - детектор бачить факт
    експлуатації вже ПІСЛЯ того, як вона сталась, відновлення повертає стан
    ПІСЛЯ інциденту. Для розриву шляху (трансверсаль) рахується лише
    mitigates; це свідоме змістове рішення, а не недогляд."""
    cov = coverage(CORP)
    assert "ctrl-siem" not in cov, "detects не мусить розривати шлях"
    assert "ctrl-backup" not in cov, "recovers_from не мусить розривати шлях"


def test_coverage_returns_nothing_when_the_model_has_no_control_nodes():
    model = _synthetic_model(
        nodes=[twin.TwinNode("v1", "vuln", "V1")],
        edges=[],
    )
    assert coverage(model) == {}


# ---------------------------------------------------------------------------
# vulnerabilities_on_path()
# ---------------------------------------------------------------------------

def test_vulnerabilities_on_path_finds_the_vuln_node_touched_by_the_edge():
    web_edge = _edge_index(CORP, "web01", "has_vuln", "vuln-web")
    assert vulnerabilities_on_path(CORP, [web_edge]) == frozenset({"vuln-web"})


def test_vulnerabilities_on_path_is_empty_for_an_empty_path():
    assert vulnerabilities_on_path(CORP, []) == frozenset()


# ---------------------------------------------------------------------------
# cut_sets(): крайові випадки на синтетичних моделях
# ---------------------------------------------------------------------------

def test_cut_sets_reports_the_path_unbreakable_when_there_are_no_controls():
    model = _synthetic_model(
        nodes=[twin.TwinNode("a1", "asset", "A1"), twin.TwinNode("v1", "vuln", "V1")],
        edges=[twin.TwinEdge("a1", "v1", "has_vuln")],
    )
    result = cut_sets(model, [[0]])
    assert result["sets"] == []
    assert result["unbreakable_path_indices"] == [0]


def test_cut_sets_reports_unbreakable_when_no_control_covers_the_paths_vulnerabilities():
    """Засіб у моделі є, але лікує ІНШУ вразливість - не ту, що на шляху."""
    model = _synthetic_model(
        nodes=[
            twin.TwinNode("a1", "asset", "A1"),
            twin.TwinNode("v1", "vuln", "V1"),
            twin.TwinNode("v2", "vuln", "V2 (не на жодному шляху)"),
            twin.TwinNode("ctrl", "control", "Control"),
        ],
        edges=[
            twin.TwinEdge("a1", "v1", "has_vuln"),
            twin.TwinEdge("ctrl", "v2", "mitigates"),
        ],
    )
    result = cut_sets(model, [[0]])
    assert result["sets"] == []
    assert result["unbreakable_path_indices"] == [0]


def test_cut_sets_finds_the_single_control_covering_every_path():
    model = _synthetic_model(
        nodes=[
            twin.TwinNode("a1", "asset", "A1"),
            twin.TwinNode("a2", "asset", "A2"),
            twin.TwinNode("v1", "vuln", "V1"),
            twin.TwinNode("v2", "vuln", "V2"),
            twin.TwinNode("ctrl", "control", "Control"),
        ],
        edges=[
            twin.TwinEdge("a1", "v1", "has_vuln"),      # 0
            twin.TwinEdge("a2", "v2", "has_vuln"),      # 1
            twin.TwinEdge("ctrl", "v1", "mitigates"),   # 2
            twin.TwinEdge("ctrl", "v2", "mitigates"),   # 3
        ],
    )
    result = cut_sets(model, [[0], [1]])
    assert len(result["sets"]) == 1
    only = result["sets"][0]
    assert only["controls"] == ("ctrl",)
    assert only["exact"] is True
    assert only["covers_all"] is True
    assert only["uncovered_path_indices"] == []
    assert result["unbreakable_path_indices"] == []


def _two_ties_model():
    """Один шлях з ДВОМА вразливостями, два засоби, кожен лікує лише одну -
    кожен з них ОКРЕМО повністю розриває єдиний шлях. Тест на два мінімальні
    набори ОДНАКОВОГО розміру (1), а не один."""
    return _synthetic_model(
        nodes=[
            twin.TwinNode("a1", "asset", "A1"),
            twin.TwinNode("v1", "vuln", "V1"),
            twin.TwinNode("v2", "vuln", "V2"),
            twin.TwinNode("ctrl-b", "control", "Control B"),
            twin.TwinNode("ctrl-a", "control", "Control A"),
        ],
        edges=[
            twin.TwinEdge("a1", "v1", "has_vuln"),       # 0
            twin.TwinEdge("v1", "v2", "leads_to"),       # 1 (синтетичний ланцюг: обидві vuln на одному шляху)
            twin.TwinEdge("ctrl-b", "v2", "mitigates"),  # 2
            twin.TwinEdge("ctrl-a", "v1", "mitigates"),  # 3
        ],
    )


def test_cut_sets_finds_two_equal_size_minimal_sets():
    model = _two_ties_model()
    result = cut_sets(model, [[0, 1]])
    found = {entry["controls"] for entry in result["sets"]}
    assert found == {("ctrl-a",), ("ctrl-b",)}
    assert all(entry["size"] == 1 and entry["exact"] for entry in result["sets"])


def test_cut_sets_order_does_not_depend_on_node_or_edge_declaration_order():
    """Порядок вузлів/ребер у документі не має впливати на впорядкування
    результату - інакше "перший" набір був би артефактом порядку в YAML, не
    змісту графа. Перевіряємо детермінізм, а не залежність від порядку
    словників (`coverage` будує їх з обходу `model.edges`)."""
    forward = _two_ties_model()
    reversed_model = twin.TwinModel(
        name=forward.name,
        nodes=list(reversed(forward.nodes)),
        edges=list(reversed(forward.edges)),
        criteria=forward.criteria,
        constraints=forward.constraints,
        current_state_id=forward.current_state_id,
        goal_state_id=forward.goal_state_id,
    )
    last = len(reversed_model.edges) - 1
    a = cut_sets(forward, [[0, 1]])
    b = cut_sets(reversed_model, [[last, last - 1]])
    assert [entry["controls"] for entry in a["sets"]] == [
        entry["controls"] for entry in b["sets"]
    ]


def test_cut_sets_reports_an_empty_result_for_an_empty_path_list():
    result = cut_sets(CORP, [])
    assert result == {"sets": [], "unbreakable_path_indices": [], "truncated": False}


# ---------------------------------------------------------------------------
# cut_sets(): реалістичний випадок на фікстурі
# ---------------------------------------------------------------------------

# П'ять незалежних "шляхів атаки" - кожен через одну вразливість фікстури.
# Кожен шлях - список ІНДЕКСІВ ребер, як їх повертає enumerate_scenarios у
# полі edge_path; тут для простоти по одному ребру has_vuln на шлях.
PATH_WEB = [_edge_index(CORP, "web01", "has_vuln", "vuln-web")]
PATH_DB = [_edge_index(CORP, "db01", "has_vuln", "vuln-db")]
PATH_AD = [_edge_index(CORP, "ad01", "has_vuln", "vuln-ad")]
PATH_WS = [_edge_index(CORP, "ws01", "has_vuln", "vuln-ws")]
PATH_WIFI = [_edge_index(CORP, "wif01", "has_vuln", "vuln-wifi")]
ALL_FIXTURE_PATHS = [PATH_WEB, PATH_DB, PATH_AD, PATH_WS, PATH_WIFI]


def test_cut_sets_on_the_fixture_finds_the_unique_minimal_set_within_max_size_four():
    # vuln-ad/vuln-ws/vuln-wifi мають по ОДНОМУ mitigates-засобу кожна
    # (ctrl-auth/ctrl-seg/ctrl-ids) - вони обов'язкові в будь-якому повному
    # розриві. vuln-web і vuln-db разом покриває ctrl-patch (по одному
    # засобу на кожну коштувало б розміру 5 - за межею max_size=4).
    result = cut_sets(CORP, ALL_FIXTURE_PATHS, max_size=4)
    assert len(result["sets"]) == 1
    only = result["sets"][0]
    assert only["controls"] == ("ctrl-auth", "ctrl-ids", "ctrl-patch", "ctrl-seg")
    assert only["exact"] is True
    assert only["covers_all"] is True
    assert only["uncovered_path_indices"] == []
    assert result["unbreakable_path_indices"] == []
    assert result["truncated"] is False


def test_cut_sets_falls_back_to_a_partial_set_when_max_size_is_too_small():
    """Мінімальний повний розрив - розмір 4 (перевірено вище). При max_size=3
    точного розриву не існує, тож функція повертає ОДИН прозоро позначений
    (`exact=False`) частковий набір і явно називає, який шлях лишився
    нерозірваним - замість мовчазної неповноти."""
    result = cut_sets(CORP, ALL_FIXTURE_PATHS, max_size=3)
    assert len(result["sets"]) == 1
    only = result["sets"][0]
    assert only["controls"] == ("ctrl-auth", "ctrl-ids", "ctrl-patch")
    assert only["size"] == 3
    assert only["exact"] is False
    assert only["covers_all"] is False
    assert only["uncovered_path_indices"] == [3]  # PATH_WS


# ---------------------------------------------------------------------------
# defense_within_constraints(): бюджет/місяці/персонал/regulatory
# ---------------------------------------------------------------------------

def test_defense_within_constraints_accepts_the_fixtures_minimal_set():
    assert (
        defense_within_constraints(
            CORP, ("ctrl-auth", "ctrl-ids", "ctrl-patch", "ctrl-seg")
        )
        is True
    )


def _tight_budget_model():
    """Один засіб коштує 100, бюджет моделі - лише 10: ПОКРИТТЯ є, а БЮДЖЕТУ -
    немає. Перевіряє, що `cut_sets` (лише про розрив шляху) і
    `defense_within_constraints` (лише про бюджет) - справді два незалежні
    питання, а не одне змішане."""
    return _synthetic_model(
        nodes=[
            twin.TwinNode("a1", "asset", "A1"),
            twin.TwinNode("v1", "vuln", "V1"),
            twin.TwinNode("ctrl", "control", "Expensive control",
                          attributes={"cost": 100.0}),
        ],
        edges=[
            twin.TwinEdge("a1", "v1", "has_vuln"),
            twin.TwinEdge("ctrl", "v1", "mitigates"),
        ],
        constraints=[twin.TwinConstraint("budget", 10.0)],
    )


def test_defense_within_constraints_rejects_a_set_that_breaks_the_budget():
    model = _tight_budget_model()
    # cut_sets і далі бачить повне покриття - бюджет тут ні до чого:
    covering = cut_sets(model, [[0]])["sets"][0]
    assert covering["controls"] == ("ctrl",)
    assert covering["covers_all"] is True
    # а обмеження - окрема, незалежна перевірка:
    assert defense_within_constraints(model, covering["controls"]) is False


# ---------------------------------------------------------------------------
# Третій стан: НЕОЦІНЮВАНЕ обмеження (regulatory без жодних даних у моделі)
#
# ПРОБЛЕМА, яку ці тести стережуть. `<task><constraints>` мереж датасету
# зводить текстову вимогу "відповідність ISO/IEC 27035" у `TwinConstraint
# ("regulatory", 1.0)`, але вихідні XML НЕ називають жодного поля
# "regulatory_approved" (`experiments/corporate_networks/dataset.py::
# _parse_defense` читає лише kind/product/state) - тож ЖОДЕН control-вузол
# датасету НІКОЛИ не матиме цього атрибута. Стара двійкова перевірка тоді
# трактувала відсутність ДАНИХ як доведене ПОРУШЕННЯ - будь-який непорожній
# набір засобів автоматично "не проходив" вимогу, якої фізично нема з чим
# звірити. Це і є різниця, яку перевіряють тести нижче: "порушено" (є дані,
# і вони не сходяться) відрізняється від "нічим підтвердити" (даних нема
# взагалі, ні в кандидаті, ні деінде в моделі).
# ---------------------------------------------------------------------------

def _regulatory_unevaluable_model():
    """Жоден control-вузол моделі не має `regulatory_approved` - НАВІТЬ той,
    що не входить до жодного розглянутого набору. Обмеження `regulatory`
    активне (`limit=1`), але дані про затвердження відсутні структурно: жоден
    набір НІКОЛИ не зможе довести відповідність, тож перевірка має визнати
    вимогу НЕОЦІНЮВАНОЮ, а не автоматично порушеною."""
    return _synthetic_model(
        nodes=[
            twin.TwinNode("a1", "asset", "A1"),
            twin.TwinNode("v1", "vuln", "V1"),
            twin.TwinNode("ctrl", "control", "Control"),
        ],
        edges=[
            twin.TwinEdge("a1", "v1", "has_vuln"),
            twin.TwinEdge("ctrl", "v1", "mitigates"),
        ],
        constraints=[twin.TwinConstraint("regulatory", 1.0)],
    )


def _regulatory_violation_model():
    """На відміну від попередньої фікстури, ТУТ у моделі Є control-вузол з
    `regulatory_approved` (`ctrl-other`) - хай і не той, що покриває
    розглянутий шлях (він лікує ІНШУ вразливість, `v2`, поза розглянутими
    шляхами). Дані про затвердження в моделі ІСНУЮТЬ, тож вимога ОЦІНЮВАНА -
    і єдиний кандидат (`ctrl`, без власного атрибута) РЕАЛЬНО порушує її, як
    і до введення третього стану."""
    return _synthetic_model(
        nodes=[
            twin.TwinNode("a1", "asset", "A1"),
            twin.TwinNode("v1", "vuln", "V1"),
            twin.TwinNode("ctrl", "control", "Control"),
            twin.TwinNode("a2", "asset", "A2"),
            twin.TwinNode("v2", "vuln", "V2 (не на розглянутому шляху)"),
            twin.TwinNode("ctrl-other", "control", "Other control",
                          attributes={"regulatory_approved": True}),
        ],
        edges=[
            twin.TwinEdge("a1", "v1", "has_vuln"),
            twin.TwinEdge("ctrl", "v1", "mitigates"),
            twin.TwinEdge("a2", "v2", "has_vuln"),
            twin.TwinEdge("ctrl-other", "v2", "mitigates"),
        ],
        constraints=[twin.TwinConstraint("regulatory", 1.0)],
    )


def test_unevaluable_constraints_is_empty_without_any_constraints():
    model = _synthetic_model(nodes=[twin.TwinNode("ctrl", "control", "Control")], edges=[])
    assert unevaluable_constraints(model) == []


def test_unevaluable_constraints_flags_regulatory_when_no_control_carries_the_attribute():
    assert unevaluable_constraints(_regulatory_unevaluable_model()) == ["regulatory"]


def test_unevaluable_constraints_is_empty_when_some_control_in_the_model_carries_the_attribute():
    """Дані десь у моделі Є (`ctrl-other`), навіть якщо не в розглянутому
    наборі - вимога лишається ОЦІНЮВАНОЮ для ВСІЄЇ моделі."""
    assert unevaluable_constraints(_regulatory_violation_model()) == []


def test_defense_within_constraints_ignores_an_unevaluable_regulatory_requirement():
    """Жоден control моделі не має `regulatory_approved` - вимога
    НЕОЦІНЮВАНА, тож не повинна автоматично відкидати набір, що інакше (за
    покриттям і бюджетом) прийнятний."""
    model = _regulatory_unevaluable_model()
    assert defense_within_constraints(model, ("ctrl",)) is True


def test_defense_within_constraints_still_rejects_when_regulatory_data_exists_elsewhere():
    """Дані про затвердження в моделі є (на ІНШОМУ control-вузлі) - вимога
    ОЦІНЮВАНА, і кандидат без власного `regulatory_approved` РЕАЛЬНО порушує
    її, як і до введення третього стану."""
    model = _regulatory_violation_model()
    assert defense_within_constraints(model, ("ctrl",)) is False


# ---------------------------------------------------------------------------
# rank_defenses(): інтегральна оцінка J набору захисних заходів
# ---------------------------------------------------------------------------

def test_default_defense_criteria_weights_sum_to_one():
    assert abs(sum(c.weight for c in DEFAULT_DEFENSE_CRITERIA) - 1.0) < 1e-9
    assert {c.direction for c in DEFAULT_DEFENSE_CRITERIA} <= {"benefit", "cost"}


def test_rank_defenses_scores_the_fixtures_best_set():
    result = rank_defenses(CORP, ALL_FIXTURE_PATHS, max_size=4)
    assert len(result["ranked"]) == 1
    best = result["ranked"][0]
    assert best["controls"] == ("ctrl-auth", "ctrl-ids", "ctrl-patch", "ctrl-seg")
    assert best["within_constraints"] is True
    # coverage=1.0 (усі 5 шляхів розірвано), cost=15000/100000=0.15 (інверт.
    # -> 0.85), size=4/4=1.0 (інверт. -> 0.0); J=1.0*0.5+0.85*0.3+0.0*0.2=0.755
    assert abs(best["j"] - 0.755) < 1e-6
    assert {c["criterion"] for c in best["criteria"]} == {"coverage", "cost", "size"}
    assert result["excluded_by_constraints"] == []
    assert result["unbreakable_path_indices"] == []


def test_rank_defenses_returns_nothing_for_no_attack_paths():
    # СВІДОМА ПРАВКА ІСНУЮЧОГО ТЕСТА: контракт `rank_defenses` розширено з
    # двох кошиків до трьох (третій - `unevaluable_requirements`, набори, що
    # проходять усе ОЦІНЮВАНЕ, але мають вимогу, якої нічим підтвердити).
    # Порожній вхід так само порожній у ВСІХ трьох кошиках - точна рівність
    # словника лишається сильною перевіркою форми відповіді, лише з новим
    # полем.
    result = rank_defenses(CORP, [])
    assert result == {
        "ranked": [],
        "excluded_by_constraints": [],
        "unevaluable_requirements": [],
        "unbreakable_path_indices": [],
        "truncated": False,
    }


def test_rank_defenses_excludes_a_covering_set_that_breaks_the_budget():
    """Той самий крайовий випадок, що й для `defense_within_constraints`, але
    видимий через публічний вихід `rank_defenses`: набір, що ПРОХОДИТЬ
    покриття, але НЕ проходить бюджет, не має тихо зникати - він має
    лишитись видимим аналітику, лише в іншому кошику."""
    model = _tight_budget_model()
    result = rank_defenses(model, [[0]])
    assert result["ranked"] == []
    assert len(result["excluded_by_constraints"]) == 1
    excluded = result["excluded_by_constraints"][0]
    assert excluded["controls"] == ("ctrl",)
    assert excluded["covers_all"] is True
    assert excluded["within_constraints"] is False


def test_rank_defenses_accepts_custom_criteria():
    """`criteria=None` бере `DEFAULT_DEFENSE_CRITERIA`; явний список підміняє
    його - інакше параметр був би оздобою без ефекту."""
    model = _tight_budget_model()
    only_coverage = [twin.TwinCriterion("coverage", "лише покриття", 1.0, "benefit")]
    default_j = rank_defenses(model, [[0]])["excluded_by_constraints"][0]["j"]
    custom_j = rank_defenses(model, [[0]], criteria=only_coverage)["excluded_by_constraints"][0]["j"]
    assert custom_j == 1.0  # повне покриття єдиного шляху, інших критеріїв нема
    assert default_j != custom_j


# ---------------------------------------------------------------------------
# rank_defenses(): третій кошик - "проходить усе оцінюване, але має вимогу,
# якої нічим підтвердити" (`unevaluable_requirements`).
# ---------------------------------------------------------------------------

def test_rank_defenses_returns_unevaluable_requirements_group_when_regulatory_data_is_absent():
    """Набір, що повністю розриває шлях і не порушує жодного ОЦІНЮВАНОГО
    обмеження, але має вимогу, якої модель у принципі не може підтвердити
    (regulatory без жодного `regulatory_approved` в усій моделі), - НЕ
    порушник (`excluded_by_constraints`) і НЕ «просто прохідний» (`ranked`),
    а третя, окрема група."""
    model = _regulatory_unevaluable_model()
    result = rank_defenses(model, [[0]])
    assert result["ranked"] == []
    assert result["excluded_by_constraints"] == []
    assert len(result["unevaluable_requirements"]) == 1
    entry = result["unevaluable_requirements"][0]
    assert entry["controls"] == ("ctrl",)
    assert entry["within_constraints"] is True
    assert entry["unevaluable_constraints"] == ["regulatory"]


def test_rank_defenses_mixed_case_keeps_a_real_budget_violation_a_violation():
    """Змішаний випадок вимоги задачі: бюджет ОЦІНЮВАНИЙ і порушений
    (реальні ціни) + regulatory НЕОЦІНЮВАНИЙ (жодних даних про затвердження
    в усій моделі). Порушення реального обмеження не повинно розчинитись у
    третьому стані - "не послаблюй перевірку там, де дані є"."""
    model = _synthetic_model(
        nodes=[
            twin.TwinNode("a1", "asset", "A1"),
            twin.TwinNode("v1", "vuln", "V1"),
            twin.TwinNode("ctrl", "control", "Expensive control",
                          attributes={"cost": 100.0}),
        ],
        edges=[
            twin.TwinEdge("a1", "v1", "has_vuln"),
            twin.TwinEdge("ctrl", "v1", "mitigates"),
        ],
        constraints=[
            twin.TwinConstraint("budget", 10.0),
            twin.TwinConstraint("regulatory", 1.0),
        ],
    )
    result = rank_defenses(model, [[0]])
    assert result["ranked"] == []
    assert result["unevaluable_requirements"] == []
    assert len(result["excluded_by_constraints"]) == 1
    excluded = result["excluded_by_constraints"][0]
    assert excluded["within_constraints"] is False
    # Метадана про неоцінюваність лишається видимою навіть тут - аналітик
    # бачить ОБИДВІ причини, а не лише ту, що визначила кошик.
    assert excluded["unevaluable_constraints"] == ["regulatory"]


def test_rank_defenses_unevaluable_requirements_group_is_empty_when_the_fixture_has_the_attribute():
    """Регрес: фікстура `CORP` має `regulatory_approved` на КОЖНОМУ
    control-вузлі - вимога оцінювана й задоволена, тож третя група лишається
    порожньою, як і до введення третього стану."""
    result = rank_defenses(CORP, ALL_FIXTURE_PATHS, max_size=4)
    assert result["unevaluable_requirements"] == []


# --- ручний запуск без pytest --------------------------------------------
if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
