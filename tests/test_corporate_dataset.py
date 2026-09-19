"""
Адаптер `experiments/corporate_networks/dataset.py`: XML корпоративних мереж
-> типізований орієнтований `qa.twin.TwinModel`.

ГОЛОВНА ГАРАНТІЯ, яку тут стережемо: `<attacks>` у кожному `net-*.xml` —
прихований еталон (той самий ланцюг, що й `attack_path` у HTB-експерименті).
Функція побудови моделі (`parse_network` / `build_twin_model`) не має права
його читати НІЯК — ні вузлом, ні ребром, ні атрибутом. Це стереже
`test_gold_never_leaks_into_the_model`, за зразком
`tests/test_twin_bridge.py::test_gold_never_leaks_into_the_model`.

Ідуть і під pytest, і напряму: `python tests/test_corporate_dataset.py`.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa.twin import ALLOWED_KINDS, ALLOWED_STATE_TYPES, TwinConstraint  # noqa: E402
from experiments.corporate_networks.dataset import (  # noqa: E402
    build_twin_model,
    cve_exploitability_index,
    load_answers,
    parse_gold_attacks,
    parse_network,
    reproduce_effectiveness,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_DIR = os.path.join(_REPO_ROOT, "experiments", "corporate_networks", "dataset")


def _read(network_id: str) -> str:
    path = os.path.join(_DATASET_DIR, f"{network_id}.xml")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _net001() -> dict:
    return parse_network(_read("net-001"))


def _model001():
    return build_twin_model(_net001())


# --- ПРЕДОХРАНИТЕЛЬ: еталон не тече в модель --------------------------------
def test_gold_never_leaks_into_the_model():
    """Вилучення `<attacks>` з XML не має змінити жоден вузол чи ребро."""
    xml_text = _read("net-001")
    without_attacks = re.sub(r"<attacks>.*</attacks>", "", xml_text, flags=re.S)
    assert "<attack " not in without_attacks  # сурогатна перевірка регексу

    full = build_twin_model(parse_network(xml_text))
    blind = build_twin_model(parse_network(without_attacks))
    assert [(n.id, n.kind, n.label) for n in full.nodes] == \
           [(n.id, n.kind, n.label) for n in blind.nodes]
    assert [(e.source, e.relation, e.target) for e in full.edges] == \
           [(e.source, e.relation, e.target) for e in blind.edges]


def test_parse_network_dict_has_no_attacks_key():
    """`parse_network` навіть не повертає ключа `attacks` — його там немає
    структурно, а не просто "порожній"."""
    assert "attacks" not in _net001()


def test_parse_network_output_contains_no_attack_step_identifiers():
    """Сурогатний детектор витоку: серіалізований результат `parse_network`
    не повинен містити жодного значення, яке трапляється ЛИШЕ в `<attacks>`
    (атрибут `steps` — лічильник кроків атаки, унікальний для цього блока)."""
    dumped = json.dumps(_net001(), ensure_ascii=False)
    assert "steps" not in dumped


def test_parse_gold_attacks_reads_only_the_attacks_block():
    attacks = parse_gold_attacks(_read("net-001"))
    ids = {a["id"] for a in attacks}
    assert ids == {"a1", "a2", "a3", "a4", "a5"}
    a2 = next(a for a in attacks if a["id"] == "a2")
    assert [s["cve"] for s in a2["steps"]] == [
        "CVE-2020-1472", "CVE-2021-21974", "CVE-2021-34527",
    ]
    assert a2["steps"][0]["from"] == "web-01"
    assert a2["steps"][0]["to"] == "db-01"
    assert a2["steps"][0]["mitre"] == ["T1068", "T1078"]


def test_parse_gold_attacks_returns_nothing_when_attacks_block_is_absent():
    xml_text = _read("net-001")
    without_attacks = re.sub(r"<attacks>.*</attacks>", "", xml_text, flags=re.S)
    assert parse_gold_attacks(without_attacks) == []


# --- граф справді орієнтований і типізований --------------------------------
def test_every_node_kind_is_one_twin_understands():
    for node in _model001().nodes:
        assert node.kind in ALLOWED_KINDS, (node.id, node.kind)


def test_state_types_are_from_the_allowed_set():
    for node in _model001().nodes:
        if node.kind == "state":
            st = node.attributes.get("state_type")
            assert st in ALLOWED_STATE_TYPES, (node.id, st)


def test_edges_are_named_and_directed_not_symmetric():
    model = _model001()
    assert model.edges
    assert all(e.relation for e in model.edges)
    pairs = {(e.source, e.target) for e in model.edges}
    mirrored = {(t, s) for s, t in pairs} & pairs
    assert not mirrored, f"знайдено симетричні пари: {sorted(mirrored)}"


def test_every_edge_declares_whether_it_is_a_transition():
    for edge in _model001().edges:
        assert "transition" in edge.attributes, edge.relation


def test_no_duplicate_node_ids():
    ids = [n.id for n in _model001().nodes]
    assert len(ids) == len(set(ids)), "дублікати id: " + str(
        sorted({i for i in ids if ids.count(i) > 1})
    )


# --- розміщення активу в підмережі (за id, а не за зоною) -------------------
def test_asset_is_located_in_its_subnet_by_id():
    model = _model001()
    located = {(e.source, e.target) for e in model.edges if e.relation == "located_in"}
    assert ("web-01", "SUBNET:dmz") in located
    # siem-01 лежить у підмережі з id="mgmt", а зона цієї підмережі -
    # "management" ("management" != "mgmt"); located_in зіставляє САМЕ ЗА id.
    assert ("siem-01", "SUBNET:mgmt") in located
    assert model.node_by_id["SUBNET:mgmt"].attributes["zone"] == "management"


def test_located_in_is_not_a_transition():
    model = _model001()
    for edge in model.edges:
        if edge.relation == "located_in":
            assert edge.attributes.get("transition") is False


# --- вразливості --------------------------------------------------------
def test_vulnerability_node_carries_the_required_attributes():
    model = _model001()
    node = model.node_by_id["VULN:web-01:CVE-2023-34362"]
    assert node.kind == "vuln"
    assert node.attributes["cve"] == "CVE-2023-34362"
    assert node.attributes["cvss"] == 9.8
    assert node.attributes["exploitability"] == 0.980
    assert node.attributes["exploit_class"] == "initial-access"
    assert node.attributes["mitre"] == ["T1190"]
    assert node.attributes["external_ids"] == ["CVE-2023-34362"]


def test_asset_has_vuln_edge_for_both_service_and_os_vulnerabilities():
    model = _model001()
    has_vuln = {(e.source, e.target) for e in model.edges if e.relation == "has_vuln"}
    assert ("web-01", "VULN:web-01:CVE-2023-34362") in has_vuln  # сервісна
    assert ("web-01", "VULN:web-01:CVE-2021-4034") in has_vuln   # ОС
    assert all(
        e.attributes.get("transition") is True
        for e in model.edges if e.relation == "has_vuln"
    )


def test_same_cve_on_different_assets_gets_distinct_vuln_nodes():
    """CVE-2021-4034 сидить одразу на трьох активах net-001 (web-01, nas-01,
    siem-01) — це РІЗНІ встановлення, а не одна вразливість."""
    model = _model001()
    ids = {n.id for n in model.nodes if n.attributes.get("cve") == "CVE-2021-4034"}
    assert ids == {
        "VULN:web-01:CVE-2021-4034",
        "VULN:nas-01:CVE-2021-4034",
        "VULN:siem-01:CVE-2021-4034",
    }


# --- потоки та політика ------------------------------------------------
def test_flow_becomes_a_directed_edge_with_port_and_kind():
    model = _model001()
    flows = [e for e in model.edges if e.relation == "flow"
             and e.source == "adm-01" and e.target == "dc-01"]
    assert flows, "потік adm-01 -> dc-01 має стати ребром"
    assert flows[0].attributes["kind"] == "domain-auth"
    assert flows[0].attributes["port"] == "88"
    assert flows[0].attributes["transition"] is True


def test_policy_allow_rule_becomes_an_edge_between_zone_resources():
    model = _model001()
    allowed = [e for e in model.edges if e.relation == "policy_allows"]
    pairs = {(e.source, e.target) for e in allowed}
    # internet -> dmz allow tcp 80,443
    assert ("ZONE:internet", "SUBNET:dmz") in pairs


def test_policy_deny_rule_creates_no_edge():
    """internet -> lan заборонено (`action="deny"`) — жодного ребра."""
    model = _model001()
    pairs = {(e.source, e.target) for e in model.edges if e.relation == "policy_allows"}
    assert ("ZONE:internet", "SUBNET:lan") not in pairs


def test_management_zone_policy_edge_targets_the_mgmt_subnet_node():
    """Ключова перевірка: правило політики іменує ЗОНУ ("management"), а
    вузол підмережі має id за ІНШИМ полем (`mgmt`). Обидва ребра мають
    сходитись в одному й тому самому вузлі."""
    model = _model001()
    pairs = {(e.source, e.target) for e in model.edges if e.relation == "policy_allows"}
    assert any(target == "SUBNET:mgmt" for _src, target in pairs) or \
           any(src == "SUBNET:mgmt" for src, _tgt in pairs)


# --- облікові записи -----------------------------------------------------
def test_domain_account_is_a_privilege_state_node():
    model = _model001()
    node = model.node_by_id["ACCT:Administrator"]
    assert node.kind == "state"
    assert node.attributes["state_type"] == "privilege"
    assert node.attributes["privilege"] == "domain-admin"


def test_local_account_is_scoped_to_its_own_asset():
    """`svc-web-01` (domain=false) — локальний сервісний обліковий запис,
    прив'язаний саме до web-01, а не спільний принципал."""
    model = _model001()
    assert "ACCT:web-01:svc-web-01" in model.node_by_id
    assert "ACCT:svc-web-01" not in model.node_by_id


def test_same_domain_account_name_on_two_assets_merges_into_one_node():
    """`backup-operator` (domain=true) названо і на db-01, і на nas-01, і на
    esx-01 без жодного `reusedOn` — це той самий доменний обліковий запис."""
    model = _model001()
    has_account = {(e.source, e.target) for e in model.edges if e.relation == "has_account"}
    node_id = "ACCT:backup-operator"
    assert node_id in model.node_by_id
    assert (model.node_by_id[node_id].kind, model.node_by_id[node_id].attributes["state_type"]) == \
           ("state", "privilege")
    assert ("db-01", node_id) in has_account
    assert ("nas-01", node_id) in has_account
    assert ("esx-01", node_id) in has_account


def test_reused_credential_adds_a_has_account_edge_on_the_target_asset():
    """`helpdesk` на adm-01 має `<reusedOn asset="nas-01"/>`, хоча в nas-01
    власного `<account name="helpdesk">` немає — ребро додається ЛИШЕ тому,
    що поле `reusedOn` це прямо стверджує."""
    model = _model001()
    has_account = {(e.source, e.target) for e in model.edges if e.relation == "has_account"}
    node_id = "ACCT:helpdesk"
    assert ("adm-01", node_id) in has_account
    assert ("nas-01", node_id) in has_account


def test_has_account_is_not_a_transition():
    model = _model001()
    for edge in model.edges:
        if edge.relation == "has_account":
            assert edge.attributes.get("transition") is False


# --- засоби захисту й mitigates/detects/recovers_from -----------------------
def test_every_defense_becomes_a_control_node_regardless_of_state():
    """dc-01 має вимкнений backup — вузол все одно існує (це факт
    інвентаризації захисту), просто без ребра лікування."""
    model = _model001()
    disabled = [n for n in model.nodes
                if n.kind == "control" and n.attributes.get("kind") == "backup"
                and n.attributes.get("state") == "disabled"]
    assert disabled, "очікували вимкнений backup-контроль на dc-01"


def test_disabled_defense_gets_no_treatment_edge():
    model = _model001()
    disabled_ctrl = next(
        n.id for n in model.nodes
        if n.kind == "control" and n.attributes.get("kind") == "backup"
        and n.attributes.get("state") == "disabled"
    )
    outgoing = model.outgoing(disabled_ctrl)
    assert not outgoing, f"вимкнений засіб не повинен нічого лікувати: {outgoing}"


def test_enabled_firewall_mitigates_every_vulnerability_of_the_same_asset():
    model = _model001()
    ctrl = next(
        n.id for n in model.nodes
        if n.kind == "control" and n.attributes.get("kind") == "firewall-host"
        and n.id.startswith("CTRL:web-01:")
    )
    mitigated = {e.target for e in model.outgoing(ctrl) if e.relation == "mitigates"}
    assert mitigated == {
        "VULN:web-01:CVE-2023-34362",
        "VULN:web-01:CVE-2017-5638",
        "VULN:web-01:CVE-2021-4034",
    }


def test_enabled_siem_agent_detects_rather_than_mitigates():
    """`siem-agent` на siem-01 (Filebeat, enabled) спостерігає, а не
    запобігає — той самий поділ, що вже закладений у
    `qa/fixtures/corporate_12node.yaml` (ctrl-siem -> detects)."""
    model = _model001()
    ctrl = next(
        n.id for n in model.nodes
        if n.kind == "control" and n.attributes.get("kind") == "siem-agent"
        and n.id.startswith("CTRL:siem-01:")
    )
    relations = {e.relation for e in model.outgoing(ctrl)}
    assert relations == {"detects"}


def test_enabled_backup_recovers_from_rather_than_mitigates():
    """`backup` на siem-01 (Veeam Agent for Linux, enabled) — той самий
    поділ, що вже закладений у corporate_12node.yaml (ctrl-backup ->
    recovers_from)."""
    model = _model001()
    ctrl = next(
        n.id for n in model.nodes
        if n.kind == "control" and n.attributes.get("kind") == "backup"
        and n.id.startswith("CTRL:siem-01:")
    )
    relations = {e.relation for e in model.outgoing(ctrl)}
    assert relations == {"recovers_from"}


def test_treatment_edges_never_cross_between_assets():
    """Захист на одному активі не може «лікувати» вразливість іншого —
    зіставлення `mitigates`/`detects`/`recovers_from` завжди в межах ОДНОГО
    й того самого active (див. коментар у `build_twin_model`)."""
    model = _model001()
    from qa.twin import TREATMENT_RELATIONS
    ctrl_asset = {n.id: n.id.split(":")[1] for n in model.nodes if n.kind == "control"}
    vuln_asset = {n.id: n.id.split(":")[1] for n in model.nodes if n.kind == "vuln"}
    for edge in model.edges:
        if edge.relation in TREATMENT_RELATIONS:
            assert ctrl_asset[edge.source] == vuln_asset[edge.target], edge


def test_every_defense_kind_present_in_the_dataset_maps_to_a_known_relation():
    """Усі 7 значень `kind`, які реально трапляються в датасеті, повинні
    мати визначену категорію лікування (жодна не мовчки ігнорується)."""
    from experiments.corporate_networks.dataset import _DEFENSE_RELATION
    assert set(_DEFENSE_RELATION) == {
        "ids", "firewall-host", "backup", "edr", "siem-agent", "av", "mfa",
    }
    assert set(_DEFENSE_RELATION.values()) <= {"mitigates", "detects", "recovers_from"}


# --- початковий і цільовий стан -----------------------------------------
def test_initial_state_matches_the_task_element():
    model = _model001()
    assert model.current_state_id == "STATE:web-01:compromised"
    node = model.node_by_id[model.current_state_id]
    assert node.kind == "state"
    assert node.attributes["asset"] == "web-01"
    assert node.attributes["condition"] == "compromised"


def test_goal_state_matches_the_task_element():
    model = _model001()
    assert model.goal_state_id == "STATE:dc-01:domain-admin"
    node = model.node_by_id[model.goal_state_id]
    assert node.kind == "state"
    assert node.attributes["asset"] == "dc-01"
    assert node.attributes["condition"] == "domain-admin"


def test_initial_state_is_granted_by_the_via_cve_vulnerability():
    """`<initialState viaCve="CVE-2023-34362">` — це НАЗВАНЕ поле, тож
    ребро `grants` від відповідного вузла вразливості виправдане, а не
    вигадане."""
    model = _model001()
    granted_by = [e.source for e in model.edges
                  if e.relation == "grants" and e.target == model.current_state_id]
    assert granted_by == ["VULN:web-01:CVE-2023-34362"]


def test_networks_without_via_cve_have_no_grants_edge():
    """net-002: `initialState` без `viaCve` (вхід — фішинг, не CVE) —
    ребра `grants` у поточний стан немає, бо немає поля, яке б його
    виправдало."""
    model = build_twin_model(parse_network(_read("net-002")))
    granted_by = [e for e in model.edges
                  if e.relation == "grants" and e.target == model.current_state_id]
    assert granted_by == []


# --- переходи атаки: досяжність активу з активу + стан->вразливість->стан --
# ПРОБЛЕМА, яку фіксує цей блок: до фікса `STATE:web-01:compromised` мав 0
# вихідних ребер, `STATE:dc-01:domain-admin` — 0 вхідних, `expand_forward`
# давав 1 вузол/0 ребер, `enumerate_scenarios` — 0 шляхів на ВСІХ 100
# мережах. Граф описував структуру (located_in/has_vuln/flow/policy_allows/
# has_account/mitigates), але не описував, як зі стану на одному активі
# потрапити в стан на іншому.
def test_current_state_has_outgoing_transition_edges():
    """До фікса `STATE:web-01:compromised` мав 0 вихідних ребер —
    `expand_forward` не міг зрушити з місця. З публічних правил досяжності
    (підмережа/flow/policy) має з'явитися хоча б один вихідний перехід."""
    model = _model001()
    assert model.outgoing(model.current_state_id), (
        "початковий стан net-001 має мати хоча б один вихідний перехід"
    )


def test_exploit_edge_declares_which_reachability_rule_produced_it():
    """Кожне ребро `exploits` (стан → вразливість) несе атрибут, ЯКИМ саме
    з трьох публічних правил (підмережа/flow/policy) підтверджена
    досяжність — без цього результат неможливо розібрати по правилах."""
    model = _model001()
    exploit_edges = [e for e in model.edges if e.relation == "exploits"]
    assert exploit_edges, "очікували хоча б одне ребро exploits на net-001"
    for edge in exploit_edges:
        assert edge.attributes.get("reachability_rule") in {
            "same_subnet", "flow", "policy",
        }, edge
        assert edge.attributes.get("transition") is True


def test_exploits_web01_to_dc01_via_the_dmz_to_server_policy_rule():
    """net-001: правило політики дозволяє `dmz -> server`, тож web-01
    (зона dmz) має діставати вразливості dc-01 (зона server) правилом
    `policy` — конкретний перевірений вручну випадок, а не лише "щось є"."""
    model = _model001()
    edge = next(
        e for e in model.edges
        if e.relation == "exploits"
        and e.source == "STATE:web-01:compromised"
        and e.target == "VULN:dc-01:CVE-2021-34527"
    )
    assert edge.attributes["reachability_rule"] == "policy"


def test_vulnerability_yields_a_condition_on_its_own_asset():
    """`yields` — модельне припущення "ця вразливість НА ЦЬОМУ активі дає
    таку умову", окреме від `grants` (те лишається лише для явного
    `viaCve` задачі — факту, а не припущення)."""
    model = _model001()
    yields_edges = {(e.source, e.target) for e in model.edges if e.relation == "yields"}
    assert ("VULN:dc-01:CVE-2021-34527", "STATE:dc-01:domain-admin") in yields_edges


def test_privilege_escalation_condition_depends_on_domain_admin_account():
    """Той самий `exploitClass="privilege-escalation"`, різний наслідок:
    dc-01 має обліковий запис `privilege="domain-admin"` → ескалація дає
    domain-admin; web-01 такого облікового запису не має → local-admin."""
    model = _model001()
    yields_edges = {(e.source, e.target) for e in model.edges if e.relation == "yields"}
    assert ("VULN:dc-01:CVE-2021-34527", "STATE:dc-01:domain-admin") in yields_edges
    assert ("VULN:web-01:CVE-2021-4034", "STATE:web-01:local-admin") in yields_edges


def test_unmapped_exploit_class_produces_no_condition():
    """Захист від вигадування: exploitClass поза чотирма відомими
    значеннями не повинен приносити жодної умови — код мовчить, а не
    вгадує."""
    from experiments.corporate_networks.dataset import _exploit_class_condition
    assert _exploit_class_condition("unknown-class", {"accounts": []}) == ""


def test_reachability_transitions_do_not_depend_on_the_attacks_block():
    """Додатковий, вужчий запобіжник поверх `test_gold_never_leaks_into_the_
    model`: САМЕ нові переходи (`exploits`/`yields`) не змінюються, якщо
    вирізати `<attacks>` з XML — вони виводяться виключно з підмереж, flow,
    політики й облікових записів."""
    xml_text = _read("net-001")
    without_attacks = re.sub(r"<attacks>.*</attacks>", "", xml_text, flags=re.S)
    full = build_twin_model(parse_network(xml_text))
    blind = build_twin_model(parse_network(without_attacks))

    def transitions(model):
        return sorted(
            (e.source, e.relation, e.target, e.attributes.get("reachability_rule"))
            for e in model.edges if e.relation in {"exploits", "yields"}
        )

    assert transitions(full) == transitions(blind)
    assert transitions(full), "очікували непорожній список переходів"


def test_enumerate_scenarios_finds_a_path_from_current_to_goal_for_net_001():
    """Наскрізна перевірка повного конвеєра (forward/backward/reconcile/
    enumerate): після фікса він справді знаходить шлях web-01 → … → dc-01,
    а не 0 шляхів, як було до фікса (виміряно в постановці задачі)."""
    from qa.twin import enumerate_scenarios, expand_backward, expand_forward, reconcile
    model = _model001()
    forward = expand_forward(model)
    backward = expand_backward(model)
    gc = reconcile(model, forward, backward)
    result = enumerate_scenarios(model, gc, matched=set(), max_paths=5)
    assert result["scenarios"], "очікували хоча б один сценарій на net-001"
    for scenario in result["scenarios"]:
        assert scenario["path"][0] == model.current_state_id
        assert scenario["path"][-1] == model.goal_state_id


def test_goal_state_reachability_across_the_dataset():
    """Скільки зі 100 мереж мають ціль, досяжну в побудованому графі (BFS
    уперед від початкового стану) — вимірюємо чесно, НЕ підганяючи правило
    під 100%: якщо десь ціль недосяжна — це реальний результат експерименту,
    а не помилка тесту. Точне число — у звіті задачі."""
    from qa.twin import expand_forward
    xml_paths = sorted(glob.glob(os.path.join(_DATASET_DIR, "net-*.xml")))
    assert len(xml_paths) == 100
    reachable = 0
    for xml_path in xml_paths:
        with open(xml_path, "r", encoding="utf-8") as f:
            network = parse_network(f.read())
        model = build_twin_model(network)
        forward = expand_forward(model)
        if model.goal_state_id in forward["node_ids"]:
            reachable += 1
    print(f"\nціль досяжна у {reachable}/100 мереж")
    # Точне число зафіксовано як регрес-запобіжник ПІСЛЯ чесного вимірювання
    # (той самий прийом, що й `checked == 500` у тесті ефективності вище) —
    # не мета, під яку підганяли правило: 35/100 мереж, де ціль лишається
    # недосяжною за трьома публічними правилами, — реальний результат
    # експерименту, а не помилка. Деталі — у звіті задачі.
    assert reachable == 65, reachable


# --- критерії -------------------------------------------------------------
def test_model_carries_criteria_with_weights_and_polarity():
    model = _model001()
    assert model.criteria
    assert abs(sum(c.weight for c in model.criteria) - 1.0) < 1e-9
    assert {c.direction for c in model.criteria} <= {"benefit", "cost"}


# --- обмеження задачі: <task><constraints> ---------------------------------
# ЩО ЦЕ ЗАКРИВАЄ: `parse_network` уже читає `<constraints>` як список сирих
# українських рядків (`task["constraints"]`), але `build_twin_model` досі
# ігнорував їх узагалі — `constraints=[]` завжди, через що перевірка
# обмежень у `twin_defense`/`qa.twin.enumerate_scenarios` працювала
# вхолосту. Частоти РЕАЛЬНИХ формулювань зі 100 файлів (перевірено `grep`
# перед написанням цього коду, а не вгадано):
#   46 без зупинки виробничих сервісів
#   44 збереження доказової бази для розслідування
#   44 без залучення зовнішнього підрядника
#   40 час реагування не більше 4 годин
#   29 відповідність ISO/IEC 27035
# Жодних інших формулювань і жодної варіації числа/одиниці в межах кожного
# — усі 40 випадків "час реагування" дослівно "4 годин".

def test_parse_network_reads_task_constraints_as_a_raw_text_list():
    """`parse_network` (уже реалізовано) читає `<constraints>` як список
    сирих рядків, у порядку файлу — фіксуємо це контрактом, а не лише
    побічно через `build_twin_model` нижче."""
    network = _net001()
    assert network["task"]["constraints"] == [
        "відповідність ISO/IEC 27035",
        "час реагування не більше 4 годин",
        "збереження доказової бази для розслідування",
    ]


def test_classify_task_constraints_maps_iso27035_to_a_regulatory_flag():
    """«відповідність ISO/IEC 27035» — числа немає, але вимога явно
    регуляторна за змістом (стандарт), тож природно лягає на прапорець
    `regulatory` (`qa.twin.ALLOWED_CONSTRAINTS`)."""
    from experiments.corporate_networks.dataset import classify_task_constraints
    constraints, leftover = classify_task_constraints(["відповідність ISO/IEC 27035"])
    assert constraints == [TwinConstraint(criterion="regulatory", limit=1.0)]
    assert leftover == []


def test_classify_task_constraints_leaves_texts_without_a_matching_unit_as_text():
    """«час реагування не більше 4 годин» МАЄ явне число й одиницю, але
    одиниця (години) не має відповідника серед `qa.twin.ALLOWED_
    CONSTRAINTS` (там лише "months" — місяці) — підміняти одиниці означало б
    вигадати курс перерахунку. «збереження доказової бази...» взагалі не
    описує жодної з чотирьох категорій обмежень. Обидва лишаються текстом,
    а не вигаданим числом."""
    from experiments.corporate_networks.dataset import classify_task_constraints
    texts = [
        "час реагування не більше 4 годин",
        "збереження доказової бази для розслідування",
    ]
    constraints, leftover = classify_task_constraints(texts)
    assert constraints == []
    assert leftover == texts


def test_classify_task_constraints_without_a_number_and_without_regulatory_meaning_stays_text():
    """«без зупинки виробничих сервісів» / «без залучення зовнішнього
    підрядника» — заборони-виключення без числа, не regulatory-вимога —
    лишаються текстом."""
    from experiments.corporate_networks.dataset import classify_task_constraints
    texts = ["без зупинки виробничих сервісів", "без залучення зовнішнього підрядника"]
    constraints, leftover = classify_task_constraints(texts)
    assert constraints == []
    assert leftover == texts


def test_classify_task_constraints_on_net_001_mixed_list():
    """Реальний змішаний список net-001: одне зводиться, два — ні."""
    from experiments.corporate_networks.dataset import classify_task_constraints
    network = _net001()
    constraints, leftover = classify_task_constraints(network["task"]["constraints"])
    assert constraints == [TwinConstraint(criterion="regulatory", limit=1.0)]
    assert leftover == [
        "час реагування не більше 4 годин",
        "збереження доказової бази для розслідування",
    ]


def test_classify_task_constraints_ignores_blank_entries():
    from experiments.corporate_networks.dataset import classify_task_constraints
    constraints, leftover = classify_task_constraints(["", "   "])
    assert constraints == []
    assert leftover == []


# --- обмеження в TwinModel: build_twin_model --------------------------------

def test_build_twin_model_sets_a_regulatory_constraint_for_net_001():
    """net-001 має «відповідність ISO/IEC 27035» серед вимог задачі — модель
    більше не повинна віддавати порожній `constraints`."""
    model = _model001()
    assert model.constraints == [TwinConstraint(criterion="regulatory", limit=1.0)]


def test_build_twin_model_creates_requirement_nodes_for_text_only_constraints():
    """Вимоги, які НЕ звелися до числа (час реагування, доказова база),
    не зникають з моделі — стають вузлами kind="requirement" з текстом
    у атрибутах, а не мовчки губляться."""
    model = _model001()
    requirement_nodes = [n for n in model.nodes if n.kind == "requirement"]
    texts = {n.attributes.get("text") for n in requirement_nodes}
    assert texts == {
        "час реагування не більше 4 годин",
        "збереження доказової бази для розслідування",
    }
    # ISO 27035 звівся до числового обмеження — вузлом requirement НЕ
    # дублюється (інакше вимога рахувалась би двічі).
    assert "відповідність ISO/IEC 27035" not in texts


def test_build_twin_model_net_006_has_no_numeric_constraint_only_requirement_text():
    """net-006: жодного з трьох формулювань цієї мережі не зводиться до
    числа (немає ні ISO 27035, ні "час реагування") — `constraints`
    порожній, а всі три лишаються текстовими вузлами вимог."""
    model = build_twin_model(parse_network(_read("net-006")))
    assert model.constraints == []
    requirement_texts = {n.attributes.get("text") for n in model.nodes if n.kind == "requirement"}
    assert requirement_texts == {
        "без зупинки виробничих сервісів",
        "збереження доказової бази для розслідування",
        "без залучення зовнішнього підрядника",
    }


def test_requirement_nodes_have_unique_ids_and_no_edges():
    """Вузли вимог — чиста метадана: жодне ребро моделі їх не торкається
    (вони не частина шляху атаки чи захисту), і кожен має власний id."""
    model = build_twin_model(parse_network(_read("net-006")))
    requirement_ids = [n.id for n in model.nodes if n.kind == "requirement"]
    assert len(requirement_ids) == len(set(requirement_ids)) == 3
    touched = {e.source for e in model.edges} | {e.target for e in model.edges}
    assert not (set(requirement_ids) & touched)


def test_regulatory_constraint_count_across_the_dataset_matches_iso27035_frequency():
    """Скільки зі 100 мереж отримують числовий `regulatory`-прапорець —
    має точно збігатися з кількістю мереж, де СЛОВО в слово трапляється
    «відповідність ISO/IEC 27035» (виміряно `grep` по датасету: 29)."""
    xml_paths = sorted(glob.glob(os.path.join(_DATASET_DIR, "net-*.xml")))
    assert len(xml_paths) == 100
    with_regulatory = 0
    for xml_path in xml_paths:
        with open(xml_path, "r", encoding="utf-8") as f:
            model = build_twin_model(parse_network(f.read()))
        if any(c.criterion == "regulatory" for c in model.constraints):
            with_regulatory += 1
    assert with_regulatory == 29, with_regulatory


def test_model_constraints_across_the_dataset_are_only_ever_the_regulatory_flag():
    """Захист від надмірного зведення: жодна мережа не повинна отримати
    вигаданий budget/months/personnel — джерело даних жодного з них не
    підтверджує (обґрунтування — докстрінг `classify_task_constraints`)."""
    xml_paths = sorted(glob.glob(os.path.join(_DATASET_DIR, "net-*.xml")))
    for xml_path in xml_paths:
        with open(xml_path, "r", encoding="utf-8") as f:
            model = build_twin_model(parse_network(f.read()))
        assert {c.criterion for c in model.constraints} <= {"regulatory"}


# --- еталон: answers.json ---------------------------------------------
def test_load_answers_returns_the_expected_fields_for_net_001():
    answers_path = os.path.join(_DATASET_DIR, "answers.json")
    answers = load_answers(answers_path)
    record = answers["net-001"]
    assert record["best_attack"] == "a2"
    assert record["margin"] == 0.6461
    assert record["effectiveness"]["a2"] == 0.379456
    assert record["lengths"]["a2"] == 3


def test_reproduced_effectiveness_matches_answers_json_for_net_001():
    """`answers.json["scoring"] == "product of per-step CVSS exploitability"`
    — відтворюємо це число з ПУБЛІЧНИХ даних активів (не з еталона напряму)
    і звіряємо з заявленим значенням."""
    xml_text = _read("net-001")
    network = parse_network(xml_text)
    gold = parse_gold_attacks(xml_text)
    cve_index = cve_exploitability_index(network)
    reproduced = reproduce_effectiveness(gold, cve_index)

    answers = load_answers(os.path.join(_DATASET_DIR, "answers.json"))
    expected = answers["net-001"]["effectiveness"]
    assert reproduced.keys() == expected.keys()
    for attack_id, value in expected.items():
        assert abs(reproduced[attack_id] - value) < 1e-6, (attack_id, reproduced[attack_id], value)


def test_reproduced_effectiveness_matches_answers_json_across_the_dataset():
    """Той самий тест, але по всіх 100 мережах — жодного відхилення понад
    допуск: якби формула не сходилась, ми зафіксували б розбіжність тут
    прямо, а не підганяли тест."""
    answers = load_answers(os.path.join(_DATASET_DIR, "answers.json"))
    checked = 0
    for xml_path in sorted(glob.glob(os.path.join(_DATASET_DIR, "net-*.xml"))):
        network_id = os.path.splitext(os.path.basename(xml_path))[0]
        with open(xml_path, "r", encoding="utf-8") as f:
            xml_text = f.read()
        network = parse_network(xml_text)
        gold = parse_gold_attacks(xml_text)
        cve_index = cve_exploitability_index(network)
        reproduced = reproduce_effectiveness(gold, cve_index)
        expected = answers[network_id]["effectiveness"]
        for attack_id, value in expected.items():
            assert abs(reproduced[attack_id] - value) < 1e-6, (network_id, attack_id)
            checked += 1
    assert checked == 500, checked  # 100 мереж x 5 атак кожна


# --- всі 100 мереж парсяться без винятків ---------------------------------
def test_all_100_networks_parse_and_build_a_model_without_error():
    xml_paths = sorted(glob.glob(os.path.join(_DATASET_DIR, "net-*.xml")))
    assert len(xml_paths) == 100, len(xml_paths)
    node_counts = []
    edge_counts = []
    for xml_path in xml_paths:
        with open(xml_path, "r", encoding="utf-8") as f:
            xml_text = f.read()
        network = parse_network(xml_text)
        model = build_twin_model(network)
        assert model.current_state_id != model.goal_state_id
        assert model.node_by_id[model.current_state_id].kind == "state"
        assert model.node_by_id[model.goal_state_id].kind == "state"
        node_counts.append(len(model.nodes))
        edge_counts.append(len(model.edges))
    assert len(node_counts) == 100
    print(f"\nсередньо вузлів: {sum(node_counts) / 100:.1f}, "
          f"середньо ребер: {sum(edge_counts) / 100:.1f}, "
          f"max вузлів: {max(node_counts)}, max ребер: {max(edge_counts)}")


# --- відтворюваність побудови ---------------------------------------------

def test_transition_edges_are_built_in_a_deterministic_order():
    """Порядок ребер не має залежати від PYTHONHASHSEED.

    НАВІЩО. `_asset_reachability` і реєстр станів усередині
    `build_twin_model` збираються через `set`, а порядок обходу множини
    рядків у CPython залежить від хеш-сіда процесу. Сам НАБІР ребер від
    цього не змінюється, але змінюється їхня ПОСЛІДОВНІСТЬ — а
    `qa.twin.enumerate_scenarios` обриває пошук у глибину на межі
    кандидатів, тому з різним порядком у вибірку потрапляють різні шляхи.
    Виміряно на 100 мережах: середнє покриття гуляло 0,2900 / 0,3047 /
    0,3247 / 0,3142 для сідів 0 / 1 / 42 / 777. Робота заявляє
    відтворюваність, тож порядок треба закріпити.

    Перевіряємо канонічний порядок, а не рівність двох прогонів у одному
    процесі: у межах одного процесу хеш-сід сталий, і такий тест був би
    зеленим завжди."""
    model = build_twin_model(_net001())
    exploits = [(e.source, e.target) for e in model.edges if e.relation == "exploits"]
    assert exploits, "у моделі немає переходів `exploits`"
    assert exploits == sorted(exploits), (
        "ребра `exploits` йдуть не в канонічному порядку — побудова "
        "залежить від порядку обходу множин"
    )


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
