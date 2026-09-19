"""
Міст: модель машини HTB -> типізований ОРІЄНТОВАНИЙ граф (`qa/twin.py`).

НАВІЩО. У `semantic_reasoner` граф неорієнтований, а ребро означає просту
співзустрічальність понять у документі. Через це «OpenSSL» і `root` стають
прямими сусідами, обхід проскакує проміжні кроки, а стрілка на схемі не
означає переходу. Твердження статті про орієнтовану семантичну мережу й
причинність спираються на модель, якої в коді не було.

Тут напрям і тип ребра беруться з ІМЕНОВАНИХ ПОЛІВ вихідної моделі
(`parent_id`, `runs_as`, `affected_component`, `resulting_access`,
`permission_edges.subject/object`), а не з LLM і не зі статистики. Після
історії з вигаданим T1190 це принципово: жодне ребро не існує, доки його не
назвало поле даних.

ГОЛОВНИЙ ПРЕДОХРАНИТЕЛЬ — `test_gold_never_leaks_into_the_model`: `attack_path`
це еталон, і він не має впливати на граф НІЯК. Якщо його вилучення щось
змінює — порівняння з еталоном нічого не варте.

Запуск:
    python tests/test_twin_bridge.py
або pytest.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa.twin import ALLOWED_KINDS, ALLOWED_STATE_TYPES  # noqa: E402
from twin_bridge import (  # noqa: E402
    EXTERNAL_ID,
    build_twin_model,
    path_concepts,
)

# Мінімальна, але повна машина: усі поля, з яких міст будує ребра.
MACHINE = {
    "machine": "Testbox",
    "host": {"operating_system_or_build": "Windows Server 2003",
             "deployment_role": "standalone"},
    "components": [
        {"component_id": "H1", "parent_id": "-", "layer_or_type": "Host",
         "name": "Testbox server", "role": "OS boundary",
         "binding_or_path": "10.0.0.1", "runs_as": "SYSTEM",
         "exposure": "VPN reachable"},
        {"component_id": "C2", "parent_id": "H1", "layer_or_type": "Service",
         "name": "IIS WebDAV", "role": "web front end",
         "binding_or_path": "0.0.0.0:80", "runs_as": "NETWORK SERVICE",
         "exposure": "Public"},
        {"component_id": "C3", "parent_id": "H1", "layer_or_type": "Service",
         "name": "internal scheduler", "role": "task runner",
         "runs_as": "SYSTEM", "exposure": "localhost only"},
    ],
    "identities": [
        {"principal": "NETWORK SERVICE", "principal_type": "service identity",
         "operational_role": "IIS worker"},
        {"principal": "SYSTEM", "principal_type": "machine account",
         "operational_role": "full host rights"},
    ],
    "permission_edges": [
        {"subject": "NETWORK SERVICE", "permission_or_action": "write files",
         "object": "A1", "condition_or_scope": "writable directory",
         "security_consequence": "SYSTEM execution"},
    ],
    "artifacts": [
        {"artifact_or_path": "C:\\\\wmpub", "hosted_on": "H1",
         "purpose": "writable staging directory",
         "write_or_execute_access": "NETWORK SERVICE write"},
    ],
    "vulnerabilities": [
        {"finding_id": "V1", "affected_component": "C2",
         "issue": "ScStoragePathFromUrl buffer overflow",
         "preconditions": "WebDAV reachable",
         "resulting_access": "NETWORK SERVICE",
         "reference": "CVE-2017-7269"},
    ],
    # Еталон. На граф впливати НЕ МАЄ.
    "attack_path": [
        {"step": 1, "source_principal": "unauthenticated",
         "action": "send WebDAV overflow", "target_component_or_artifact": "C2",
         "resulting_principal_or_access": "NETWORK SERVICE"},
        {"step": 2, "source_principal": "NETWORK SERVICE",
         "action": "abuse token privilege", "target_component_or_artifact": "H1",
         "resulting_principal_or_access": "SYSTEM"},
    ],
}


def _model(machine=None, goal="system"):
    # Ціль тепер приходить ЗЗОВНІ: модуль не має доступу до `attack_path`.
    return build_twin_model(machine or MACHINE, goal)


def _relations(model):
    return {edge.relation for edge in model.edges}


def _edge(model, relation):
    return [e for e in model.edges if e.relation == relation]


# --- ПРЕДОХРАНИТЕЛЬ: еталон не тече в граф --------------------------------
def test_gold_never_leaks_into_the_model():
    """Вилучення `attack_path` не має змінити ані вузол, ані ребро."""
    without = copy.deepcopy(MACHINE)
    without.pop("attack_path")
    full, blind = _model(), _model(without)
    assert [(n.id, n.kind, n.label) for n in full.nodes] == \
           [(n.id, n.kind, n.label) for n in blind.nodes]
    assert [(e.source, e.relation, e.target) for e in full.edges] == \
           [(e.source, e.relation, e.target) for e in blind.edges]


def test_action_wording_of_the_gold_is_absent_from_labels():
    labels = " ".join(n.label for n in _model().nodes).lower()
    assert "abuse token privilege" not in labels
    assert "send webdav overflow" not in labels


# --- граф справді орієнтований і типізований ------------------------------
def test_every_node_kind_is_one_twin_understands():
    for node in _model().nodes:
        assert node.kind in ALLOWED_KINDS, (node.id, node.kind)


def test_state_types_are_from_the_allowed_set():
    for node in _model().nodes:
        if node.kind == "state":
            st = node.attributes.get("state_type")
            assert st in ALLOWED_STATE_TYPES, (node.id, st)


def test_edges_are_named_not_anonymous():
    model = _model()
    assert model.edges
    assert all(e.relation for e in model.edges)


def test_edges_are_directed_not_symmetric():
    """Орієнтованість перевіряємо по суті: зворотного ребра бути не повинно."""
    model = _model()
    pairs = {(e.source, e.target) for e in model.edges}
    mirrored = {(t, s) for s, t in pairs} & pairs
    assert not mirrored, f"знайдено симетричні пари: {sorted(mirrored)}"


# --- напрям береться з полів моделі ---------------------------------------
def test_parent_id_becomes_a_containment_edge():
    assert ("C2", "H1") in {(e.source, e.target) for e in _edge(_model(), "hosted_on")}


def test_runs_as_links_component_to_its_principal():
    model = _model()
    edges = {(e.source, e.target) for e in _edge(model, "runs_as")}
    by_id = model.node_by_id
    assert any(s == "C2" and "network service" in by_id[t].label.lower()
               for s, t in edges), edges


def test_vulnerability_hangs_on_the_component_it_affects():
    assert ("C2", "V1") in {(e.source, e.target)
                            for e in _edge(_model(), "has_vulnerability")}


def test_vulnerability_grants_the_access_it_declares():
    model = _model()
    granted = [model.node_by_id[e.target] for e in _edge(model, "grants")
               if e.source == "V1"]
    assert granted, "resulting_access не перетворився на стан"
    assert all(n.kind == "state" for n in granted)


def test_permission_edge_keeps_subject_and_object_direction():
    model = _model()
    acts = _edge(model, "write files")
    assert acts, _relations(model)
    src = model.node_by_id[acts[0].source]
    assert src.kind == "state" and "network service" in src.label.lower()


# --- периметр -------------------------------------------------------------
def test_external_state_reaches_only_exposed_components():
    model = _model()
    reached = {e.target for e in _edge(model, "reaches")}
    assert "C2" in reached, "публічний компонент має бути досяжним"
    assert "C3" not in reached, "localhost-only компонент назовні не світиться"


def test_external_state_is_the_current_state():
    model = _model()
    assert model.current_state_id == EXTERNAL_ID
    assert model.node_by_id[EXTERNAL_ID].kind == "state"


def test_external_state_carries_no_concepts_of_its_own():
    """Синтетичний вузол периметра не факт із моделі, тож у метрику не лізе."""
    assert not _model().node_by_id[EXTERNAL_ID].attributes.get("concepts")


def test_model_has_no_goal_when_none_is_named():
    """Ціль не вигадується: без явного поняття цільового вузла немає."""
    assert build_twin_model(MACHINE).goal_state_id == ""


def test_goal_state_matches_the_declared_goal_concept():
    model = _model()
    goal = model.node_by_id[model.goal_state_id]
    assert goal.kind == "state"
    assert "system" in goal.attributes.get("concepts", [])


# --- стани не дробляться на уточненнях -----------------------------------
# Інакше граф рветься саме там, де важливо: «SYSTEM» із `identities` і
# «SYSTEM execution» із `security_consequence` — один і той самий принципал,
# а різними вузлами вони роблять ціль недосяжною.
def test_state_with_extra_words_merges_into_the_named_principal():
    model = _model()
    system_states = [n for n in model.nodes
                     if n.kind == "state" and "system" in n.attributes.get("concepts", [])]
    assert len(system_states) == 1, [n.label for n in system_states]


def test_service_principal_merges_across_fields():
    model = _model()
    ns = [n for n in model.nodes if n.kind == "state"
          and {"network", "service"} <= set(n.attributes.get("concepts", []))]
    assert len(ns) == 1, [n.label for n in ns]


def test_distinct_principals_stay_separate():
    model = _model()
    labels = {n.label.lower() for n in model.nodes if n.kind == "state"}
    assert any("system" in l for l in labels)
    assert any("network service" in l for l in labels)
    states = [n for n in model.nodes if n.kind == "state" and n.id != EXTERNAL_ID]
    assert len(states) >= 2, labels


def test_merged_state_keeps_the_shortest_name_as_canonical():
    model = _model()
    node = next(n for n in model.nodes
                if n.kind == "state" and "system" in n.attributes.get("concepts", []))
    assert node.attributes["concepts"] == ["system"], node.attributes["concepts"]


# --- не кожне ребро є ПЕРЕХОДОМ -------------------------------------------
# Напрям, узятий з поля даних, ще не робить ребро кроком атаки. `hosted_on`
# каже «сервіс лежить на хості», а не «зламавши сервіс, ти отримав хост».
# Без цієї різниці обхід їде вгору по вкладеності й «здобуває» SYSTEM просто
# тому, що хост під ним працює, — та сама вигадка, лише в орієнтованому одязі.
# Типізація тим і корисна, що дає це побачити й відсікти явно.
def test_structural_relations_are_not_marked_as_transitions():
    model = _model()
    for edge in model.edges:
        if edge.relation in ("hosted_on", "stored_on"):
            assert edge.attributes.get("transition") is False, edge.relation


def test_attack_relations_are_marked_as_transitions():
    model = _model()
    marked = {e.relation for e in model.edges if e.attributes.get("transition")}
    assert {"reaches", "has_vulnerability", "grants", "runs_as"} <= marked, marked


def test_every_edge_declares_whether_it_is_a_transition():
    for edge in _model().edges:
        assert "transition" in edge.attributes, edge.relation


def test_named_permission_action_counts_as_a_transition():
    model = _model()
    acts = _edge(model, "write files")
    assert acts and acts[0].attributes.get("transition") is True


def test_transition_edges_are_exposed_for_traversal():
    from twin_bridge import transition_edge_ids
    model = _model()
    ids = transition_edge_ids(model)
    assert ids
    assert all(model.edges[i].attributes.get("transition") for i in ids)
    assert not any(model.edges[i].relation == "hosted_on" for i in ids)


def test_runs_as_from_a_bare_host_is_not_a_transition():
    """Дійти до хоста мережею — не те саме, що виконати на ньому код.

    Хост потрапляє в `components` з власним `exposure`, тож без цього правила
    обхід робить «external -> Bashed VM -> root» і зараховує собі перемогу, не
    назвавши жодної дії. Здобути ідентичність хоста можна лише через
    вразливість або названу дію, тобто іншим ребром."""
    model = _model()
    host_runs = [e for e in _edge(model, "runs_as")
                 if model.node_by_id[e.source].kind == "asset"]
    assert host_runs, "фікстура має хост із runs_as, інакше тест нічого не ловить"
    assert all(e.attributes.get("transition") is False for e in host_runs)


def test_runs_as_from_a_service_stays_a_transition():
    model = _model()
    svc_runs = [e for e in _edge(model, "runs_as")
                if model.node_by_id[e.source].kind == "service"]
    assert svc_runs
    assert all(e.attributes.get("transition") is True for e in svc_runs)


# --- критерії та обмеження для J ------------------------------------------
def test_model_carries_criteria_with_weights_and_polarity():
    model = _model()
    assert model.criteria
    assert abs(sum(c.weight for c in model.criteria) - 1.0) < 1e-9
    assert {c.direction for c in model.criteria} <= {"benefit", "cost"}


def test_domain_prefix_does_not_turn_into_a_magnet_token():
    """`ACTIVE\SVC_TGS` і `ACTIVE\Administrator` — РІЗНІ принципали.

    Без зняття доменного префікса від першого лишається сам лише токен
    `active` (svc і tgs коротші за поріг), він виявляється підмножиною
    {active, administrator} — і два різні акаунти зливаються в один вузол.
    На машині Active це проміжна ціль GPP-атаки та кінцева ціль."""
    machine = {
        "machine": "Active",
        "components": [{"component_id": "C1", "name": "SMB share",
                        "layer_or_type": "Service", "exposure": "Public",
                        "runs_as": "ACTIVE\SVC_TGS"}],
        "identities": [{"principal": "ACTIVE\SVC_TGS"},
                       {"principal": "ACTIVE\Administrator"}],
        "vulnerabilities": [{"finding_id": "V1", "affected_component": "C1",
                             "issue": "GPP cpassword",
                             "resulting_access": "ACTIVE\Administrator"}],
    }
    model = build_twin_model(machine, "administrator")
    concepts = [n.attributes.get("concepts", []) for n in model.nodes
                if n.kind == "state" and n.id != EXTERNAL_ID]
    # Кінцева ціль має бути представлена САМА, а не розчинена в доменному
    # префіксі, спільному з іншим акаунтом.
    assert ["administrator"] in concepts, concepts


def test_slash_in_a_component_name_is_not_treated_as_a_domain_prefix():
    """Зняття префікса стосується ЛИШЕ імен принципалів.

    Назва компонента цілком може містити «/» («browse.php / listfiles.php»), а
    шлях артефакта — «\» («C:\wmpub»). Якщо різати їх тим самим правилом,
    половина назви зникає, поняття не збігаються з еталоном, і ланцюг рветься —
    саме так у Poison розпався шлях через LFI."""
    machine = {
        "machine": "Poison",
        "components": [{"component_id": "C1", "layer_or_type": "Service",
                        "name": "browse.php / listfiles.php", "exposure": "Public"}],
        "artifacts": [{"artifact_or_path": "C:\wmpub", "hosted_on": "C1"}],
    }
    model = build_twin_model(machine, "root")
    by_id = model.node_by_id
    assert "browse" in by_id["C1"].attributes["concepts"], by_id["C1"].attributes
    assert "listfiles" in by_id["C1"].attributes["concepts"]
    assert "wmpub" in by_id["A1"].attributes["concepts"], by_id["A1"].attributes


def test_forward_slash_in_a_principal_is_not_a_domain_separator():
    """`charix/attacker` — це «charix, тобто атакувальник», а не «домен/логін».

    Розділювач домену у Windows-принципалі — ЗВОРОТНИЙ слеш. Якщо різати й по
    прямому, від `charix/attacker` лишається `attacker`, він перестає зводитись
    із `charix`, і ланцюг Poison через LFI розпадається — при тому, що всі
    юніт-тести лишаються зеленими."""
    machine = {
        "machine": "Poison",
        "components": [{"component_id": "C1", "layer_or_type": "Service",
                        "name": "vnc", "exposure": "Public"}],
        "vulnerabilities": [{"finding_id": "V1", "affected_component": "C1",
                             "issue": "lfi", "resulting_access": "charix"}],
        "permission_edges": [{"subject": "charix/attacker",
                              "permission_or_action": "authenticate",
                              "object": "C1", "security_consequence": "root"}],
    }
    model = build_twin_model(machine, "root")
    charix = [n for n in model.nodes if n.kind == "state"
              and "charix" in n.attributes.get("concepts", [])]
    assert len(charix) == 1, [n.attributes.get("concepts") for n in charix]


def test_domain_prefix_is_stripped_from_principal_names():
    model = build_twin_model(
        {"machine": "Active", "identities": [{"principal": "ACTIVE\Administrator"}]},
        "administrator")
    node = next(n for n in model.nodes if n.kind == "state" and n.id != EXTERNAL_ID)
    assert node.attributes["concepts"] == ["administrator"], node.attributes


# --- ранжування сценаріїв через J (формули 12-13) -------------------------
# `enumerate_scenarios` сортує кандидатів за `_treatment_count` — кількістю
# вузлів kind=control, які лікують вразливості. Це критерій із ІНШОГО
# застосування модуля (планування усунення), і в нашому графі control-вузлів
# немає взагалі, тож він завжди нуль, а сортування вироджується в алфавітне
# за id вузлів. На Poison це ставило першим «external -> OpenSSH -> root»
# (покриття 0,5), тоді як справжній ланцюг із покриттям 1,0 стояв другим.
# Ранжуємо власною інтегральною оцінкою — тими самими критеріями, що вже
# оголошені в DEFAULT_CRITERIA, і ЖОДЕН із них не бачить еталона.

def test_scenario_scores_cover_every_declared_criterion():
    from twin_bridge import (
        DEFAULT_CRITERIA, score_scenario)
    model = _model()
    scores = score_scenario(model, [EXTERNAL_ID, "C2", "V1"], [0, 1])
    assert {s["criterion"] for s in scores} == {c.name for c in DEFAULT_CRITERIA}


def test_named_mechanism_scores_higher_feasibility_than_bare_reachability():
    """Крок, що називає механізм (експлойт, дія), здійсненніший за твердження
    «сервіс просто працює під root»."""
    from twin_bridge import score_scenario
    model = _model()
    named = next(i for i, e in enumerate(model.edges) if e.relation == "has_vulnerability")
    generic = next(i for i, e in enumerate(model.edges) if e.relation == "reaches")
    f_named = next(s["score"] for s in score_scenario(model, ["C2", "V1"], [named])
                   if s["criterion"] == "feasibility")
    f_generic = next(s["score"] for s in score_scenario(model, [EXTERNAL_ID, "C2"], [generic])
                     if s["criterion"] == "feasibility")
    assert f_named > f_generic


def test_effort_grows_with_path_length():
    from twin_bridge import score_scenario
    model = _model()
    short = next(s["score"] for s in score_scenario(model, ["C2", "V1"], [0])
                 if s["criterion"] == "effort")
    long_ = next(s["score"] for s in score_scenario(model, ["C2", "V1", "S1", "A1"], [0, 1, 2])
                 if s["criterion"] == "effort")
    assert long_ > short


def test_scenario_scores_never_look_at_the_gold():
    """Критерії рахуються лише зі структури шляху: жоден не приймає еталона."""
    import inspect
    from twin_bridge import score_scenario
    params = set(inspect.signature(score_scenario).parameters)
    assert params == {"model", "path", "edge_path"}, params


def test_ranking_puts_the_higher_j_first():
    from twin_bridge import rank_scenarios
    model = _model()
    weak = {"path": [EXTERNAL_ID, "H1"], "edge_path": [], "labels": []}
    strong = {"path": [EXTERNAL_ID, "C2", "V1"], "edge_path": [], "labels": []}
    ranked = rank_scenarios(model, [weak, strong])
    assert ranked[0]["path"] == strong["path"], [r["path"] for r in ranked]
    assert ranked[0]["j"] >= ranked[1]["j"]


def test_ranking_drops_duplicate_node_paths():
    from twin_bridge import rank_scenarios
    model = _model()
    one = {"path": [EXTERNAL_ID, "C2"], "edge_path": [], "labels": []}
    ranked = rank_scenarios(model, [one, dict(one)])
    assert len(ranked) == 1


def test_ranking_is_deterministic_for_equal_scores():
    from twin_bridge import rank_scenarios
    model = _model()
    a = {"path": [EXTERNAL_ID, "C2"], "edge_path": [], "labels": []}
    b = {"path": [EXTERNAL_ID, "C3"], "edge_path": [], "labels": []}
    assert [r["path"] for r in rank_scenarios(model, [a, b])] ==            [r["path"] for r in rank_scenarios(model, [b, a])]


# --- зіставність із метрикою ----------------------------------------------
def test_path_concepts_flattens_nodes_into_metric_tokens():
    model = _model()
    chain = path_concepts(model, [EXTERNAL_ID, "C2", "V1"])
    assert isinstance(chain, list)
    assert all(isinstance(token, str) for token in chain)
    assert "webdav" in chain or "iis" in chain


def test_path_concepts_preserves_node_order():
    model = _model()
    chain = path_concepts(model, ["C2", "C3"])
    assert chain.index("webdav") < chain.index("scheduler")


def test_path_concepts_does_not_repeat_a_token():
    model = _model()
    chain = path_concepts(model, ["C2", "C2"])
    assert len(chain) == len(set(chain))


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
