"""Гарантії методу зі статті над моделлю двійника.

Покриває властивості, які стаття заявляє для детермінованих стадій: типізовані
переходи станів, кон'юнктивні передумови, набір обмежень R, свідчення на рівні
ребер, полярність критеріїв та обмежене лагодження.
"""
from pathlib import Path

import pytest

from qa import twin

FIXTURE = Path(__file__).resolve().parents[1] / "qa" / "fixtures" / "corporate_12node.yaml"


def build(nodes: str, edges: str, extra: str = "") -> twin.TwinModel:
    document = f"""
name: unit
current_state: s0
goal_state: st
nodes:
{nodes}
edges:
{edges}
criteria:
  - {{name: risk, weight: 1.0, description: "residual risk"}}
{extra}
"""
    return twin.parse_twin_model(document)


def paths_of(model: twin.TwinModel, max_paths: int = 10, **kwargs) -> dict:
    forward = twin.expand_forward(model)
    backward = twin.expand_backward(model)
    gc = twin.reconcile(model, forward, backward)
    return twin.enumerate_scenarios(model, gc, set(), max_paths, **kwargs)


# --- typed states -----------------------------------------------------------


def test_access_state_requires_a_named_action():
    with pytest.raises(twin.TwinModelError, match="without naming an action"):
        build(
            nodes="""
  - {id: s0, kind: state, label: Start, state_type: known_identity}
  - {id: st, kind: state, label: Goal, state_type: execution_context}
""",
            edges="""
  - {source: s0, target: st, relation: enables}
""",
        )


def test_named_action_admits_the_access_transition():
    model = build(
        nodes="""
  - {id: s0, kind: state, label: Start, state_type: known_identity}
  - {id: st, kind: state, label: Goal, state_type: execution_context}
""",
        edges="""
  - {source: s0, target: st, relation: enables, action: "smb exec"}
""",
    )
    assert model.state_type("s0") == "known_identity"
    assert paths_of(model)["total_found"] == 1


def test_security_state_needs_no_action():
    model = build(
        nodes="""
  - {id: s0, kind: state, label: Start, state_type: security_state}
  - {id: st, kind: state, label: Goal, state_type: security_state}
""",
        edges="""
  - {source: s0, target: st, relation: hardens}
""",
    )
    assert model.state_type("st") == "security_state"


def test_unknown_state_type_is_rejected():
    with pytest.raises(twin.TwinModelError, match="unsupported state_type"):
        build(
            nodes="""
  - {id: s0, kind: state, label: Start, state_type: root_shell}
  - {id: st, kind: state, label: Goal}
""",
            edges="""
  - {source: s0, target: st, relation: enables}
""",
        )


def test_state_type_rejected_on_a_non_state_node():
    with pytest.raises(twin.TwinModelError, match="state_type applies to state nodes"):
        build(
            nodes="""
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
  - {id: a1, kind: asset, label: Host, state_type: privilege}
""",
            edges="""
  - {source: s0, target: st, relation: enables}
""",
        )


def test_untyped_states_keep_the_earlier_behaviour():
    model = build(
        nodes="""
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
""",
        edges="""
  - {source: s0, target: st, relation: enables}
""",
    )
    assert paths_of(model)["total_found"] == 1
    assert model.state_type("s0") == ""


# --- Гейти AND/OR -----------------------------------------------------------


AND_GATE_NODES = """
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
  - {id: cred, kind: asset, label: Credential}
  - {id: host, kind: asset, label: Reachable Host}
  - {id: both, kind: gate, label: Credential And Host, gate: and}
"""


def test_and_gate_rejects_a_path_missing_one_precondition():
    model = build(
        nodes=AND_GATE_NODES,
        edges="""
  - {source: s0, target: cred, relation: yields}
  - {source: s0, target: host, relation: reaches}
  - {source: cred, target: both, relation: satisfies}
  - {source: host, target: both, relation: satisfies}
  - {source: both, target: st, relation: enables}
""",
    )
    result = paths_of(model)
    assert result["and_gates"] == ["both"]
    assert result["total_found"] == 0
    assert result["rejected_by_gate"] == 2


def test_and_gate_admits_a_path_that_collects_both():
    model = build(
        nodes=AND_GATE_NODES,
        edges="""
  - {source: s0, target: cred, relation: yields}
  - {source: cred, target: host, relation: reaches}
  - {source: cred, target: both, relation: satisfies}
  - {source: host, target: both, relation: satisfies}
  - {source: both, target: st, relation: enables}
""",
    )
    result = paths_of(model)
    assert result["total_found"] == 1
    assert result["scenarios"][0]["path"] == ["s0", "cred", "host", "both", "st"]


def test_or_gate_needs_only_one_predecessor():
    model = build(
        nodes="""
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
  - {id: cred, kind: asset, label: Credential}
  - {id: host, kind: asset, label: Reachable Host}
  - {id: either, kind: gate, label: Either, gate: or}
""",
        edges="""
  - {source: s0, target: cred, relation: yields}
  - {source: s0, target: host, relation: reaches}
  - {source: cred, target: either, relation: satisfies}
  - {source: host, target: either, relation: satisfies}
  - {source: either, target: st, relation: enables}
""",
    )
    result = paths_of(model)
    assert result["total_found"] == 2
    assert result["rejected_by_gate"] == 0


def test_gate_node_needs_a_declared_gate_kind():
    with pytest.raises(twin.TwinModelError, match="needs gate"):
        build(
            nodes="""
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
  - {id: g1, kind: gate, label: Gate}
""",
            edges="""
  - {source: s0, target: st, relation: enables}
""",
        )


def test_gate_attribute_rejected_on_a_non_gate_node():
    with pytest.raises(twin.TwinModelError, match="gate applies to gate nodes"):
        build(
            nodes="""
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
  - {id: a1, kind: asset, label: Host, gate: and}
""",
            edges="""
  - {source: s0, target: st, relation: enables}
""",
        )


# --- constraint set R -------------------------------------------------------


CONSTRAINT_NODES = """
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
  - {id: cheap, kind: control, label: Cheap Control, cost: 10, attributes: {regulatory_approved: true}}
  - {id: dear, kind: control, label: Expensive Control, cost: 900, attributes: {regulatory_approved: true}}
"""

CONSTRAINT_EDGES = """
  - {source: s0, target: cheap, relation: applies}
  - {source: s0, target: dear, relation: applies}
  - {source: cheap, target: st, relation: reaches}
  - {source: dear, target: st, relation: reaches}
"""


def test_budget_constraint_removes_the_expensive_path():
    model = build(
        nodes=CONSTRAINT_NODES,
        edges=CONSTRAINT_EDGES,
        extra="""
constraints:
  - {criterion: budget, limit: 100}
""",
    )
    result = paths_of(model)
    assert result["total_found"] == 1
    assert result["rejected_by_constraint"] == 1
    assert result["constraints_applied"] == [{"criterion": "budget", "limit": 100.0}]


def test_a_run_without_constraints_records_an_empty_set():
    model = build(nodes=CONSTRAINT_NODES, edges=CONSTRAINT_EDGES)
    result = paths_of(model)
    assert result["total_found"] == 2
    assert result["rejected_by_constraint"] == 0
    assert result["constraints_applied"] == []


def test_within_constraints_is_public():
    """`twin_defense.py` (планування наборів захисту) рахує той самий бюджет,
    що й перебір шляхів атаки тут - і робить це через ім'я з підкресленням
    (`_within_constraints`), позичене з приватної частини цього модуля. Для
    публічного репозиторію так не годиться: перевіряємо, що є публічний
    синонім з тією самою поведінкою, яким можна користуватись ЗВІДКИ завгодно."""
    model = build(
        nodes=CONSTRAINT_NODES,
        edges=CONSTRAINT_EDGES,
        extra="""
constraints:
  - {criterion: budget, limit: 100}
""",
    )
    # "cheap" (10) укладається в бюджет 100, "dear" (900) - ні.
    assert twin.within_constraints(model, ["s0", "cheap", "st"], [0, 2]) is True
    assert twin.within_constraints(model, ["s0", "dear", "st"], [1, 3]) is False


def test_regulatory_constraint_requires_approved_controls():
    model = build(
        nodes="""
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
  - {id: ok, kind: control, label: Approved, attributes: {regulatory_approved: true}}
  - {id: bad, kind: control, label: Unapproved}
""",
        edges="""
  - {source: s0, target: ok, relation: applies}
  - {source: s0, target: bad, relation: applies}
  - {source: ok, target: st, relation: reaches}
  - {source: bad, target: st, relation: reaches}
""",
        extra="""
constraints:
  - {criterion: regulatory, limit: 1}
""",
    )
    result = paths_of(model)
    assert result["total_found"] == 1
    assert result["scenarios"][0]["path"] == ["s0", "ok", "st"]


# --- edge-level evidence ----------------------------------------------------


EVIDENCE_MODEL_NODES = """
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
  - {id: tomcat, kind: service, label: "Tomcat Manager", evidence_terms: ["tomcat manager"]}
  - {id: rce, kind: vuln, label: "Struts RCE", evidence_terms: ["struts rce"]}
"""

EVIDENCE_MODEL_EDGES = """
  - {source: s0, target: tomcat, relation: reaches}
  - {source: tomcat, target: rce, relation: exposes}
  - {source: rce, target: st, relation: enables}
"""


def test_edge_evidence_needs_both_endpoints_in_one_document():
    model = build(nodes=EVIDENCE_MODEL_NODES, edges=EVIDENCE_MODEL_EDGES)
    documents = [
        {
            "mongo_id": "0" * 24,
            "content": "The tomcat manager console yields a struts rce to anonymous users.",
            "observed_at": "2026-09-05T00:00:00Z",
        }
    ]
    records = twin.verify_edges(model, documents, {"tomcat", "rce"})
    assert [record["edge_index"] for record in records] == [1]
    record = records[0]
    assert record["mongo_id"] == "0" * 24
    assert record["observed_at"] == "2026-09-05T00:00:00Z"
    assert "tomcat manager" in record["fragment"]
    assert record["confidence"] == 0.6


def test_edge_evidence_confidence_rises_when_the_relation_appears():
    model = build(nodes=EVIDENCE_MODEL_NODES, edges=EVIDENCE_MODEL_EDGES)
    documents = [
        {
            "mongo_id": "1" * 24,
            "content": "The tomcat manager exposes a struts rce.",
        }
    ]
    records = twin.verify_edges(model, documents, {"tomcat", "rce"})
    assert records[0]["confidence"] == 0.9


def test_endpoints_in_separate_documents_do_not_support_the_edge():
    model = build(nodes=EVIDENCE_MODEL_NODES, edges=EVIDENCE_MODEL_EDGES)
    documents = [
        {"mongo_id": "2" * 24, "content": "The tomcat manager is reachable."},
        {"mongo_id": "3" * 24, "content": "A struts rce was published."},
    ]
    assert twin.verify_edges(model, documents, {"tomcat", "rce"}) == []


def test_unmatched_nodes_carry_no_edge_evidence():
    model = build(nodes=EVIDENCE_MODEL_NODES, edges=EVIDENCE_MODEL_EDGES)
    documents = [{"mongo_id": "4" * 24, "content": "tomcat manager exposes struts rce"}]
    assert twin.verify_edges(model, documents, {"tomcat"}) == []


def test_edge_evidence_coverage_reaches_the_structural_score():
    model = build(nodes=EVIDENCE_MODEL_NODES, edges=EVIDENCE_MODEL_EDGES)
    result = paths_of(model, supported_edges={1})
    structural = result["scenarios"][0]["structural"]
    assert structural["hops"] == 3
    assert structural["edge_evidence_coverage"] == pytest.approx(1 / 3, abs=1e-4)


# --- criterion polarity -----------------------------------------------------


def test_cost_criterion_is_inverted_before_the_integral_score():
    criteria = [
        twin.TwinCriterion("safety", "higher is better", 1.0, "benefit"),
        twin.TwinCriterion("months", "how long it takes", 1.0, "cost"),
    ]
    scores = [
        {"criterion": "safety", "score": 0.8},
        {"criterion": "months", "score": 0.8},
    ]
    integral, disagreement = twin.aggregate(scores, criteria)
    assert integral == pytest.approx(0.5)
    assert disagreement == pytest.approx(0.3)


def test_benefit_only_criteria_keep_the_earlier_result():
    criteria = [
        twin.TwinCriterion("safety", "higher is better", 1.0),
        twin.TwinCriterion("fit", "higher is better", 1.0),
    ]
    scores = [
        {"criterion": "safety", "score": 0.8},
        {"criterion": "fit", "score": 0.6},
    ]
    integral, _ = twin.aggregate(scores, criteria)
    assert integral == pytest.approx(0.7)


def test_unsupported_direction_is_rejected():
    with pytest.raises(twin.TwinModelError, match="unsupported direction"):
        twin.parse_twin_model(
            """
name: unit
current_state: s0
goal_state: st
nodes:
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
edges:
  - {source: s0, target: st, relation: enables}
criteria:
  - {name: risk, weight: 1.0, description: d, direction: sideways}
"""
        )


# --- targeted repair --------------------------------------------------------


BROKEN = """
name: broken
current_state: s0
goal_state: st
nodes:
  - {id: s0, kind: state, label: Start}
  - {id: st, kind: state, label: Goal}
  - {id: s0, kind: state, label: Duplicate}
  - {id: a1, kind: asset, label: Host}
edges:
  - {source: s0, target: a1, relation: reaches}
  - {source: a1, target: st, relation: reaches}
  - {source: a1, target: ghost, relation: reaches}
  - {source: a1, target: a1, relation: loops}
  - {source: s0, target: a1, relation: reaches}
criteria:
  - {name: risk, weight: 1.0, description: d}
"""


def test_repair_removes_only_structural_defects():
    model, repairs = twin.parse_or_repair(BROKEN)
    assert len(model.nodes) == 3
    assert len(model.edges) == 2
    assert any("duplicate node" in line for line in repairs)
    assert any("undeclared endpoint" in line for line in repairs)
    assert any("self-loop" in line for line in repairs)
    assert any("repeated edge" in line for line in repairs)


def test_repair_never_invents_structure():
    _, repairs = twin.repair_model(BROKEN)
    assert all(line.startswith("dropped") for line in repairs)


def test_a_sound_document_is_parsed_without_repair():
    text = FIXTURE.read_text(encoding="utf-8")
    model, repairs = twin.parse_or_repair(text)
    assert repairs == []
    assert len(model.nodes) == 34


def test_a_document_repair_cannot_fix_still_raises():
    with pytest.raises(twin.TwinModelError):
        twin.parse_or_repair(
            """
name: unfixable
current_state: missing
goal_state: st
nodes:
  - {id: st, kind: state, label: Goal}
edges: []
criteria:
  - {name: risk, weight: 1.0, description: d}
"""
        )


# --- the shipped fixture stays admissible ----------------------------------


def test_fixture_enumeration_reports_the_active_constraint_set():
    model = twin.parse_twin_model(FIXTURE.read_text(encoding="utf-8"))
    result = paths_of(model, max_paths=5)
    assert result["rejected_by_constraint"] > 0
    assert {entry["criterion"] for entry in result["constraints_applied"]} == {
        "budget",
        "months",
        "personnel",
        "regulatory",
    }


# --- evaluator panel --------------------------------------------------------


def vote(score, confidence, state, evaluator, rationale=""):
    return {
        "criterion": "risk",
        "score": score,
        "confidence": confidence,
        "evidence_state": state,
        "rationale": rationale,
        "evaluator": evaluator,
    }


def test_panel_takes_the_median_and_records_the_spread():
    merged = twin.merge_panel_votes(
        [
            vote(0.2, 0.9, "sufficient", "a", "low"),
            vote(0.6, 0.7, "sufficient", "b", "mid"),
            vote(0.8, 0.5, "insufficient", "c", "high"),
        ]
    )
    assert merged["score"] == pytest.approx(0.6)
    assert merged["confidence"] == pytest.approx(0.7)
    assert merged["panel_disagreement"] > 0.2
    assert [entry["evaluator"] for entry in merged["panel"]] == ["a", "b", "c"]


def test_panel_keeps_the_rationale_of_the_settled_vote():
    merged = twin.merge_panel_votes(
        [
            vote(0.1, 0.9, "sufficient", "a", "the low reading"),
            vote(0.5, 0.9, "sufficient", "b", "the middle reading"),
            vote(0.9, 0.9, "sufficient", "c", "the high reading"),
        ]
    )
    assert merged["rationale"] == "the middle reading"


def test_panel_calls_evidence_sufficient_only_when_unanimous():
    unanimous = twin.merge_panel_votes(
        [vote(0.5, 0.9, "sufficient", "a"), vote(0.5, 0.9, "sufficient", "b")]
    )
    split = twin.merge_panel_votes(
        [vote(0.5, 0.9, "sufficient", "a"), vote(0.5, 0.9, "insufficient", "b")]
    )
    conflict = twin.merge_panel_votes(
        [vote(0.5, 0.9, "sufficient", "a"), vote(0.5, 0.9, "contradictory", "b")]
    )
    assert unanimous["evidence_state"] == "sufficient"
    assert split["evidence_state"] == "insufficient"
    assert conflict["evidence_state"] == "contradictory"


def test_a_single_evaluator_reports_no_disagreement():
    merged = twin.merge_panel_votes([vote(0.4, 0.8, "sufficient", "a")])
    assert merged["score"] == pytest.approx(0.4)
    assert merged["panel_disagreement"] == 0.0


def test_an_empty_panel_is_rejected():
    with pytest.raises(twin.TwinModelError):
        twin.merge_panel_votes([])


def test_enumerated_scenario_exposes_the_edges_it_used():
    """Виклик має бачити, ЯКИМИ ребрами пройдено шлях.

    Без цього зовнішнє ранжування не може відрізнити крок, що називає механізм,
    від простого твердження про досяжність: назви відношень лишаються тільки
    всередині перебору."""
    model = twin.parse_twin_model(FIXTURE.read_text(encoding="utf-8"))
    result = paths_of(model, max_paths=5)
    assert result["scenarios"], "фікстура має давати хоч один шлях"
    for scenario in result["scenarios"]:
        assert "edge_path" in scenario
        assert len(scenario["edge_path"]) == len(scenario["path"]) - 1
        for index in scenario["edge_path"]:
            assert 0 <= index < len(model.edges)
