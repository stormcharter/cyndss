"""
Бойовий шар сценаріїв: опис інфраструктури + корпус OSINT -> ранжовані сценарії.

НАВІЩО ОКРЕМИЙ ШАР. Орієнтований типізований граф виміряний на контрольному
наборі, але в UI аналітик досі отримує неорієнтовану мережу співзустрічальності,
де стрілка означає лише порядок читання. Щоб підключити його до бойового шляху,
потрібне те, чого в OSINT-тексті немає: опис інфраструктури з полями
(`components`, `permission_edges`, `vulnerabilities`). Тому поділ такий:

    структура   <- опис інфраструктури, який дає АНАЛІТИК;
    доказовість <- корпус OSINT, зібраний системою.

Витягати структуру з тексту мовною моделлю було б тим самим класом помилки, що
й вигаданий T1190, тільки на рівень вище: там вигадувалась мітка, тут
вигадувався б перехід. Тому структура не інферується — вона подається.

Запуск:
    python tests/test_twin_scenarios.py
або pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twin_scenarios import (  # noqa: E402
    build_scenarios,
    parse_infrastructure_model,
)

MACHINE = {
    "machine": "Prod web",
    "components": [
        {"component_id": "H1", "parent_id": "-", "layer_or_type": "Host",
         "name": "web host", "runs_as": "root", "exposure": "internal"},
        {"component_id": "C2", "parent_id": "H1", "layer_or_type": "Service",
         "name": "nginx gateway", "role": "reverse proxy",
         "runs_as": "www-data", "exposure": "Public"},
    ],
    "identities": [{"principal": "www-data"}, {"principal": "root"}],
    "vulnerabilities": [
        {"finding_id": "V1", "affected_component": "C2",
         "issue": "path traversal in gateway",
         "resulting_access": "www-data", "reference": "CVE-2026-11111"},
    ],
    "permission_edges": [
        {"subject": "www-data", "permission_or_action": "run sudo helper",
         "object": "H1", "security_consequence": "root"},
    ],
}

DOCS = [
    {"mongo_id": "d1", "title": "Path traversal exploited in nginx gateway",
     "summary": "Attackers abuse path traversal in the nginx gateway to read files.",
     "content": "nginx gateway path traversal exploited in the wild"},
    {"mongo_id": "d2", "title": "Unrelated news",
     "summary": "A logo generator was released.", "content": "logo generator"},
]


# --- розбір опису ---------------------------------------------------------
def test_parses_json_infrastructure_model():
    model = parse_infrastructure_model(
        '{"machine": "X", "components": [{"component_id": "C1", "name": "svc"}]}')
    assert model["machine"] == "X"
    assert model["components"][0]["component_id"] == "C1"


def test_rejects_text_that_is_not_a_model():
    for bad in ("", "не json", "[1,2,3]", '{"machine": "X"}'):
        try:
            parse_infrastructure_model(bad)
        except ValueError:
            continue
        raise AssertionError(f"мав відхилити: {bad!r}")


def test_error_message_names_what_is_missing():
    try:
        parse_infrastructure_model('{"machine": "X"}')
    except ValueError as exc:
        assert "components" in str(exc)
    else:
        raise AssertionError("мав відхилити модель без компонентів")


# --- побудова сценаріїв ---------------------------------------------------
def test_finds_a_scenario_to_the_named_goal():
    got = build_scenarios(MACHINE, "root", DOCS)
    assert got["found"] is True
    assert got["scenarios"], got.get("note")


def test_scenario_carries_named_transitions_not_bare_nodes():
    top = build_scenarios(MACHINE, "root", DOCS)["scenarios"][0]
    assert top["steps"], top
    for step in top["steps"]:
        assert step["relation"], step
        assert step["from"] and step["to"]


def test_scenario_reports_integral_score_and_criteria():
    top = build_scenarios(MACHINE, "root", DOCS)["scenarios"][0]
    assert 0.0 <= top["j"] <= 1.0
    names = {c["criterion"] for c in top["criteria"]}
    assert names == {"feasibility", "evidence", "effort"}


def test_scenarios_are_ordered_by_integral_score():
    scenarios = build_scenarios(MACHINE, "root", DOCS)["scenarios"]
    scores = [s["j"] for s in scenarios]
    assert scores == sorted(scores, reverse=True)


def test_unknown_goal_is_reported_not_invented():
    got = build_scenarios(MACHINE, "domain-admin-that-does-not-exist", DOCS)
    assert got["found"] is False
    assert not got["scenarios"]
    assert got["note"]


def test_missing_goal_is_reported():
    got = build_scenarios(MACHINE, "", DOCS)
    assert got["found"] is False
    assert got["note"]


def test_step_carries_node_ids_for_matching_evidence():
    """Доказовість приходить із `verify_edges` у термінах ID вузлів, а крок
    показує ПІДПИСИ. Без ID у кроці позначку «є згадка в корпусі» неможливо
    прив'язати до конкретного переходу — вона просто ніколи не з'явиться."""
    top = build_scenarios(MACHINE, "root", DOCS)["scenarios"][0]
    for step in top["steps"]:
        assert step["from_id"] and step["to_id"], step


def test_step_ids_match_the_evidence_keys():
    got = build_scenarios(MACHINE, "root", DOCS)
    keys = {(e["source"], e["relation"], e["target"]) for e in got["evidence"]}
    steps = {(s["from_id"], s["relation"], s["to_id"])
             for sc in got["scenarios"] for s in sc["steps"]}
    assert keys, "фікстура має давати хоч один підтверджений перехід"
    assert keys & steps, f"ключі доказів не збігаються з кроками: {keys} vs {steps}"


# --- доказовість із корпусу ------------------------------------------------
def test_edge_supported_by_corpus_gets_its_document():
    got = build_scenarios(MACHINE, "root", DOCS)
    ids = {e["mongo_id"] for e in got["evidence"]}
    assert "d1" in ids, got["evidence"]


def test_irrelevant_document_is_not_offered_as_evidence():
    got = build_scenarios(MACHINE, "root", DOCS)
    assert "d2" not in {e["mongo_id"] for e in got["evidence"]}


def test_evidence_is_optional_and_empty_corpus_is_not_an_error():
    got = build_scenarios(MACHINE, "root", [])
    assert got["found"] is True
    assert got["evidence"] == []


def test_model_summary_reports_graph_size():
    got = build_scenarios(MACHINE, "root", DOCS)
    assert got["nodes"] > 0
    assert got["edges"] > 0
    assert got["transitions"] <= got["edges"]


# --- дані для захисного плеча (twin_defense.rank_defenses) ----------------
# НАВІЩО. `twin_defense.rank_defenses(model, attack_paths)` рахує набори
# засобів захисту САМЕ за `edge_path` шляхів атаки й моделлю, з якої вони
# побудовані. Якщо `build_scenarios` не віддає ні те, ні інше, UI не може
# підключити захисне плече, не перебудовуючи модель вдруге (а це вже друга
# копія логіки побудови графа - джерело розбіжностей). Це той самий клас
# розриву, що вже стався з `bidirectional_path`: код існує й покритий
# тестами, але нізвідки не викликається, бо викликати нема чим.
def test_scenario_carries_edge_path_for_the_defense_layer():
    top = build_scenarios(MACHINE, "root", DOCS)["scenarios"][0]
    assert top["edge_path"], top
    # довжина має збігатись з кількістю кроків - це той самий шлях, лише в
    # двох поданнях (індекси ребер і вже розгорнуті кроки).
    assert len(top["edge_path"]) == len(top["steps"])


def test_result_exposes_the_underlying_model_for_the_defense_layer():
    got = build_scenarios(MACHINE, "root", DOCS)
    model = got["model"]
    # Не перевіряємо конкретний тип імпортом qa.twin (модуль і так залежить
    # від нього транзитивно) - лише те, що це та сама модель, з якої побудовані
    # кроки сценарію: вузол з `steps[0]["from_id"]` мусить у ній існувати.
    top = got["scenarios"][0]
    from_id = top["steps"][0]["from_id"]
    assert from_id in model.node_by_id


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
