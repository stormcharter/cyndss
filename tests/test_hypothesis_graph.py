"""
Юніт-тести для hypothesis_graph.py (побудова networkx.DiGraph із ланцюгів
гіпотез semantic_reasoner.hypotheses() — чистий модуль, без streamlit/pyvis).

Запуск:
    pytest tests/test_hypothesis_graph.py -v
або як звичайний скрипт (без pytest):
    python tests/test_hypothesis_graph.py
"""

import os
import sys

# Дозволяємо запуск і як `pytest`, і як `python tests/...` з кореня репо.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis_graph import build_hypothesis_graph, VIRTUAL_START_NODE  # noqa: E402


def _hyp(chain, edges, is_anchor=False):
    """Мінімальний хеш-гіпотези у форматі semantic_reasoner.hypotheses()."""
    return {"chain": chain, "edges": edges, "is_anchor": is_anchor}


# --- goal-вузол присутній, kind == "goal" -----------------------------------
def test_goal_node_present_with_kind_goal():
    hyps = [_hyp(["CVE-2024-1111", "GOAL"], [("CVE-2024-1111", "GOAL", 5.0)], is_anchor=True)]
    G = build_hypothesis_graph(hyps, "GOAL")
    assert "GOAL" in G.nodes
    assert G.nodes["GOAL"]["kind"] == "goal"


# --- усі вузли з усіх ланцюгів присутні --------------------------------------
# ПРИМІТКА: очікуваний набір вузлів свідомо доповнено VIRTUAL_START_NODE — це
# зміна ВИМОГИ (єдиний початок ланцюга, див. блок «фіктивний старт» нижче), а
# не послаблення тесту. Щоб перевірити ЛИШЕ реальні вузли ланцюгів, той самий
# виклик із start_node=None перевіряється у test_virtual_start_can_be_disabled.
def test_all_chain_nodes_present():
    hyps = [
        _hyp(["CVE-2024-1111", "midterm", "GOAL"],
             [("CVE-2024-1111", "midterm", 5.0), ("midterm", "GOAL", 3.0)], is_anchor=True),
        _hyp(["MITRE T1059", "GOAL"], [("MITRE T1059", "GOAL", 2.0)], is_anchor=False),
    ]
    G = build_hypothesis_graph(hyps, "GOAL")
    assert set(G.nodes) == {"CVE-2024-1111", "midterm", "GOAL", "MITRE T1059",
                            VIRTUAL_START_NODE}


# --- ФІКТИВНИЙ СТАРТ: єдиний початок ланцюга --------------------------------
# ВИМОГА (з обговорення з науковим керівником). Обхід від цілі назад дає N
# незалежних ланцюгів, тож на екрані виходив граф із ~5 початками й 1 кінцем —
# читається не як ланцюг міркування, а як віяло. Додаємо фіктивний вузол
# «Теперішній стан», з якого стартують УСІ ланцюги: граф стає «єдиний початок →
# … → єдина ціль». Вузол службовий (kind="start", virtual=True), у корпусі його
# немає — тому й ребра з нього позначені virtual=True з нульовою вагою, щоб
# рендер і будь-яка метрика ваги могли їх відрізнити від реальних звʼязків.
def test_virtual_start_node_added_with_kind_start():
    hyps = [_hyp(["CVE-2024-1111", "GOAL"], [("CVE-2024-1111", "GOAL", 5.0)], is_anchor=True)]
    G = build_hypothesis_graph(hyps, "GOAL")
    assert VIRTUAL_START_NODE in G.nodes
    assert G.nodes[VIRTUAL_START_NODE]["kind"] == "start"
    assert G.nodes[VIRTUAL_START_NODE]["virtual"] is True


def test_every_chain_start_is_linked_from_virtual_start():
    """Кожен ланцюг починається з фіктивного вузла — жодного «висячого» початку."""
    hyps = [
        _hyp(["CVE-2024-1111", "midterm", "GOAL"],
             [("CVE-2024-1111", "midterm", 5.0), ("midterm", "GOAL", 3.0)], is_anchor=True),
        _hyp(["MITRE T1059", "GOAL"], [("MITRE T1059", "GOAL", 2.0)], is_anchor=False),
        _hyp(["term2", "GOAL"], [("term2", "GOAL", 1.0)], is_anchor=False),
    ]
    G = build_hypothesis_graph(hyps, "GOAL")
    assert G.has_edge(VIRTUAL_START_NODE, "CVE-2024-1111")
    assert G.has_edge(VIRTUAL_START_NODE, "MITRE T1059")
    assert G.has_edge(VIRTUAL_START_NODE, "term2")
    # проміжні вузли з фіктивного старту НЕ зʼєднуються
    assert not G.has_edge(VIRTUAL_START_NODE, "midterm")


def test_graph_has_exactly_one_source_and_one_sink():
    """Власне вимога: єдиний початок (in_degree 0) і єдиний кінець (out_degree 0)."""
    hyps = [
        _hyp(["CVE-2024-1111", "midterm", "GOAL"],
             [("CVE-2024-1111", "midterm", 5.0), ("midterm", "GOAL", 3.0)], is_anchor=True),
        _hyp(["MITRE T1059", "GOAL"], [("MITRE T1059", "GOAL", 2.0)], is_anchor=False),
        _hyp(["term2", "GOAL"], [("term2", "GOAL", 1.0)], is_anchor=False),
    ]
    G = build_hypothesis_graph(hyps, "GOAL")
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    assert sources == [VIRTUAL_START_NODE], sources
    assert sinks == ["GOAL"], sinks


def test_virtual_edges_are_marked_and_weightless():
    """Службові ребра не мають підмішувати вагу в реальні метрики ланцюга."""
    hyps = [_hyp(["term2", "GOAL"], [("term2", "GOAL", 1.0)], is_anchor=False)]
    G = build_hypothesis_graph(hyps, "GOAL")
    edge = G[VIRTUAL_START_NODE]["term2"]
    assert edge.get("virtual") is True
    assert edge["weight"] == 0.0
    # реальне ребро службовим НЕ позначене
    assert not G["term2"]["GOAL"].get("virtual")


def test_virtual_start_can_be_disabled():
    """start_node=None — стара поведінка (лише реальні вузли корпусу)."""
    hyps = [
        _hyp(["CVE-2024-1111", "midterm", "GOAL"],
             [("CVE-2024-1111", "midterm", 5.0), ("midterm", "GOAL", 3.0)], is_anchor=True),
        _hyp(["MITRE T1059", "GOAL"], [("MITRE T1059", "GOAL", 2.0)], is_anchor=False),
    ]
    G = build_hypothesis_graph(hyps, "GOAL", start_node=None)
    assert set(G.nodes) == {"CVE-2024-1111", "midterm", "GOAL", "MITRE T1059"}


def test_no_virtual_start_when_no_hypotheses():
    """Порожній список гіпотез — нема чого починати, службовий вузол не зʼявляється."""
    G = build_hypothesis_graph([], "GOAL")
    assert VIRTUAL_START_NODE not in G.nodes
    assert set(G.nodes) == {"GOAL"}


def test_anchor_flag_survives_virtual_start():
    """Фіктивний старт не «перебиває» позначку якоря на реальному початку."""
    hyps = [_hyp(["CVE-2024-1111", "GOAL"], [("CVE-2024-1111", "GOAL", 5.0)], is_anchor=True)]
    G = build_hypothesis_graph(hyps, "GOAL")
    assert G.nodes["CVE-2024-1111"].get("is_anchor") is True
    assert not G.nodes[VIRTUAL_START_NODE].get("is_anchor")


# --- ребра направлені вздовж ланцюга до цілі, вага проставлена --------------
def test_edges_directed_toward_goal_with_weight():
    hyps = [
        _hyp(["CVE-2024-1111", "midterm", "GOAL"],
             [("CVE-2024-1111", "midterm", 5.0), ("midterm", "GOAL", 3.0)], is_anchor=True),
    ]
    G = build_hypothesis_graph(hyps, "GOAL")
    assert G.has_edge("CVE-2024-1111", "midterm")
    assert G["CVE-2024-1111"]["midterm"]["weight"] == 5.0
    assert G.has_edge("midterm", "GOAL")
    assert G["midterm"]["GOAL"]["weight"] == 3.0
    # напрям — ЛИШЕ вперед (до цілі), зворотних ребер немає.
    assert not G.has_edge("midterm", "CVE-2024-1111")
    assert not G.has_edge("GOAL", "midterm")


# --- start якірної гіпотези позначено is_anchor=True -------------------------
def test_anchor_start_node_marked():
    hyps = [
        _hyp(["CVE-2024-1111", "midterm", "GOAL"],
             [("CVE-2024-1111", "midterm", 5.0), ("midterm", "GOAL", 3.0)], is_anchor=True),
        _hyp(["term2", "GOAL"], [("term2", "GOAL", 1.0)], is_anchor=False),
    ]
    G = build_hypothesis_graph(hyps, "GOAL")
    assert G.nodes["CVE-2024-1111"].get("is_anchor") is True
    # проміжний і goal вузол якірного ланцюга — НЕ якірні (лише start).
    assert not G.nodes["midterm"].get("is_anchor")
    assert not G.nodes["term2"].get("is_anchor")


# --- label_map анотація застосована до label вузлів --------------------------
def test_label_map_annotation_applied():
    hyps = [
        _hyp(["MITRE T1059", "GOAL"], [("MITRE T1059", "GOAL", 2.0)], is_anchor=False),
    ]
    label_map = {"MITRE T1059": "Command and Scripting Interpreter", "GOAL": "Ціль атаки"}
    G = build_hypothesis_graph(hyps, "GOAL", label_map=label_map)
    assert G.nodes["MITRE T1059"]["label"] == "MITRE T1059 (Command and Scripting Interpreter)"
    assert G.nodes["GOAL"]["label"] == "GOAL (Ціль атаки)"


def test_label_map_none_leaves_label_as_node_id():
    hyps = [_hyp(["term2", "GOAL"], [("term2", "GOAL", 1.0)], is_anchor=False)]
    G = build_hypothesis_graph(hyps, "GOAL")
    assert G.nodes["term2"]["label"] == "term2"
    assert G.nodes["GOAL"]["label"] == "GOAL"


# --- спільний вузол/ребро між ланцюгами не дублюються, вага = максимум -------
def test_shared_node_and_edge_deduped_with_max_weight():
    hyps = [
        _hyp(["CVE-2024-1111", "midterm", "GOAL"],
             [("CVE-2024-1111", "midterm", 5.0), ("midterm", "GOAL", 3.0)], is_anchor=True),
        _hyp(["otherterm", "midterm", "GOAL"],
             [("otherterm", "midterm", 1.0), ("midterm", "GOAL", 9.0)], is_anchor=False),
    ]
    G = build_hypothesis_graph(hyps, "GOAL")
    # "midterm" присутній рівно один раз (властивість DiGraph — дублю немає).
    assert list(G.nodes).count("midterm") == 1
    # Спільне ребро midterm->GOAL зустрічається у двох гіпотезах з вагами
    # 3.0 і 9.0 — лишається МАКСИМАЛЬНА.
    assert G["midterm"]["GOAL"]["weight"] == 9.0


# --- kind вузла за префіксом ідентифікатора ----------------------------------
def test_node_kind_by_prefix():
    hyps = [
        _hyp(["CVE-2024-1111", "midterm", "GOAL"],
             [("CVE-2024-1111", "midterm", 5.0), ("midterm", "GOAL", 3.0)], is_anchor=True),
        _hyp(["MITRE T1059", "GOAL"], [("MITRE T1059", "GOAL", 2.0)], is_anchor=False),
    ]
    G = build_hypothesis_graph(hyps, "GOAL")
    assert G.nodes["CVE-2024-1111"]["kind"] == "cve"
    assert G.nodes["MITRE T1059"]["kind"] == "mitre"
    assert G.nodes["midterm"]["kind"] == "term"
    assert G.nodes["GOAL"]["kind"] == "goal"


# --- goal перекриває kind, навіть якщо ціль сама схожа на CVE/MITRE ----------
def test_goal_kind_overrides_prefix_match():
    hyps = [_hyp(["term2", "CVE-2024-0001"], [("term2", "CVE-2024-0001", 1.0)], is_anchor=False)]
    G = build_hypothesis_graph(hyps, "CVE-2024-0001")
    assert G.nodes["CVE-2024-0001"]["kind"] == "goal"


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
