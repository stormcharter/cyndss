"""
Юніт-тести для параметра `label_map` у semantic_reasoner.format_hypotheses_for_llm.

Запуск:
    pytest tests/test_format_hypotheses_labels.py -v
або як звичайний скрипт (без pytest):
    python tests/test_format_hypotheses_labels.py
"""

import os
import sys

# Дозволяємо запуск і як `pytest`, і як `python tests/...` з кореня репо.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_reasoner import format_hypotheses_for_llm  # noqa: E402


def _hyp(rank=1, chain=None, hops=2, total_weight=1.5, score=0.9,
         source_ids=None, is_anchor=False):
    return {
        "rank": rank,
        "chain": chain if chain is not None else ["ransomware", "MITRE T1056"],
        "hops": hops,
        "total_weight": total_weight,
        "score": score,
        "source_ids": source_ids if source_ids is not None else ["doc_1", "doc_2"],
        "is_anchor": is_anchor,
    }


# --- label_map анотує вузол-якір усередині ланцюга -------------------------
def test_label_map_annotates_chain_node():
    hyp = _hyp(chain=["ransomware", "MITRE T1056"])
    out = format_hypotheses_for_llm(
        [hyp], "MITRE T1056", label_map={"MITRE T1056": "Input Capture"}
    )
    assert "MITRE T1056 (Input Capture)" in out


# --- label_map анотує ціль (goal_concept) -----------------------------------
def test_label_map_annotates_goal():
    hyp = _hyp(chain=["ransomware", "MITRE T1056"])
    out = format_hypotheses_for_llm(
        [hyp], "MITRE T1056", label_map={"MITRE T1056": "Input Capture"}
    )
    header = [ln for ln in out.splitlines() if ln.startswith("ЦІЛЬОВИЙ РЕЗУЛЬТАТ")]
    assert header, "очікувався рядок заголовку цілі"
    assert "MITRE T1056 (Input Capture)" in header[0]


# --- регрес-гард: без label_map (None) вивід не змінюється -----------------
def test_no_label_map_matches_baseline_output():
    hyp = _hyp(chain=["ransomware", "MITRE T1056"])
    baseline = format_hypotheses_for_llm([hyp], "MITRE T1056")
    with_none = format_hypotheses_for_llm([hyp], "MITRE T1056", label_map=None)
    assert baseline == with_none


# --- вузол без запису в label_map лишається неанотованим --------------------
def test_node_without_label_map_entry_unchanged():
    hyp = _hyp(chain=["ransomware", "MITRE T1056"])
    out = format_hypotheses_for_llm(
        [hyp], "MITRE T1056", label_map={"MITRE T1056": "Input Capture"}
    )
    # "ransomware" без запису в мапі — без дужок-анотації.
    assert "ransomware (" not in out
    assert "ransomware" in out


# --- порожній label_map={} еквівалентний None -------------------------------
def test_empty_label_map_matches_none():
    hyp = _hyp(chain=["ransomware", "MITRE T1056"])
    with_none = format_hypotheses_for_llm([hyp], "MITRE T1056", label_map=None)
    with_empty = format_hypotheses_for_llm([hyp], "MITRE T1056", label_map={})
    assert with_none == with_empty


# --- регрес-гард на порожньому корпусі (hyps=[]) ----------------------------
def test_no_label_map_matches_baseline_output_empty_corpus():
    baseline = format_hypotheses_for_llm([], "MITRE T1056")
    with_none = format_hypotheses_for_llm([], "MITRE T1056", label_map=None)
    assert baseline == with_none


# --- label_map анотує ціль і в гілці порожнього корпусу ---------------------
def test_label_map_annotates_goal_in_empty_corpus_branch():
    out = format_hypotheses_for_llm(
        [], "MITRE T1056", label_map={"MITRE T1056": "Input Capture"}
    )
    assert "MITRE T1056 (Input Capture)" in out


# --- Джерел: N — явне число джерел у блоці гіпотези -------------------------
def test_output_shows_source_count():
    """Блок гіпотези має явно показувати кількість джерел (щоб LLM бачив, чи
    ланцюжок спирається на 1 документ, чи на кілька)."""
    hyp = _hyp(source_ids=["doc_1", "doc_2", "doc_3"])
    out = format_hypotheses_for_llm([hyp], "MITRE T1056")
    assert "Джерел: 3" in out


# --- одне джерело -> маркер «слабке підтвердження» --------------------------
def test_single_source_gets_weak_marker():
    hyp = _hyp(source_ids=["doc_1"])
    out = format_hypotheses_for_llm([hyp], "MITRE T1056")
    assert "Джерел: 1" in out
    assert "одне джерело — слабке підтвердження" in out


def test_multiple_sources_no_weak_marker():
    hyp = _hyp(source_ids=["doc_1", "doc_2"])
    out = format_hypotheses_for_llm([hyp], "MITRE T1056")
    assert "одне джерело — слабке підтвердження" not in out
    assert "слабке підтвердження" not in out


def test_zero_sources_gets_no_confirmation_marker():
    # 0 джерел — не слабший, а взагалі не підтверджений випадок: свій маркер.
    hyp = _hyp(source_ids=[])
    out = format_hypotheses_for_llm([hyp], "MITRE T1056")
    assert "Джерел: 0" in out
    assert "немає джерел — не підтверджено" in out


# --- пом'якшений ярлик якоря: верифіковано ІДЕНТИФІКАТОР, не факт демонстрації
def test_anchor_label_clarifies_id_verified_not_fact():
    hyp = _hyp(is_anchor=True)
    out = format_hypotheses_for_llm([hyp], "MITRE T1056")
    assert "[ЯКІР: код сутності верифіковано в MITRE/NVD]" in out
    # Старий (уводить в оману) ярлик має зникнути.
    assert "[ЯКІР: верифікований факт]" not in out


# --- сумарна вага + середня вага ребра (score) показані окремо --------------
def test_output_shows_total_and_mean_edge_weight():
    hyp = _hyp(total_weight=12.5, score=6.25)
    out = format_hypotheses_for_llm([hyp], "MITRE T1056")
    assert "сумарна вага: 12.500" in out
    assert "середня вага ребра: 6.250" in out


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
