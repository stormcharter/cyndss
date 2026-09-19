"""
Юніт-тести для чистих хелперів hypothesis_labels.py
(мапінг MITRE-імен для панелі гіпотез у ui.py).

Запуск:
    pytest tests/test_hypothesis_labels.py -v
або як звичайний скрипт (без pytest):
    python tests/test_hypothesis_labels.py
"""

import os
import sys

# Дозволяємо запуск і як `pytest`, і як `python tests/...` з кореня репо.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis_labels import (  # noqa: E402
    build_mitre_label_map,
    annotate_chain_for_display,
)


# --- build_mitre_label_map: додає префікс "MITRE " до кожного ключа --------
def test_build_mitre_label_map_adds_prefix():
    result = build_mitre_label_map({"T1056": "Input Capture"})
    assert result == {"MITRE T1056": "Input Capture"}


# --- build_mitre_label_map: декілька записів, порожній вхід -----------------
def test_build_mitre_label_map_multiple_entries():
    result = build_mitre_label_map({"T1056": "Input Capture", "T1003": "OS Credential Dumping"})
    assert result == {
        "MITRE T1056": "Input Capture",
        "MITRE T1003": "OS Credential Dumping",
    }


def test_build_mitre_label_map_empty_input():
    assert build_mitre_label_map({}) == {}


# --- annotate_chain_for_display: анотує вузол-техніку, лишає термін як є ---
def test_annotate_chain_for_display_annotates_known_node():
    out = annotate_chain_for_display(
        ["MITRE T1056", "ransomware"], {"MITRE T1056": "Input Capture"}
    )
    assert out[0] == "MITRE T1056 (Input Capture)"
    assert out[1] == "ransomware"


# --- annotate_chain_for_display: вузол без запису лишається незмінним ------
def test_annotate_chain_for_display_unknown_node_unchanged():
    out = annotate_chain_for_display(["ransomware"], {"MITRE T1056": "Input Capture"})
    assert out == ["ransomware"]


# --- annotate_chain_for_display: порожній label_map не падає ---------------
def test_annotate_chain_for_display_empty_label_map():
    out = annotate_chain_for_display(["MITRE T1056", "ransomware"], {})
    assert out == ["MITRE T1056", "ransomware"]


# --- annotate_chain_for_display: порожній ланцюг -> порожній список --------
def test_annotate_chain_for_display_empty_chain():
    assert annotate_chain_for_display([], {"MITRE T1056": "Input Capture"}) == []


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
