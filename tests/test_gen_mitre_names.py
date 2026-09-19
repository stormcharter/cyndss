"""
Юніт-тести для tools/gen_mitre_names.py::extract_technique_names.

Мережевий тест НЕ пишемо (мережа в CI недоступна) — перевіряємо лише чисту
функцію вибірки {external_id: name} на синтетичному STIX-бандлі.

Запуск:
    pytest tests/test_gen_mitre_names.py -v
або як звичайний скрипт (без pytest):
    python tests/test_gen_mitre_names.py
"""

import os
import sys

# Дозволяємо запуск і як `pytest`, і як `python tests/...` з кореня репо.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.gen_mitre_names import extract_technique_names  # noqa: E402


def _attack_pattern(name, external_id, revoked=False, deprecated=False,
                     source_name="mitre-attack"):
    obj = {
        "type": "attack-pattern",
        "name": name,
        "external_references": [
            {"source_name": source_name, "external_id": external_id},
        ],
    }
    if revoked:
        obj["revoked"] = True
    if deprecated:
        obj["x_mitre_deprecated"] = True
    return obj


# --- базовий випадок: валідна техніка потрапляє у словник -----------------
def test_extracts_valid_technique():
    bundle = {"objects": [_attack_pattern("Input Capture", "T1056")]}
    result = extract_technique_names(bundle)
    assert result == {"T1056": "Input Capture"}


# --- субтехніки (T####.###) теж валідні ------------------------------------
def test_extracts_subtechnique_id():
    bundle = {"objects": [_attack_pattern("Keylogging", "T1056.001")]}
    result = extract_technique_names(bundle)
    assert result == {"T1056.001": "Keylogging"}


# --- revoked обʼєкти виключаються ------------------------------------------
def test_excludes_revoked():
    bundle = {"objects": [_attack_pattern("Old Technique", "T1999", revoked=True)]}
    result = extract_technique_names(bundle)
    assert result == {}


# --- x_mitre_deprecated обʼєкти виключаються -------------------------------
def test_excludes_deprecated():
    bundle = {"objects": [_attack_pattern("Deprecated Technique", "T1998", deprecated=True)]}
    result = extract_technique_names(bundle)
    assert result == {}


# --- обʼєкти інших типів (не attack-pattern) ігноруються -------------------
def test_ignores_non_attack_pattern_objects():
    bundle = {"objects": [
        {"type": "malware", "name": "SomeMalware",
         "external_references": [{"source_name": "mitre-attack", "external_id": "S0001"}]},
    ]}
    result = extract_technique_names(bundle)
    assert result == {}


# --- external_reference з іншим source_name ігнорується --------------------
def test_ignores_non_mitre_attack_source():
    bundle = {"objects": [_attack_pattern("Capec Thing", "CAPEC-1", source_name="capec")]}
    result = extract_technique_names(bundle)
    assert result == {}


# --- external_id, що не відповідає формату T####(.###)?, ігнорується -------
def test_ignores_malformed_external_id():
    bundle = {"objects": [_attack_pattern("Bad Id", "TA0001")]}
    result = extract_technique_names(bundle)
    assert result == {}


# --- ручний запуск без pytest ----------------------------------------------
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
