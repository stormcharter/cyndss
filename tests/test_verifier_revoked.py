"""
Юніт-тести для shipper/verifier.py: MitreAttackKB._add_technique не має
вносити у KB відкликані (revoked) або deprecated MITRE ATT&CK техніки.

Баг: `_add_technique` додавав ЛЮБИЙ attack-pattern з mitre-attack external_id,
не перевіряючи `revoked` / `x_mitre_deprecated`. Через це, наприклад, T1494
(revoked) потрапляв у KB і потім "верифікувався" як реальна техніка — це
підриває символьний шар верифікації.

Запуск:
    pytest tests/test_verifier_revoked.py -v
або як звичайний скрипт (без pytest):
    python tests/test_verifier_revoked.py
"""

import os
import sys
import tempfile

# Дозволяємо запуск і як `pytest`, і як `python tests/...` з кореня репо.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ВАЖЛИВО: shipper/config.py на імпорті створює глобальний `settings = Settings()`
# (pydantic-settings), який читає .env репозиторію (env_file=".env", за
# замовчуванням extra="forbid"). Локальний .env розробника містить легасі-ключі
# (API_ID/API_HASH для tg.py, OLLAMA_LLM_MODEL — до спліту на ollama_*/llm_*),
# яких немає у схемі Settings — це ламає `import shipper.*` з кореня репо. Це
# наявна проблема оточення, НЕ пов'язана з багом revoked/deprecated MITRE, що
# тестується тут. Щоб тест не залежав від вмісту локального .env, на час
# імпорту тимчасово йдемо у каталог без .env (і одразу повертаємось назад).
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cwd_before_import = os.getcwd()
with tempfile.TemporaryDirectory() as _tmpdir:
    os.chdir(_tmpdir)
    try:
        from shipper.verifier import MitreAttackKB, VerificationStatus  # noqa: E402
    finally:
        os.chdir(_cwd_before_import)


def _normal_attack_pattern():
    """Звичайний (не revoked, не deprecated) attack-pattern: T1059."""
    return {
        "type": "attack-pattern",
        "name": "Command and Scripting Interpreter",
        "description": "Adversaries may abuse command and script interpreters.",
        "external_references": [
            {"source_name": "mitre-attack", "external_id": "T1059"},
        ],
        "kill_chain_phases": [{"phase_name": "execution"}],
    }


def _revoked_attack_pattern():
    """Відкликана (revoked=True) техніка: T1494."""
    return {
        "type": "attack-pattern",
        "name": "Distributed Component Object Model",
        "description": "Deprecated/revoked technique.",
        "revoked": True,
        "external_references": [
            {"source_name": "mitre-attack", "external_id": "T1494"},
        ],
        "kill_chain_phases": [{"phase_name": "lateral-movement"}],
    }


def _deprecated_attack_pattern():
    """Deprecated (x_mitre_deprecated=True) техніка: T1000 (фейкова)."""
    return {
        "type": "attack-pattern",
        "name": "Some Deprecated Technique",
        "description": "Marked deprecated by MITRE.",
        "x_mitre_deprecated": True,
        "external_references": [
            {"source_name": "mitre-attack", "external_id": "T1000"},
        ],
        "kill_chain_phases": [{"phase_name": "execution"}],
    }


# --- _add_technique: revoked / deprecated не потрапляють у KB -------------
def test_add_technique_skips_revoked_and_deprecated():
    kb = MitreAttackKB()

    kb._add_technique(_normal_attack_pattern())
    kb._add_technique(_revoked_attack_pattern())
    kb._add_technique(_deprecated_attack_pattern())

    assert "T1059" in kb.techniques
    assert "T1494" not in kb.techniques, "revoked-техніка не має потрапляти у KB"
    assert "T1000" not in kb.techniques, "deprecated-техніка не має потрапляти у KB"
    assert len(kb.techniques) == 1


# --- lookup: revoked-техніка недосяжна через lookup ------------------------
def test_lookup_returns_none_for_revoked_technique():
    kb = MitreAttackKB()
    kb._add_technique(_revoked_attack_pattern())

    assert kb.lookup("T1494") is None


# --- _verify_ttp: revoked-техніка верифікується як NOT_FOUND ---------------
def test_verify_ttp_revoked_technique_is_not_found():
    from shipper.verifier import EntityVerifier

    kb = MitreAttackKB()
    kb._add_technique(_revoked_attack_pattern())

    # NVDClient тут не потрібен для _verify_ttp (синхронний, без httpx-клієнта).
    verifier = EntityVerifier(mitre_kb=kb, nvd_client=None)
    result = verifier._verify_ttp("T1494")

    assert result.status == VerificationStatus.NOT_FOUND
    assert result.technique_id == "T1494"


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
