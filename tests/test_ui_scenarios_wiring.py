"""Структурний тест: орієнтований граф справді підключений до UI.

НАВІЩО САМЕ ТАКИЙ ТЕСТ. Логіку перевіряють `tests/test_twin_bridge.py` і
`tests/test_twin_scenarios.py`, а вимір — експеримент. Але між ними лишається
розрив, який юніт-тестом не ловиться: усі частини можуть бути справні, а `ui.py`
так і не покликати жодну з них. Саме так уже було з `bidirectional_path` — він
існував, був покритий тестами й не викликався НІЗВІДКИ, крім самих тестів.

`ui.py` не імпортується (Streamlit виконує сторінку на імпорті й вимагає
логіну), тому перевіряємо вихідний текст — той самий підхід, що в
`tests/test_ui_gate.py` і `tests/test_ui_hypothesis_wiring.py`.

Запуск:
    pytest tests/test_ui_scenarios_wiring.py -v
або:
    python tests/test_ui_scenarios_wiring.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UI = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "ui.py"), encoding="utf-8").read()


def test_ui_imports_the_scenario_layer():
    assert re.search(r"from twin_scenarios import[^\n]*build_scenarios", UI), \
        "ui.py не імпортує бойовий шар сценаріїв"


def test_ui_calls_build_scenarios():
    assert "build_scenarios(" in UI, "ui.py не викликає build_scenarios"


def test_ui_parses_the_infrastructure_model():
    assert "parse_infrastructure_model(" in UI


def test_scenario_mode_is_reachable_from_the_mode_switch():
    """Режим має бути в перемикачі, інакше він недосяжний користувачу."""
    switch = re.search(r"app_mode = st\.sidebar\.radio\((.*?)\)\n", UI, re.S)
    assert switch, "не знайдено перемикач режимів"
    assert "Сценарі" in switch.group(1), switch.group(1)


def test_dispatcher_routes_to_the_scenario_renderer():
    assert re.search(r"render_scenarios_mode\(", UI), \
        "диспетчер режимів не викликає рендерер сценаріїв"


def test_renderer_is_defined_before_the_dispatcher():
    definition = UI.find("def render_scenarios_mode")
    dispatcher = UI.find("ДИСПЕТЧЕР РЕЖИМІВ")
    assert definition != -1 and dispatcher != -1
    assert definition < dispatcher, "рендерер визначено після диспетчера"


def test_named_transitions_are_shown_not_just_nodes():
    """Сенс орієнтованого графа — саме НАЗВА переходу; якщо UI її не показує,
    користувач бачить те саме, що й у неорієнтованій мережі."""
    block = UI[UI.find("def render_scenarios_mode"):]
    block = block[:block.find("\n# ===")] if "\n# ===" in block else block
    assert 'step["relation"]' in block or "['relation']" in block, \
        "рендерер не виводить назви переходів"


def test_integral_score_is_shown():
    block = UI[UI.find("def render_scenarios_mode"):]
    assert '"j"' in block or "['j']" in block, "рендерер не показує інтегральну оцінку"


def test_corpus_evidence_is_surfaced():
    block = UI[UI.find("def render_scenarios_mode"):]
    assert "evidence" in block, "рендерер не показує доказовість із корпусу"


def test_scenario_mode_does_not_break_the_two_existing_modes():
    assert "render_chat_mode(" in UI
    assert "render_feed_mode(" in UI


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
