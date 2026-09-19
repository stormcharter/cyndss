"""Структурний тест: захисне плече (`twin_defense.py`) справді підключене до UI.

НАВІЩО САМЕ ТАКИЙ ТЕСТ. `tests/test_twin_defense.py` перевіряє логіку 20
тестами, а зв'язок з інтерфейсом аналітика не ловиться жодним юніт-тестом:
усі частини можуть бути справні, а `ui.py` так і не покликати жодну з них.
Саме так у цьому репозиторії вже з'являвся мертвий код — `bidirectional_path`
існував, був покритий тестами й не викликався НІЗВІДКИ, крім самих тестів.
Той самий структурний тест уже написаний для шару сценаріїв
(`tests/test_ui_scenarios_wiring.py`) — тут те саме для захисту.

`ui.py` не імпортується (Streamlit виконує сторінку на імпорті й вимагає
логіну), тому перевіряємо вихідний текст регулярними виразами — той самий
підхід, що в `tests/test_ui_scenarios_wiring.py`.

Запуск:
    pytest tests/test_ui_defense_wiring.py -v
або:
    python tests/test_ui_defense_wiring.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UI = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "ui.py"), encoding="utf-8").read()


def _scenario_mode_block() -> str:
    """Тіло `render_scenarios_mode` — те саме вирізання, що в
    `test_ui_scenarios_wiring.py`, аби перевіряти саме цю функцію, а не
    випадковий збіг рядка десь-інде у файлі."""
    block = UI[UI.find("def render_scenarios_mode"):]
    return block[:block.find("\n# ===")] if "\n# ===" in block else block


def test_ui_imports_twin_defense():
    assert re.search(r"from twin_defense import[^\n]*rank_defenses", UI), \
        "ui.py не імпортує rank_defenses з twin_defense"


def test_ui_calls_rank_defenses():
    assert "rank_defenses(" in UI, "ui.py не викликає rank_defenses"


def test_defense_call_is_inside_the_scenario_renderer():
    """Захист — не окремий режим, а продовження екрана сценаріїв: він
    рахується САМЕ за знайденими там шляхами."""
    block = _scenario_mode_block()
    assert "rank_defenses(" in block, \
        "виклик rank_defenses має бути всередині render_scenarios_mode"


def test_renderer_surfaces_excluded_by_constraints():
    """Набори, що розривають усі шляхи, але не проходять бюджет/строки/
    персонал/regulatory, не повинні тихо зникати — аналітик має їх бачити."""
    block = _scenario_mode_block()
    assert "excluded_by_constraints" in block, \
        "рендерер не показує набори, виключені обмеженнями"


def test_renderer_surfaces_unevaluable_requirements_as_a_third_state():
    """Набір, що проходить УСЕ оцінюване, але має вимогу (наприклад
    регуляторну), якої в моделі нічим підтвердити — не порушник і не
    «просто прохідний», а третій, окремий стан
    (`twin_defense.rank_defenses`, ключ `unevaluable_requirements`).
    Видавати брак даних за доведену невідповідність — саме та помилка, яку
    цей блок мав закрити."""
    block = _scenario_mode_block()
    assert "unevaluable_requirements" in block, \
        "рендерер не показує набори з вимогами, яких нічим підтвердити, окремо від порушників"


def test_renderer_surfaces_unbreakable_paths():
    """Шляхи, які не розриває жоден наявний засіб, — реальний залишковий
    ризик і мають бути названі явно, а не загублені серед знайдених наборів."""
    block = _scenario_mode_block()
    assert "unbreakable_path_indices" in block, \
        "рендерер не показує нерозривні шляхи"


def test_renderer_surfaces_truncated_flag():
    """Якщо перебір наборів обірвано межею кандидатів — це має бути видно,
    інакше часткова відповідь виглядає як повна."""
    block = _scenario_mode_block()
    assert "truncated" in block, \
        "рендерер не показує ознаку обірваного перебору"


def test_renderer_reports_when_the_model_has_no_control_nodes():
    """Якщо в моделі взагалі немає control-вузлів, `rank_defenses` мовчки
    поверне порожні списки — аналітик має побачити зрозуміле пояснення, а не
    порожній розділ, що виглядає як «усе розірвано» чи як баг."""
    block = _scenario_mode_block()
    assert re.search(r'kind\s*==\s*"control"', block), \
        "рендерер не перевіряє явно наявність control-вузлів у моделі"


def test_defense_block_is_shown_after_the_scenario_list():
    """Захист рахується за шляхами атаки — блок має йти ПІСЛЯ переліку
    сценаріїв, а не до нього (інакше даних для розрахунку ще нема)."""
    block = _scenario_mode_block()
    scenarios_pos = block.find('for scenario in result["scenarios"]')
    defense_pos = block.find("rank_defenses(")
    assert scenarios_pos != -1 and defense_pos != -1
    assert scenarios_pos < defense_pos, \
        "блок захисту має йти після переліку ранжованих сценаріїв"


def test_scenario_mode_does_not_break_the_two_existing_modes():
    assert "render_chat_mode(" in UI
    assert "render_feed_mode(" in UI
    assert re.search(r"render_scenarios_mode\(", UI)


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
