"""Структурний тест: режим «Від проблеми до розв'язку» справді підключений до ONYX.

Логіку перевіряє `tests/test_semantic_bridge.py`, а тут — розрив, який юніт-тестом
не ловиться: усі частини можуть бути справні, а `ui.py` так і не покликати
жодну з них. `ui.py` не імпортується (Streamlit виконує сторінку на імпорті й
вимагає логіну), тому перевіряємо вихідний текст — той самий підхід, що в
`tests/test_ui_scenarios_wiring.py`.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
UI = open(os.path.join(ROOT, "ui.py"), encoding="utf-8").read()
LAB = open(os.path.join(ROOT, "scenario_lab.py"), encoding="utf-8").read()


def test_mode_is_in_the_switch():
    switch = re.search(r"app_mode = st\.sidebar\.radio\((.*?)\)\n", UI, re.S)
    assert switch and "Від проблеми до розв'язку" in switch.group(1)


def test_dispatcher_renders_the_shared_page():
    assert re.search(r"from solution_ui import[^\n]*render_solution_mode", UI)
    assert re.search(r"elif IS_SOLUTION:\s*\n\s*render_solution_mode\(solution_env\(\)\)", UI)


def test_env_is_defined_before_the_dispatcher():
    definition = UI.find("def solution_env")
    dispatcher = UI.find("ДИСПЕТЧЕР РЕЖИМІВ")
    assert definition != -1 and dispatcher != -1 and definition < dispatcher


def test_onyx_uses_its_own_backends_corpus_and_dataset():
    env = UI[UI.find("def solution_env"):UI.find("ДИСПЕТЧЕР РЕЖИМІВ")]
    assert "available_backends()" in env
    assert "retrieve=_solution_retrieve" in env
    assert '"corporate_networks"' in env


def test_generate_raises_instead_of_hiding_errors():
    """Метод розбирає JSON; рядок «❌ Помилка…» замість винятку зробив би фронт
    мовчки порожнім."""
    body = UI[UI.find("def _solution_generate"):UI.find("def _solution_embed")]
    assert "raise_for_status()" in body and "except" not in body


def test_lab_and_onyx_share_one_page():
    assert "render_solution_mode(" in LAB
    assert "st.subheader(\"8." not in LAB   # екран не продубльовано у стенді
