"""
Структурний тест: ворота автентифікації справді стоять ПЕРЕД усім іншим.

НАВІЩО САМЕ ТАКИЙ ТЕСТ. Логіку входу перевіряють tests/test_auth_store.py та
сусіди; тут перевіряється те, що юніт-тестом не ловиться взагалі — МІСЦЕ
виклику. Найдешевший спосіб зламати автентифікацію в Streamlit — не зламати
перевірку, а посунути її нижче: скрипт виконується згори вниз, тож будь-який
віджет із даними ДО `require_login()` встигне намалюватись анонімному
відвідувачу, і `st.stop()` вже нічого не врятує.

Другий блок — про tools/create_user.py: пароль не має приймати аргументом
командного рядка (аргументи видно в `ps` іншим користувачам спільного бокса й
вони осідають у ~/.bash_history).

Запуск:
    pytest tests/test_ui_gate.py -v
або:
    python tests/test_ui_gate.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Віджети, які щось показують користувачу. set_page_config сюди не входить —
# він нічого не розкриває і має стояти найпершим за вимогою самого Streamlit.
_RENDERING_CALL = re.compile(
    r"^\s*st\.(title|header|subheader|markdown|write|dataframe|table|tabs|"
    r"columns|metric|plotly_chart|sidebar|expander|caption|image)\b"
)
_GATE_CALL = re.compile(r"^CURRENT_USER\s*=\s*require_login\(\)")


def _ui_lines():
    with open(os.path.join(ROOT, "ui.py"), encoding="utf-8") as fh:
        return fh.read().splitlines()


def _gate_line(lines):
    for i, line in enumerate(lines):
        if _GATE_CALL.match(line):
            return i
    return None


def _first_render_line(lines):
    for i, line in enumerate(lines):
        if _RENDERING_CALL.match(line):
            return i
    return None


# --- ворота в ui.py ---------------------------------------------------------
def test_gate_is_called_at_module_level():
    """Не всередині функції чи if — інакше виклик може не відбутись зовсім."""
    assert _gate_line(_ui_lines()) is not None, \
        "у ui.py немає модульного виклику CURRENT_USER = require_login()"


def test_gate_precedes_any_rendering():
    lines = _ui_lines()
    gate = _gate_line(lines)
    first_render = _first_render_line(lines)
    assert gate is not None
    if first_render is not None:
        assert gate < first_render, (
            "require_login() на рядку %d, а віджет уже на рядку %d: "
            "анонім побачить дані до перевірки" % (gate + 1, first_render + 1)
        )


def test_gate_check_would_catch_a_moved_call():
    """Перевірка самої перевірки: на зіпсованому файлі вона має спрацювати."""
    broken = [
        "import streamlit as st",
        "st.title('Дані')",
        "CURRENT_USER = require_login()",
    ]
    assert _gate_line(broken) > _first_render_line(broken)


def test_ui_imports_the_gate():
    lines = _ui_lines()
    assert any(l.startswith("from auth.streamlit_gate import require_login")
               for l in lines), "ui.py не імпортує ворота"


# --- CLI керування користувачами -------------------------------------------
def test_create_user_does_not_take_password_as_argument():
    path = os.path.join(ROOT, "tools", "create_user.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "getpass" in src, "пароль має вводитись інтерактивно"
    assert 'add_argument("--password"' not in src, \
        "пароль в argv видно в ps і лишається в історії shell"
    assert "add_argument('--password'" not in src


def test_create_user_validates_policy_before_writing():
    path = os.path.join(ROOT, "tools", "create_user.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "validate_password" in src


# --- ручний запуск без pytest ----------------------------------------------
if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS " + name)
            passed += 1
        except Exception:
            print("FAIL " + name)
            traceback.print_exc()
            failed += 1
    print("\n%d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)
