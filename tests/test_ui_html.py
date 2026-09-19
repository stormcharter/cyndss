"""
Юніт-тести для ui_html.py — побудова HTML-плашок верифікованих CVE/TTP.

НАВІЩО ЦЕ ВЗАГАЛІ ВИНЕСЕНО В МОДУЛЬ. У ui.py плашки збирались конкатенацією
f-рядків і віддавались у `st.markdown(..., unsafe_allow_html=True)` БЕЗ
екранування, а самі ідентифікатори приходять з Manticore — тобто з ланцюга
«допис у Telegram → LLM-екстракція → верифікація». Сьогодні туди потрапляють
лише ті id, які підтвердив NVD/MITRE, тож реальної діри немає. Але це захист
через збіг обставин: варто змінитись верифікатору чи зʼявитись бекфілу — і
одного `"><script>` у полі досить, щоб вкрасти cookie довіреного пристрою
(HttpOnly для неї недосяжний, див. auth/streamlit_gate.py).

Тому: (1) екранування, (2) фільтр за формою ідентифікатора, (3) ці тести —
щоб екранування не «оптимізували» назад.

Запуск:
    pytest tests/test_ui_html.py -v
або:
    python tests/test_ui_html.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_html import build_verified_tags_html, split_ids  # noqa: E402

XSS = '"><script>alert(1)</script>'


# --- розбір рядка ідентифікаторів -------------------------------------------
def test_split_ids_handles_none_and_empty():
    assert split_ids(None) == []
    assert split_ids("") == []
    assert split_ids("   ") == []


def test_split_ids_trims_and_drops_empties():
    assert split_ids(" CVE-2024-1234 , ,CVE-2023-1 ") == ["CVE-2024-1234", "CVE-2023-1"]


# --- побудова HTML ----------------------------------------------------------
def test_valid_ids_are_rendered():
    html = build_verified_tags_html("CVE-2024-1234", "T1059")
    assert "CVE-2024-1234" in html
    assert "MITRE T1059" in html


def test_empty_input_gives_empty_string():
    assert build_verified_tags_html("", "") == ""
    assert build_verified_tags_html(None, None) == ""


def test_html_metacharacters_never_leave_raw():
    """Головний регрес: жодного сирого '<' від даних у результаті."""
    html = build_verified_tags_html(XSS, XSS)
    assert "<script>" not in html, html
    assert "alert(1)" not in html, html


def test_malformed_ids_are_dropped_entirely():
    """Фільтр за формою — дешевший і надійніший за саме лише екранування."""
    html = build_verified_tags_html("CVE-2024-1234," + XSS, "T1059," + XSS)
    assert "CVE-2024-1234" in html
    assert "T1059" in html
    assert "script" not in html.lower(), html


def test_lowercase_and_spaced_ids_are_normalised():
    html = build_verified_tags_html(" cve-2024-1234 ", " t1059.003 ")
    assert "CVE-2024-1234" in html
    assert "T1059.003" in html


def test_id_count_is_capped():
    """Сотні плашок ламають верстку — і це теж вектор (з боку контенту)."""
    many = ",".join("CVE-2024-%d" % i for i in range(200))
    html = build_verified_tags_html(many, "")
    assert html.count("<span") <= 32, html.count("<span")


def test_only_span_tags_are_produced():
    html = build_verified_tags_html("CVE-2024-1234", "T1059")
    for tag in ("<script", "<img", "<iframe", "onerror", "javascript:"):
        assert tag not in html.lower(), tag


# --- ui.py справді користується цим ------------------------------------------
def test_ui_uses_the_helper_instead_of_inline_concatenation():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "build_verified_tags_html" in src, "ui.py має викликати помічник"
    assert 'tags_html += f\'<span style="background-color:#ff4b4b' not in src, \
        "стара конкатенація без екранування повернулась"


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
