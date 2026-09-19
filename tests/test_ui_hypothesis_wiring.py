"""Структурний тест: двофронтовий обхід справді підключений до бойового шляху.

НАВІЩО САМЕ ТАКИЙ ТЕСТ. Логіку `hypotheses_from` перевіряє
`tests/test_semantic_reasoner.py`, розбір «від X до Y» —
`tests/test_chat_intents.py`. Але між ними лишався розрив, який юніт-тестом не
ловиться: обидві частини можуть бути справні, а `ui.py` — так і не покликати
двофронтовий пошук. Саме так і було до цієї правки: `bidirectional_path`
існував, був покритий тестами й не викликався НІЗВІДКИ, крім самих тестів.

`ui.py` не імпортується (Streamlit виконує сторінку на імпорті й вимагає
логіну), тому перевіряємо вихідний текст — той самий підхід, що в
`tests/test_ui_gate.py`.

Запуск:
    pytest tests/test_ui_hypothesis_wiring.py -v
або:
    python tests/test_ui_hypothesis_wiring.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI = os.path.join(ROOT, "ui.py")


def _source() -> str:
    with open(UI, encoding="utf-8") as fh:
        return fh.read()


def test_bidirectional_search_is_reachable_from_ui():
    """Головна перевірка файлу: двофронтовий пошук викликається з ui.py."""
    assert "hypotheses_from(" in _source(), (
        "ui.py не кличе hypotheses_from — двофронтовий обхід знову лишився "
        "мертвим кодом, а «двостороннє формування» — лише словами в статті"
    )


def test_unidirectional_search_is_still_reachable():
    """Режим без стартів має лишитись: не кожен запит має початкові свідчення."""
    assert re.search(r"\.hypotheses\(", _source()), \
        "ui.py більше не кличе hypotheses() — односторонній режим втрачено"


def test_intent_starts_are_passed_into_handler():
    """Розібрані старти мають доїжджати до обробника, а не губитись дорогою."""
    src = _source()
    call = re.search(r"handle_hypothesis_graph\((?![^)]*def)[^)]*\)", src, re.S)
    assert call, "не знайдено виклику handle_hypothesis_graph"
    assert "starts" in call.group(0), (
        f"старти не передаються в обробник: {call.group(0)!r}"
    )


def test_handler_accepts_starts_argument():
    assert re.search(r"def handle_hypothesis_graph\([^)]*starts", _source(), re.S), \
        "handle_hypothesis_graph не приймає starts"


def test_search_mode_is_reported_to_the_user():
    """Аналітик має бачити, який алгоритм відпрацював: односторонній і
    двофронтовий дають різну якість, і мовчки їх плутати не можна."""
    src = _source()
    assert '"mode"' in src or "'mode'" in src, "режим обходу не повертається назовні"
    assert "Двофронтовий обхід" in src, "режим обходу не показується в UI"


def test_goal_resolution_goes_through_reasoner():
    """Нормалізація вводу має бути одна — у резонера (вона покрита тестами),
    а не продубльована в ui.py, де тихо розійдеться зі словником графа."""
    src = _source()
    assert "resolve_node(" in src, "ui.py не використовує reasoner.resolve_node"
    assert 'goal_node = "MITRE " + goal_node.upper()' not in src, \
        "у ui.py лишилась стара копія нормалізації концептів"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001 — самописний раннер
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failed")
    sys.exit(1 if failures else 0)
