"""
Символьний збір ідентифікаторів прямо з тексту.

НАВІЩО. Гейт заземлення (`shipper/grounding.py`) уміє лише ВІДКИДАТИ те, що
назвала модель. Але вимірювання на першій порції веб-корпусу (45 статей
BleepingComputer / TheHackersNews, 07.09.2026) показало зворотну проблему:
16 статей із 45 містять CVE ДОСЛІВНО (одна — 95 штук у щотижневому огляді),
а модель називає одиниці. Тобто ми втрачаємо саме той матеріал, заради якого
веб-джерела й вмикали.

Для ідентифікатора, який стоїть у тексті дослівно, LLM взагалі не потрібна:
регулярка знаходить його детерміновано, NVD/MITRE підтверджують існування й
дають авторитетний опис. Це і є символьна половина «нейро-символічного» —
без жодного шансу на галюцинацію.

Запуск:
    python tests/test_literal_harvest.py
або pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shipper.grounding import harvest_literal_ids  # noqa: E402


def test_finds_cve_in_plain_text():
    cves, ttps = harvest_literal_ids("Патч закриває CVE-2026-13181 у RouterOS.")
    assert cves == ["CVE-2026-13181"]
    assert ttps == []


def test_uppercases_and_preserves_text_order():
    cves, _ = harvest_literal_ids("спершу cve-2026-0002, потім CVE-2026-0001")
    assert cves == ["CVE-2026-0002", "CVE-2026-0001"]


def test_deduplicates_repeated_mentions():
    cves, _ = harvest_literal_ids("CVE-2026-13181 ... знову CVE-2026-13181")
    assert cves == ["CVE-2026-13181"]


def test_survives_unicode_dashes():
    cves, _ = harvest_literal_ids("виправлено CVE‑2026–13181 у прошивці")
    assert cves == ["CVE-2026-13181"]


def test_finds_technique_ids_including_subtechniques():
    _, ttps = harvest_literal_ids("ланцюг T1566 -> T1078.004 -> ескалація")
    assert ttps == ["T1566", "T1078.004"]


def test_does_not_pick_up_glued_tokens():
    cves, ttps = harvest_literal_ids("артикул CVE-2026-13181X і тег T1190bis")
    assert cves == []
    assert ttps == []


def test_respects_the_per_document_cap():
    # Щотижневі огляди перелічують десятки CVE. Без стелі один документ
    # перетворюється на десятки зовнішніх викликів у NVD.
    text = " ".join(f"CVE-2026-{1000 + i}" for i in range(40))
    cves, _ = harvest_literal_ids(text, limit=20)
    assert len(cves) == 20
    assert cves[0] == "CVE-2026-1000"


def test_cap_applies_to_each_kind_separately():
    text = "CVE-2026-1000 CVE-2026-1001 T1566 T1078"
    cves, ttps = harvest_literal_ids(text, limit=1)
    assert cves == ["CVE-2026-1000"]
    assert ttps == ["T1566"]


def test_empty_text_yields_nothing():
    assert harvest_literal_ids("") == ([], [])
    assert harvest_literal_ids(None) == ([], [])


def test_text_without_identifiers_yields_nothing():
    assert harvest_literal_ids("Добірка сервісів для генерації логотипів") == ([], [])


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
