"""
Юніт-тести для chat_intents.py — розпізнавання наміру повідомлення в чаті ONYX.

МОТИВАЦІЯ. Раніше кожна дія мала власне поле й власну кнопку: окреме поле під
запит до агентів, окреме — під ціль гіпотез, окремі кнопки експорту графів.
Тепер вхід ОДИН (чат), тож із самого тексту треба зрозуміти, що робити:
побудувати семантичну мережу, побудувати граф гіпотез, показати граф звʼязків
чи просто відповісти агентами.

Модуль чистий (без streamlit/мережі) — тестується ізольовано.

Запуск:
    pytest tests/test_chat_intents.py -v
або як звичайний скрипт (без pytest):
    python tests/test_chat_intents.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_intents import (  # noqa: E402
    INTENT_ANALYSIS,
    INTENT_CTI_GRAPH,
    INTENT_HYPOTHESIS_GRAPH,
    INTENT_SEMANTIC_NETWORK,
    detect_intent,
)


def kind(text):
    return detect_intent(text)["kind"]


# --- семантична мережа понять ------------------------------------------------
def test_semantic_network_uk():
    assert kind("згенуй мені семантичну мережу понять") == INTENT_SEMANTIC_NETWORK


def test_semantic_network_uk_short():
    assert kind("побудуй мережу понять") == INTENT_SEMANTIC_NETWORK


def test_semantic_network_ru():
    assert kind("построй семантическую сеть понятий") == INTENT_SEMANTIC_NETWORK


def test_semantic_network_en():
    assert kind("build a semantic network for me") == INTENT_SEMANTIC_NETWORK


# --- граф гіпотез ------------------------------------------------------------
def test_hypothesis_graph_uk():
    assert kind("збудуй граф гіпотез") == INTENT_HYPOTHESIS_GRAPH


def test_hypothesis_graph_ru():
    assert kind("построй граф гипотез про ransomware") == INTENT_HYPOTHESIS_GRAPH


def test_hypothesis_graph_en():
    assert kind("build hypothesis graph") == INTENT_HYPOTHESIS_GRAPH


def test_hypothesis_wins_over_generic_graph_word():
    """«граф гіпотез» містить слово «граф» — не має падати у граф звʼязків."""
    assert kind("покажи граф гіпотез") == INTENT_HYPOTHESIS_GRAPH


# --- граф звʼязків CTI -------------------------------------------------------
def test_cti_graph_uk():
    assert kind("покажи граф звʼязків") == INTENT_CTI_GRAPH


def test_cti_graph_apostrophe_variants():
    """Апостроф в українській набирають трьома різними символами."""
    for apo in ("'", "’", "ʼ"):
        assert kind(f"побудуй граф зв{apo}язків") == INTENT_CTI_GRAPH, apo


def test_cti_graph_ru():
    assert kind("построй граф связей") == INTENT_CTI_GRAPH


# --- звичайне питання до агентів (дефолт) ------------------------------------
def test_plain_question_is_analysis():
    assert kind("розкажи про GitMCP Vulnerability") == INTENT_ANALYSIS


def test_plain_question_uk_is_analysis():
    assert kind("розкажи про останні вразливості") == INTENT_ANALYSIS


def test_question_mentioning_word_graph_alone_is_analysis():
    """Саме слово «граф» без уточнення — це не команда на візуалізацію."""
    assert kind("чи є в корпусі згадки про graph database") == INTENT_ANALYSIS


def test_empty_message_is_analysis():
    assert kind("") == INTENT_ANALYSIS
    assert kind(None) == INTENT_ANALYSIS


# --- витяг цілі для графа гіпотез --------------------------------------------
def test_goal_extracted_after_pro():
    res = detect_intent("збудуй граф гіпотез про ransomware")
    assert res["kind"] == INTENT_HYPOTHESIS_GRAPH
    assert res["goal"] == "ransomware"


def test_goal_extracted_after_dlya():
    assert detect_intent("побудуй граф гіпотез для exfiltration")["goal"] == "exfiltration"


def test_goal_extracted_after_about():
    assert detect_intent("build hypothesis graph about phishing")["goal"] == "phishing"


def test_goal_extracted_from_quotes():
    """Лапки мають пріоритет — так задають ціль із кількох слів."""
    res = detect_intent("збудуй граф гіпотез «supply chain»")
    assert res["goal"] == "supply chain", res


def test_goal_extracted_cve_identifier():
    res = detect_intent("граф гіпотез до CVE-2024-1234")
    assert res["goal"] == "CVE-2024-1234", res


def test_goal_is_none_when_not_specified():
    """Без цілі — None; UI має спитати або підставити свою (не вигадувати тут)."""
    assert detect_intent("збудуй граф гіпотез")["goal"] is None


def test_goal_strips_trailing_punctuation():
    assert detect_intent("збудуй граф гіпотез про ransomware!")["goal"] == "ransomware"


def test_goal_preserves_case_of_identifiers():
    """Ціль іде у граф як ідентифікатор вузла — регістр CVE/MITRE не чіпаємо."""
    assert detect_intent("граф гіпотез про MITRE T1059")["goal"] == "MITRE T1059"


# --- тема для ГРАФОВИХ команд ------------------------------------------------
# ВАЖЛИВО. У команді «згенуй мені семантичну мережу понять» слова «семантичну
# мережу понять» — це САМА КОМАНДА, а не тема пошуку. Якщо віддати весь текст
# у пошук, система знайде документи, де згадуються «мережа» й «поняття», і
# побудує мережу по них — тобто по випадковому зрізу корпусу замість усього
# корпусу. Тому для графових намірів тема витягується так само, як ціль
# гіпотез: після маркера («про», «для», «about») або з лапок. Немає теми —
# None, і UI бере корпус цілком.
def test_semantic_network_without_topic_has_no_goal():
    res = detect_intent("згенуй мені семантичну мережу понять")
    assert res["kind"] == INTENT_SEMANTIC_NETWORK
    assert res["goal"] is None, res


def test_semantic_network_topic_extracted():
    res = detect_intent("побудуй семантичну мережу понять про Cisco")
    assert res["kind"] == INTENT_SEMANTIC_NETWORK
    assert res["goal"] == "Cisco", res


def test_cti_graph_without_topic_has_no_goal():
    assert detect_intent("покажи граф звʼязків")["goal"] is None


def test_cti_graph_topic_extracted():
    res = detect_intent("покажи граф звʼязків про ransomware")
    assert res["kind"] == INTENT_CTI_GRAPH
    assert res["goal"] == "ransomware", res


# --- запит зберігається у результаті -----------------------------------------
def test_original_text_is_returned():
    res = detect_intent("  розкажи про Cisco  ")
    assert res["query"] == "розкажи про Cisco"


def test_result_always_has_all_keys():
    for msg in ("", "розкажи про Cisco", "збудуй граф гіпотез", "мережа понять"):
        res = detect_intent(msg)
        assert set(res) >= {"kind", "goal", "query"}, res



# --- стартові концепти: «від X до Y» --------------------------------------
def test_start_absent_when_only_goal_given():
    """Стара форма має працювати як і раніше: старт не вигадуємо."""
    r = detect_intent("збудуй граф гіпотез про ransomware")
    assert r["kind"] == INTENT_HYPOTHESIS_GRAPH
    assert r["goal"] == "ransomware"
    assert r["starts"] == []


def test_start_extracted_from_vid_do():
    r = detect_intent("збудуй граф гіпотез від фішингу до ransomware")
    assert r["kind"] == INTENT_HYPOTHESIS_GRAPH
    assert r["starts"] == ["фішингу"]
    assert r["goal"] == "ransomware"


def test_start_supports_arrow_instead_of_do():
    r = detect_intent("граф гіпотез від CVE-2024-1234 → ransomware")
    assert r["starts"] == ["CVE-2024-1234"]
    assert r["goal"] == "ransomware"


def test_start_supports_ascii_arrow():
    r = detect_intent("граф гіпотез від phishing -> ransomware")
    assert r["starts"] == ["phishing"]
    assert r["goal"] == "ransomware"


def test_several_starts_are_split():
    r = detect_intent("граф гіпотез від фішингу, CVE-2024-1234 та lockbit до ransomware")
    assert r["starts"] == ["фішингу", "CVE-2024-1234", "lockbit"]
    assert r["goal"] == "ransomware"


def test_english_from_to():
    r = detect_intent("hypothesis graph from phishing to ransomware")
    assert r["kind"] == INTENT_HYPOTHESIS_GRAPH
    assert r["starts"] == ["phishing"]
    assert r["goal"] == "ransomware"


def test_start_case_is_preserved():
    """Старт іде у граф як ідентифікатор вузла — регістр CVE/MITRE не чіпаємо."""
    r = detect_intent("граф гіпотез від MITRE T1566 до ransomware")
    assert r["starts"] == ["MITRE T1566"]


def test_start_not_extracted_for_non_hypothesis_intents():
    r = detect_intent("розкажи про перехід від фішингу до шифрування")
    assert r["kind"] == INTENT_ANALYSIS
    assert r["starts"] == []


def test_vid_without_goal_yields_no_start():
    """«від X» без цілі — команда неповна; вигадувати ціль тут не можна."""
    r = detect_intent("граф гіпотез від фішингу")
    assert r["starts"] == []

# --- ручний запуск без pytest -----------------------------------------------
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
