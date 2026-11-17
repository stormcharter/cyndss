"""Структурний тест: режим «Від проблеми до розв'язку» справді підключений до ONYX.

Логіку перевіряє `tests/test_semantic_bridge.py`, а тут — розрив, який юніт-тестом
не ловиться: усі частини можуть бути справні, а `ui.py` так і не покликати
жодну з них. `ui.py` не імпортується (Streamlit виконує сторінку на імпорті й
вимагає логіну), тому перевіряємо вихідний текст — той самий підхід, що в
`tests/test_ui_scenarios_wiring.py`.
"""

import json
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


# --- опора на корпус: два етапи замість голого KNN -----------------------------------------
# Чому: збережені вектори — це сирий оригінал (uk/ru), англійський запит ніколи не
# проходить поріг KNN. Деталі й виміри — у concept_support.py.

SOLUTION_UI = open(os.path.join(ROOT, "solution_ui.py"), encoding="utf-8").read()


def _retrieval_section() -> str:
    start = UI.find("def _solution_candidates")
    assert start != -1, "немає _solution_candidates у ui.py"
    end = UI.find("def solution_env")
    assert end > start
    return UI[start:end]


def test_onyx_solution_retrieval_goes_through_support_retriever():
    assert re.search(r"from concept_support import[^\n]*SupportRetriever", UI)
    section = _retrieval_section()
    assert "SupportRetriever(_solution_candidates, _solution_embed)" in section
    assert re.search(r"def _solution_retrieve\(query: str\) -> list\[str\]:", section)
    assert "_solution_support()(query)" in section


def test_onyx_candidates_come_from_title_and_summary_only():
    section = _retrieval_section()
    assert "build_or_match_clause(" in section
    assert 'fields=("title", "summary")' in section
    assert "def _solution_candidates(query: str, k: int) -> list[dict]:" in section


def test_onyx_solution_retrieval_no_longer_uses_knn():
    section = _retrieval_section()
    assert "knn(" not in section
    assert "KNN_MAX_DISTANCE" not in section


def test_support_retriever_is_one_per_process():
    """Інакше кеш векторів документів помирав би на кожному rerun Streamlit."""
    assert re.search(r"@st\.cache_resource\([^)]*\)\s*\ndef _solution_support\(", UI)


def test_lab_uses_the_same_selection_as_onyx():
    """Стенд відбирає через `concept_support.corpus_retriever` — той самий
    SupportRetriever і поріг, що ONYX. Саму поведінку (поріг, кеш на весь корпус)
    перевіряють поведінкові тести в test_concept_support.py; тут — лише що стенд її
    справді кличе, а старий безумовний top-2 з префіксами прибрано."""
    assert re.search(r"from concept_support import[^\n]*corpus_retriever", LAB)
    assert "corpus_retriever(documents, embed)" in LAB
    assert "search_query:" not in LAB


def test_support_scheme_goes_into_cache_key_only_with_rag():
    """Ключ мережі без RAG лишається побайтово тим самим, що й до зміни (старі
    кеші чинні); з RAG залежить від способу відбору фрагментів."""
    import hashlib

    from concept_support import SUPPORT_SCHEME
    from solution_ui import _cache_key

    case = {"problem": "П", "solution": "Р"}

    def old_key(use_rag):
        return hashlib.sha1(json.dumps(
            ["П", "Р", "ctx", 3, 2, "model", bool(use_rag)],
            ensure_ascii=False).encode()).hexdigest()[:16]

    assert _cache_key(case, "ctx", 3, 2, "model", False) == old_key(False)
    assert _cache_key(case, "ctx", 3, 2, "model", True) != old_key(True)
    expected = hashlib.sha1(json.dumps(
        ["П", "Р", "ctx", 3, 2, "model", True, SUPPORT_SCHEME],
        ensure_ascii=False).encode()).hexdigest()[:16]
    assert _cache_key(case, "ctx", 3, 2, "model", True) == expected
    assert "_cache_key(case," in SOLUTION_UI and "SUPPORT_SCHEME" in SOLUTION_UI


def test_support_message_is_rendered_with_explicit_levels():
    assert "support_message(net.support)" in SOLUTION_UI
    for mapping in ('"info": st.info', '"warning": st.warning', '"error": st.error'):
        assert mapping in SOLUTION_UI
    assert SOLUTION_UI.find("support_message(net.support)") < SOLUTION_UI.find("for line in net.log")
    assert SOLUTION_UI.find("m[3].metric") < SOLUTION_UI.find("support_message(net.support)")
    assert "getattr(st," not in SOLUTION_UI


def test_retrieve_context_comment_tells_the_truth_about_embeddings():
    """Старий коментар стверджував, що ембединги корпусу рахувались по перекладеному
    тексту — неправда: шиппер ембедив сирий оригінал (uk/ru). Відтоді шиппер рахує
    вектор від англійського «title. summary», тож коментар мусить посилатися на
    джерело істини про це — `embedding_text.py` — і на перерахунок старого індексу,
    і не повторювати хибне твердження."""
    assert "здебільшого проходив через той самий переклад" not in UI
    start = UI.find("vector = get_query_embedding(translated or query_text)")
    comment = UI[start - 900:start]
    assert "оригінал" in comment and "embedding_text" in comment
    assert "reembed_index" in comment
