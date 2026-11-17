"""
Юніт-тести для concept_support.py — опора розкриття понять на корпус.

Пошук і ембедер підмінено заглушками з ручними векторами, тому відстані точні:
перевіряється не якість nomic, а правило відбору (BM25-кандидати -> косинус
«англійський запит vs title. summary» < порогу) та побутові властивості —
кеш векторів, порожні входи, повідомлення для користувача.

Запуск:
    pytest tests/test_concept_support.py -v
або як звичайний скрипт (без pytest):
    python tests/test_concept_support.py
"""

import math
import os
import sys

# Дозволяємо запуск і як `pytest`, і як `python tests/...` з кореня репо.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_support import (  # noqa: E402
    DOCUMENT_CHARS,
    MAX_CACHED_VECTORS,
    FRAGMENT_CHARS,
    SUPPORT_CANDIDATES,
    SUPPORT_MAX_DISTANCE,
    SUPPORT_SCHEME,
    SupportRetriever,
    corpus_retriever,
    document_text,
    support_message,
)

QUERY_VEC = [1.0, 0.0]


def vec_at(distance: float) -> list[float]:
    """Одиничний вектор, що лежить від запиту [1, 0] на косинусній відстані `distance`."""
    cos = 1.0 - distance
    return [cos, math.sqrt(1.0 - cos * cos)]


class FakeEmbed:
    """Ембедер за словником «текст -> вектор»; невідомий текст ортогональний запиту
    (відстань 1). Пам'ятає кожен виклик."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors.get(t, [0.0, 1.0]) for t in texts]

    @property
    def embedded(self) -> list[str]:
        return [t for call in self.calls for t in call]


def fake_search(docs: list[dict]):
    calls: list[tuple[str, int]] = []

    def search(query: str, k: int) -> list[dict]:
        calls.append((query, k))
        return docs
    search.calls = calls
    return search


def doc(i: int, title: str = "", summary: str = "", **extra) -> dict:
    return {"id": i, "title": title, "summary": summary, **extra}


def retriever_for(docs, distances: dict[str, float], query="ransomware", **kwargs):
    """Ретривер, де вектор тексту «title. summary» стоїть на заданій відстані від запиту."""
    vectors = {query: QUERY_VEC, **{text: vec_at(d) for text, d in distances.items()}}
    embed = FakeEmbed(vectors)
    search = fake_search(docs)
    return SupportRetriever(search, embed, **kwargs), search, embed


# --- document_text --------------------------------------------------------
def test_document_text_joins_title_and_summary():
    assert document_text({"title": " Alpha ", "summary": " beta "}) == "Alpha. beta"


def test_document_text_skips_empty_parts():
    assert document_text({"title": "Alpha", "summary": "  "}) == "Alpha"
    assert document_text({"title": None, "summary": "beta"}) == "beta"


def test_document_text_falls_back_to_content():
    assert document_text({"title": "", "summary": "", "content": "  повний текст "}) == "повний текст"


def test_document_text_empty_when_nothing():
    assert document_text({}) == ""
    assert document_text({"title": "", "summary": None, "content": " "}) == ""


def test_document_text_is_bounded_for_the_embedder():
    """Текст іде в ембедер цілком: без стелі довгий документ завантаженого в стенд
    корпусу впирається в контекст моделі. Старий стенд різав на 1500 — та сама межа.
    У корпусі ONYX «title. summary» не довші за ~1000, тож їх стеля не зачіпає."""
    assert len(document_text({"content": "к" * 5000})) == DOCUMENT_CHARS == 1500
    assert len(document_text({"title": "T", "summary": "s" * 5000})) == DOCUMENT_CHARS
    assert document_text({"title": "T", "summary": "short"}) == "T. short"


def test_scheme_is_a_stable_identifier():
    """Рядок іде в ключ кешу мереж: якщо метод відбору зміниться, його треба змінити."""
    assert SUPPORT_SCHEME == "bm25:title,summary -> nomic:title.summary < 0.30"
    assert SUPPORT_MAX_DISTANCE == 0.30


# --- відбір за порогом ---------------------------------------------------
def test_keeps_only_fragments_below_threshold():
    docs = [doc(1, "Near", "doc"), doc(2, "Far", "doc")]
    retriever, _search, _embed = retriever_for(docs, {"Near. doc": 0.10, "Far. doc": 0.45})
    assert retriever("ransomware") == ["Near. doc"]


def test_returns_nothing_when_all_candidates_are_far():
    docs = [doc(1, "A", "x"), doc(2, "B", "y")]
    retriever, _s, _e = retriever_for(docs, {"A. x": 0.53, "B. y": 0.58})
    assert retriever("ransomware") == []


def test_threshold_is_strict_and_configurable():
    docs = [doc(1, "A", "x"), doc(2, "B", "y")]
    retriever, _s, _e = retriever_for(docs, {"A. x": 0.20, "B. y": 0.40}, max_distance=0.50)
    assert retriever("ransomware") == ["A. x", "B. y"]


def test_at_most_limit_ordered_by_distance():
    docs = [doc(1, "C", "x"), doc(2, "A", "x"), doc(3, "B", "x")]
    retriever, _s, _e = retriever_for(docs, {"C. x": 0.25, "A. x": 0.05, "B. x": 0.15})
    assert retriever("ransomware") == ["A. x", "B. x"]          # limit за замовчуванням 2
    wide, _s, _e = retriever_for(docs, {"C. x": 0.25, "A. x": 0.05, "B. x": 0.15}, limit=3)
    assert wide("ransomware") == ["A. x", "B. x", "C. x"]


def test_equal_distance_keeps_candidate_order():
    docs = [doc(1, "First", "x"), doc(2, "Second", "x"), doc(3, "Third", "x")]
    same = vec_at(0.1)
    embed = FakeEmbed({"ransomware": QUERY_VEC, "First. x": same, "Second. x": same,
                       "Third. x": same})
    retriever = SupportRetriever(fake_search(docs), embed)
    assert retriever("ransomware") == ["First. x", "Second. x"]


def test_compares_title_and_summary_not_content():
    """Документ, чий content збігається із запитом, а title/summary — ні, відхиляється:
    content у корпусі — оригінал (uk/ru), який з англійським запитом не порівнюється."""
    docs = [doc(1, "Unrelated", "topic", content="ransomware")]
    retriever, _s, embed = retriever_for(docs, {"Unrelated. topic": 0.80})
    assert retriever("ransomware") == []
    assert "Unrelated. topic" in embed.embedded
    assert "ransomware" not in embed.embedded[1:]       # content не ембедили


def test_content_is_the_fallback_when_title_and_summary_empty():
    docs = [doc(1, "", "", content="  ransomware report ")]
    retriever, _s, embed = retriever_for(docs, {"ransomware report": 0.05})
    assert retriever("ransomware") == ["ransomware report"]
    assert "ransomware report" in embed.embedded


def test_search_receives_query_and_candidate_count():
    retriever, search, _e = retriever_for([doc(1, "A", "x")], {"A. x": 0.1})
    retriever("ransomware")
    assert search.calls == [("ransomware", SUPPORT_CANDIDATES)]
    custom, search2, _e = retriever_for([doc(1, "A", "x")], {"A. x": 0.1}, candidates=7)
    custom("ransomware")
    assert search2.calls == [("ransomware", 7)]


def test_fragment_is_truncated():
    long_summary = "s" * 900
    text = f"T. {long_summary}"
    retriever, _s, _e = retriever_for([doc(1, "T", long_summary)], {text: 0.05})
    out = retriever("ransomware")
    assert len(out) == 1 and len(out[0]) == FRAGMENT_CHARS == 500
    assert out[0] == text[:FRAGMENT_CHARS]


def test_identical_fragments_are_not_repeated():
    """Два документи з різними id, але однаковим «title. summary» (передрук новини
    іншим каналом) не повинні займати обидва слоти: повтор у промпті нічого не дає,
    а витісняє наступний справді інший фрагмент."""
    docs = [doc(1, "A", "x"), doc(2, "A", "x"), doc(3, "B", "x")]
    retriever, _s, _e = retriever_for(docs, {"A. x": 0.05, "B. x": 0.15})
    assert retriever("ransomware") == ["A. x", "B. x"]


def test_zero_limit_returns_nothing():
    retriever, _s, _e = retriever_for([doc(1, "A", "x")], {"A. x": 0.05}, limit=0)
    assert retriever("ransomware") == []


# --- порожні входи --------------------------------------------------------
def test_blank_query_calls_nothing():
    retriever, search, embed = retriever_for([doc(1, "A", "x")], {"A. x": 0.1})
    assert retriever("") == [] and retriever("   \n") == []
    assert search.calls == [] and embed.calls == []


def test_no_candidates_means_no_embed_call():
    retriever, search, embed = retriever_for([], {})
    assert retriever("ransomware") == []
    assert len(search.calls) == 1 and embed.calls == []


def test_docs_without_any_text_are_dropped_before_embedding():
    docs = [doc(1, "", "", content=""), doc(2, None, None)]
    retriever, _s, embed = retriever_for(docs, {})
    assert retriever("ransomware") == []
    assert embed.calls == []


# --- кеш векторів ----------------------------------------------------------
def test_query_and_missing_docs_go_in_one_embed_call_query_first():
    docs = [doc(1, "A", "x"), doc(2, "B", "y")]
    retriever, _s, embed = retriever_for(docs, {"A. x": 0.1, "B. y": 0.2})
    retriever("ransomware")
    assert embed.calls == [["ransomware", "A. x", "B. y"]]


def test_doc_vectors_are_cached_across_queries():
    docs = [doc(1, "A", "x"), doc(2, "B", "y")]
    vectors = {"q1": QUERY_VEC, "q2": QUERY_VEC, "A. x": vec_at(0.1), "B. y": vec_at(0.2)}
    embed = FakeEmbed(vectors)
    retriever = SupportRetriever(fake_search(docs), embed)
    first = retriever("q1")
    second = retriever("q2")
    assert first == second == ["A. x", "B. y"]
    assert embed.calls == [["q1", "A. x", "B. y"], ["q2"]]
    assert embed.embedded.count("A. x") == embed.embedded.count("B. y") == 1


def test_partially_cached_candidates_embed_only_new_texts():
    vectors = {"q1": QUERY_VEC, "q2": QUERY_VEC, "A. x": vec_at(0.1), "B. y": vec_at(0.2)}
    embed = FakeEmbed(vectors)
    pool = [[doc(1, "A", "x")], [doc(1, "A", "x"), doc(2, "B", "y")]]
    retriever = SupportRetriever(lambda _q, _k: pool.pop(0), embed)
    retriever("q1")
    retriever("q2")
    assert embed.calls == [["q1", "A. x"], ["q2", "B. y"]]


def test_docs_without_id_are_cached_by_text():
    vectors = {"q1": QUERY_VEC, "q2": QUERY_VEC, "A. x": vec_at(0.1)}
    embed = FakeEmbed(vectors)
    retriever = SupportRetriever(lambda _q, _k: [{"title": "A", "summary": "x"}], embed)
    retriever("q1")
    retriever("q2")
    assert embed.embedded.count("A. x") == 1


def test_cache_is_bounded():
    vectors = {"q": QUERY_VEC, "A. x": vec_at(0.1), "B. x": vec_at(0.1),
               "C. x": vec_at(0.1), "D. x": vec_at(0.1)}
    embed = FakeEmbed(vectors)
    pool = [[doc(1, "A", "x"), doc(2, "B", "x")],
            [doc(3, "C", "x"), doc(4, "D", "x")],
            [doc(1, "A", "x"), doc(2, "B", "x")]]
    retriever = SupportRetriever(lambda _q, _k: pool.pop(0), embed, max_cached=2)
    assert retriever("q") == ["A. x", "B. x"]
    assert retriever("q") == ["C. x", "D. x"]
    assert len(retriever._cache) <= 2
    assert retriever("q") == ["A. x", "B. x"]
    assert len(retriever._cache) <= 2
    # Після скидання A і B довелося порахувати наново: кеш справді обмежений.
    assert embed.embedded.count("A. x") == 2


def test_cache_larger_than_bound_in_one_call_still_respects_bound():
    docs = [doc(i, f"T{i}", "x") for i in range(1, 5)]
    embed = FakeEmbed({"q": QUERY_VEC, **{f"T{i}. x": vec_at(0.1) for i in range(1, 5)}})
    retriever = SupportRetriever(lambda _q, _k: docs, embed, max_cached=2)
    assert retriever("q") == ["T1. x", "T2. x"]
    assert len(retriever._cache) <= 2


def test_concurrent_cache_clear_never_raises_keyerror():
    """Ретривер спільний для сесій Streamlit: якщо хтось скинув кеш, поки ми
    чекали ембедер, оцінка йде з локальної мапи й не падає."""
    docs = [doc(1, "A", "x"), doc(2, "B", "y")]
    vectors = {"q1": QUERY_VEC, "q2": QUERY_VEC, "A. x": vec_at(0.1), "B. y": vec_at(0.2)}
    holder = {}

    class Racing(FakeEmbed):
        def __call__(self, texts):
            out = super().__call__(texts)
            holder["retriever"]._cache.clear()      # «інша сесія» скинула кеш
            return out

    embed = Racing(vectors)
    retriever = SupportRetriever(lambda _q, _k: docs, embed)
    holder["retriever"] = retriever
    retriever("q1")
    assert retriever("q2") == ["A. x", "B. y"]


def test_cache_cleared_between_check_and_read_never_raises_keyerror():
    """Гонка, яку попередній тест не ловить: інша сесія скидає кеш МІЖ перевіркою
    «ключ є» і читанням. Читати треба одним атомарним get, а не `in` + `[]`."""
    docs = [doc(1, "A", "x")]
    retriever = SupportRetriever(lambda _q, _k: docs,
                                 FakeEmbed({"q": QUERY_VEC, "A. x": vec_at(0.1)}))
    retriever("q")                                    # вектор «A. x» уже в кеші

    class ClearsAfterCheck(dict):
        def __contains__(self, key):
            found = super().__contains__(key)
            self.clear()                              # «інша сесія» встигла скинути
            return found

    retriever._cache = ClearsAfterCheck(retriever._cache)
    assert retriever("q") == ["A. x"]


def test_same_id_different_text_never_shares_a_vector():
    """У завантаженому в стенд JSON id буває однаковий у всіх документів. Ключ кешу
    за id змусив би різні документи оцінюватись одним чужим вектором — мовчки."""
    docs = [{"id": 0, "title": "Near", "summary": "x"},
            {"id": 0, "title": "Far", "summary": "y"}]
    retriever, _s, _e = retriever_for(docs, {"Near. x": 0.10, "Far. y": 0.60})
    assert retriever("ransomware") == ["Near. x"]


def test_changed_text_under_same_id_is_embedded_again():
    """Якщо summary документа під тим самим id оновили в індексі, старий вектор
    не повинен лишатись."""
    pool = [[doc(1, "Old", "x")], [doc(1, "New", "x")]]
    embed = FakeEmbed({"q": QUERY_VEC, "Old. x": vec_at(0.6), "New. x": vec_at(0.1)})
    retriever = SupportRetriever(lambda _q, _k: pool.pop(0), embed)
    assert retriever("q") == []
    assert retriever("q") == ["New. x"]


def test_embedder_returning_too_few_vectors_is_an_explicit_error():
    """zip мовчки обрізав би відповідь ембедера, і далі вилетів би безглуздий
    KeyError. Невідповідність кількості — окрема зрозуміла помилка."""
    retriever = SupportRetriever(lambda _q, _k: [doc(1, "A", "x"), doc(2, "B", "y")],
                                 lambda texts: [QUERY_VEC])
    try:
        retriever("q")
    except ValueError as exc:
        assert "ембедер" in str(exc) and "1" in str(exc) and "3" in str(exc), exc
    else:
        raise AssertionError("очікувалась ValueError про кількість векторів")


# --- корпус у пам'яті (стенд) -------------------------------------------------
def test_corpus_retriever_is_none_without_documents():
    assert corpus_retriever([], FakeEmbed({})) is None


def test_corpus_retriever_selects_like_onyx():
    """Стенд відбирає тим самим правилом і порогом, що ONYX: лише етапу BM25 немає,
    кандидати — увесь завантажений корпус."""
    docs = [doc(1, "Near", "x"), doc(2, "Far", "y")]
    embed = FakeEmbed({"q": QUERY_VEC, "Near. x": vec_at(0.1), "Far. y": vec_at(0.5)})
    retriever = corpus_retriever(docs, embed)
    assert isinstance(retriever, SupportRetriever)
    assert retriever.max_distance == SUPPORT_MAX_DISTANCE
    assert retriever("q") == ["Near. x"]


def test_corpus_retriever_cache_holds_a_corpus_larger_than_the_default_bound():
    """Інакше корпус більший за MAX_CACHED_VECTORS скидав би кеш на кожному понятті й
    переембеджувався цілком — старий стенд ембедив корпус один раз."""
    docs = [doc(i, f"T{i}", "x") for i in range(MAX_CACHED_VECTORS + 3)]
    embed = FakeEmbed({"q1": QUERY_VEC, "q2": QUERY_VEC})
    retriever = corpus_retriever(docs, embed)
    retriever("q1")
    retriever("q2")
    assert len(embed.embedded) == len(docs) + 2      # кожен документ — один раз


def test_zero_vector_is_far_not_a_crash():
    docs = [doc(1, "Z", "x")]
    embed = FakeEmbed({"q": QUERY_VEC, "Z. x": [0.0, 0.0]})
    assert SupportRetriever(lambda _q, _k: docs, embed)("q") == []


# --- support_message ----------------------------------------------------------
def test_message_none_when_support_was_off():
    assert support_message(None) is None
    assert support_message({}) is None
    assert support_message({"asked": 0, "supported": 0, "errors": 0}) is None


def test_message_error_when_every_request_failed():
    level, text = support_message({"asked": 8, "supported": 0, "errors": 8})
    assert level == "error"
    assert "8" in text and "недоступн" in text


def test_message_warning_when_corpus_gave_nothing():
    level, text = support_message({"asked": 13, "supported": 0, "errors": 0})
    assert level == "warning"
    assert "13" in text and "без" in text


def test_message_warning_even_with_some_errors_if_nothing_supported():
    level, _text = support_message({"asked": 10, "supported": 0, "errors": 3})
    assert level == "warning"


def test_message_info_with_counts():
    level, text = support_message({"asked": 13, "supported": 5, "errors": 0})
    assert level == "info"
    assert "Опора на корпус: 5 з 13 розкриттів отримали фрагменти джерел." in text
    assert "збо" not in text


def test_message_info_mentions_failed_requests():
    level, text = support_message({"asked": 13, "supported": 5, "errors": 2})
    assert level == "info"
    assert "Опора на корпус: 5 з 13 розкриттів отримали фрагменти джерел." in text
    assert "2" in text.split("джерел.", 1)[1]


# --- ручний запуск без pytest ---------------------------------------------
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
