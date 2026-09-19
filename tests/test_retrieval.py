"""
Юніт-тести для retrieval.py — злиття результатів гібридного пошуку.

МОТИВАЦІЯ. Агентський шлях шукав ОДНИМ способом: AND-матчинг усіх слів, а якщо
нічого — KNN. Через це «say about vulnerabilities» давало 1-2 документи, і
агент чесно відповідав «джерела не містять релевантної інформації». Тепер
запускаємо кілька стратегій паралельно (лематизований OR по укр., OR по
англійському перекладі, векторний KNN) і зливаємо їхні видачі.

Злиття круговим обходом (round-robin), а не «спершу вся перша стратегія»:
інакше перша ж стратегія з великою видачею з'їдає ліміт і гібрид вироджується
в неї саму.

Запуск:
    pytest tests/test_retrieval.py -v
або як звичайний скрипт (без pytest):
    python tests/test_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import merge_hits  # noqa: E402


def _h(doc_id, **kw):
    d = {"id": doc_id, "title": f"doc-{doc_id}"}
    d.update(kw)
    return d


# --- базове ------------------------------------------------------------------
def test_empty_input():
    assert merge_hits([]) == []
    assert merge_hits(None) == []
    assert merge_hits([[], []]) == []


def test_single_group_passthrough():
    hits = merge_hits([[_h(1), _h(2)]], limit=10)
    assert [h["id"] for h in hits] == [1, 2]


def test_duplicates_removed_across_groups():
    """Той самий документ, знайдений двома стратегіями, має бути один раз."""
    hits = merge_hits([[_h(1), _h(2)], [_h(2), _h(3)]], limit=10)
    assert [h["id"] for h in hits] == [1, 2, 3]


def test_duplicates_removed_within_group():
    hits = merge_hits([[_h(1), _h(1), _h(2)]], limit=10)
    assert [h["id"] for h in hits] == [1, 2]


# --- круговий обхід ----------------------------------------------------------
def test_round_robin_between_groups():
    """Кожна стратегія отримує свою частку, а не «хто перший, той і забрав»."""
    fts = [_h(1), _h(2), _h(3), _h(4)]
    knn = [_h(10), _h(11)]
    hits = merge_hits([fts, knn], limit=10)
    assert [h["id"] for h in hits] == [1, 10, 2, 11, 3, 4]


def test_large_first_group_does_not_starve_second():
    """Головна властивість: KNN потрапляє у вибірку навіть коли FTS дав багато."""
    fts = [_h(i) for i in range(50)]
    knn = [_h(900), _h(901)]
    hits = merge_hits([fts, knn], limit=6)
    ids = [h["id"] for h in hits]
    assert 900 in ids and 901 in ids, ids


# --- ліміт -------------------------------------------------------------------
def test_limit_respected():
    hits = merge_hits([[_h(i) for i in range(20)]], limit=5)
    assert len(hits) == 5


def test_limit_zero_returns_empty():
    assert merge_hits([[_h(1)]], limit=0) == []


# --- ідентифікація документів ------------------------------------------------
def test_falls_back_to_mongo_id_when_no_id():
    """Документи з різних гілок можуть не мати числового id індексу."""
    a = {"mongo_id": "abc", "title": "x"}
    b = {"mongo_id": "abc", "title": "x (дубль)"}
    hits = merge_hits([[a], [b]], limit=10)
    assert len(hits) == 1


def test_hit_without_any_identifier_is_skipped():
    """Документ без id І без mongo_id ідентифікувати нічим — його не можна ані
    дедуплікувати, ані потім дотягнути з Mongo. Такий запис пропускаємо, а не
    тягнемо в контекст LLM: краще на один документ менше, ніж дублі й
    «привиди» без джерела. Явний контракт, а не побічний ефект реалізації."""
    hits = merge_hits([[{"title": "без ідентифікатора"}, _h(1)]], limit=10)
    assert [h["id"] for h in hits] == [1], hits


def test_hit_objects_are_not_mutated():
    doc = _h(1)
    merge_hits([[doc]], limit=10)
    assert doc == {"id": 1, "title": "doc-1"}


def test_none_groups_are_tolerated():
    """Стратегія могла впасти й повернути None — це не має ламати злиття."""
    hits = merge_hits([None, [_h(1)], []], limit=10)
    assert [h["id"] for h in hits] == [1]


def test_non_dict_entries_are_skipped():
    hits = merge_hits([["сміття", _h(1), None]], limit=10)
    assert [h["id"] for h in hits] == [1]


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
