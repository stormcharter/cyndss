"""
Юніт-тести чистого ядра мігратора tools/reindex_lemmatize.py.

Мігратор перебудовує індекс Manticore під нову схему (морфологія) і
лематизований content БЕЗ повторного ембеддингу/екстракції: зчитує вже
пораховані stored-поля зі старого індексу (title/summary/embedding/verified_*),
а content відновлює з оригіналу в Mongo, лематизуючи його.

Тут тестуємо ЛИШЕ чисту трансформацію рядка (build_migrated_doc / parse_embedding)
— без Manticore/Mongo. I/O-обвязка (main) перевіряється dry-run'ом на сервері.

Потребує pymorphy3 (як і lemmatizer). Запуск:
    python tests/test_reindex_migrator.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.reindex_lemmatize import build_migrated_doc, parse_embedding  # noqa: E402


# --- головне: content лематизується з оригіналу, а не береться зі старого рядка -
def test_content_is_lemmatized_from_original_text():
    row = {"mongo_id": "m1", "title": "T", "summary": "S", "embedding": "0.1,0.2"}
    doc = build_migrated_doc(row, "виявлено вразливості")
    assert "вразливість" in doc["content"]
    assert "вразливості" not in doc["content"]


def test_old_row_content_is_ignored():
    """Навіть якщо у старому рядку якось опинилось content — воно НЕ переноситься;
    content завжди відновлюється з оригіналу (Mongo) і лематизується."""
    doc = build_migrated_doc({"content": "сміття", "embedding": "0.1"}, "атаки")
    assert "сміття" not in doc["content"]
    assert "атака" in doc["content"]


def test_empty_original_text_gives_empty_content():
    doc = build_migrated_doc({"embedding": "0.1"}, "")
    assert doc["content"] == ""


# --- збережені поля переносяться як є (без повторного рахування) --------------
def test_preserved_fields_pass_through():
    row = {
        "mongo_id": "m1", "title": "Cisco", "summary": "S",
        "verified_cve_ids": "CVE-2024-1", "cve_count_verified": 3,
        "source_type": "telegram", "language": "uk", "embedding": "0.1,0.2",
    }
    doc = build_migrated_doc(row, "текст")
    assert doc["mongo_id"] == "m1"
    assert doc["title"] == "Cisco"
    assert doc["verified_cve_ids"] == "CVE-2024-1"
    assert doc["cve_count_verified"] == 3
    assert doc["source_type"] == "telegram"


def test_absent_fields_are_not_invented():
    """Якщо поля не було у старому рядку — не додаємо його з None."""
    doc = build_migrated_doc({"embedding": "0.1"}, "x")
    assert "verified_cve_ids" not in doc


# --- embedding: рядок «f1,f2,...» -> список float (для /bulk JSON) -------------
def test_embedding_string_parsed_to_floats():
    doc = build_migrated_doc({"embedding": "0.1,0.2,0.3"}, "x")
    assert doc["embedding"] == [0.1, 0.2, 0.3]


def test_embedding_list_passes_through():
    doc = build_migrated_doc({"embedding": [0.1, 0.2]}, "x")
    assert doc["embedding"] == [0.1, 0.2]


def test_parse_embedding_edge_cases():
    assert parse_embedding("0.1,0.2") == [0.1, 0.2]
    assert parse_embedding([0.1]) == [0.1]
    assert parse_embedding("") == []
    assert parse_embedding(None) == []
    assert parse_embedding("0.1, 0.2 , 0.3") == [0.1, 0.2, 0.3]


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
