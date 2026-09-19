"""
Юніт-тести для shipper/schema.py — побудова DDL таблиці Manticore.

МОТИВАЦІЯ (пошук). Професор поскаржився: пошук «vulnerability» не знаходить
«vulnerabilities», а україномовні запити майже не працюють. Корінь — у DDL
таблиці не ввімкнено морфологію (стемінг) та інфіксне індексування, тому
токени матчаться ТОЧНО, слово-в-слово. Ці тести фіксують вимогу до схеми:

  1. morphology зі stem_en  -> англ. словоформи нормалізуються
     (vulnerability == vulnerabilities після стемінгу);
  2. morphology зі stem_ru  -> кир. флексії (ru та частково uk) стягуються
     до спільної основи (Manticore не має власного стемера для uk, stem_ru —
     найкращий доступний важіль);
  3. min_infix_len          -> підрядковий добір як мовно-незалежна страховка
     для решти україномовних форм.

DDL винесено в ОКРЕМИЙ модуль без залежності від shipper.config навмисно:
щоб цей тест не тягнув .env / pydantic-settings і гнався ізольовано.

Запуск:
    pytest tests/test_schema.py -v
або як звичайний скрипт (без pytest):
    python tests/test_schema.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shipper.schema import build_create_table_sql  # noqa: E402


def _ddl(table="events_enriched"):
    return build_create_table_sql(table).lower()


# --- морфологія: корінь бага «vulnerability != vulnerabilities» --------------
def test_ddl_declares_morphology():
    """DDL має вмикати morphology — без неї матчинг суто потокенний."""
    assert "morphology" in _ddl()


def test_ddl_has_english_stemming():
    """stem_en нормалізує англ. словоформи (vulnerability <-> vulnerabilities)."""
    assert "stem_en" in _ddl()


def test_ddl_has_russian_stemming():
    """stem_ru стягує РОСІЙСЬКІ флексії (корпус має ru-пости). Українську НЕ
    покриває (перевірено вживу: uk-форми лишаються різними токенами) — для uk
    працює інфікс (min_infix_len) та лематизація запиту окремим кроком."""
    assert "stem_ru" in _ddl()


# --- інфікс: мовно-незалежна страховка для решти форм ------------------------
def test_ddl_enables_infix_indexing():
    """min_infix_len вмикає підрядковий добір (важливо для україномовних форм,
    які стемер не покриває)."""
    assert "min_infix_len" in _ddl()


# --- рефактор нічого не загубив ---------------------------------------------
def test_ddl_keeps_content_field_indexed():
    """Поле content лишається повнотекстово індексованим (пошук по тілу)."""
    ddl = _ddl()
    assert "content text indexed" in ddl


def test_ddl_keeps_knn_embedding():
    """KNN-вектор 768 під nomic-embed-text не має зникнути після рефактора."""
    ddl = _ddl()
    assert "float_vector" in ddl
    assert "768" in ddl


def test_ddl_uses_given_table_name():
    """Ім'я таблиці параметризоване, а не захардкоджене."""
    assert "my_custom_table" in build_create_table_sql("my_custom_table").lower()


def test_ddl_is_create_table_if_not_exists():
    """Ідемпотентний CREATE, як і було."""
    assert "create table if not exists" in _ddl()


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
