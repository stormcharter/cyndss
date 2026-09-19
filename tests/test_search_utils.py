"""
Юніт-тести для search_utils.py — безпечна побудова MATCH()-виразу для Manticore.

Мотивація (security): у класичному повнотекстовому пошуку ui.py користувацький
ввід підставлявся в MATCH('@(title,summary) {q}') СИРИМ. Це дає дві діри одразу:
  1) вихід із SQL-літерала одинарною лапкою -> SQL-ін'єкція в Manticore;
  2) вихід із full-text виразу (символи @ ( ) | - ! " ~ / ^ $ =) -> зміна
     області пошуку/логіки запиту (FTS-ін'єкція), напр. @* або @(content).

Запуск:
    pytest tests/test_search_utils.py -v
або як звичайний скрипт (без pytest):
    python tests/test_search_utils.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_utils import (  # noqa: E402
    build_language_filter,
    build_match_clause,
    build_or_match_clause,
    sanitize_match_query,
)


# --- допоміжне -------------------------------------------------------------
def mysql_escape(s):
    """Імітація pymysql conn.escape_string() (без лапок навколо): подвоює
    зворотний слеш і екранує лапки — так само, як робить сервер."""
    BS = chr(92)
    out = []
    for ch in s:
        if ch == BS:
            out.append(BS + BS)
        elif ch == chr(39):
            out.append(BS + chr(39))
        elif ch == chr(34):
            out.append(BS + chr(34))
        elif ch == chr(10):
            out.append(BS + "n")
        elif ch == chr(13):
            out.append(BS + "r")
        elif ch == chr(0):
            out.append(BS + "0")
        else:
            out.append(ch)
    return "".join(out)


def sql_literal_body(clause):
    """Дістає тіло SQL-літерала з MATCH('...') — усе між першою та останньою
    одинарною лапкою."""
    first = clause.index("'")
    last = clause.rindex("'")
    assert last > first, clause
    return clause[first + 1:last]


def has_unescaped(body, ch):
    """Чи є в тілі літерала НЕекранований символ ch (з урахуванням зворотного
    слеша як екрануючого префікса)."""
    i = 0
    while i < len(body):
        if body[i] == chr(92):
            i += 2          # наступний символ екранований — пропускаємо
            continue
        if body[i] == ch:
            return True
        i += 1
    return False


# Довірений префікс полів. Оновлено разом із розширенням дефолту до content
# (див. test_default_fields_include_content) — сам по собі це не послаблює
# перевірки ін'єкцій нижче: вони й далі звіряють користувацьку частину.
PREFIX = "@(title,summary,content) "


def sql_unescape(body):
    """Знімає ОДИН рівень SQL-екранування — рівно те, що зробить сервер
    при розборі літерала. На виході — рядок, який РЕАЛЬНО побачить full-text
    парсер Manticore. Перевіряти FTS-екранування треба саме на ньому:
    послідовність (слеш)(слеш)| у літералі означає для FTS екранований |."""
    BS = chr(92)
    named = {"n": chr(10), "r": chr(13), "t": chr(9), "0": chr(0)}
    out, i = [], 0
    while i < len(body):
        if body[i] == BS and i + 1 < len(body):
            nxt = body[i + 1]
            out.append(named.get(nxt, nxt))
            i += 2
        else:
            out.append(body[i])
            i += 1
    return "".join(out)


def fts_user_part(clause):
    """Користувацька частина full-text виразу (без довіреного префікса полів)."""
    fts = sql_unescape(sql_literal_body(clause))
    assert fts.startswith(PREFIX), fts
    return fts[len(PREFIX):]


# --- порожній ввід ---------------------------------------------------------
def test_empty_query_yields_no_clause():
    """Порожній/пробільний запит не має додавати жодної умови."""
    assert build_match_clause("", mysql_escape) == ""
    assert build_match_clause("   ", mysql_escape) == ""
    assert build_match_clause(None, mysql_escape) == ""


def test_query_of_only_operators_yields_no_clause():
    """Ввід із самих операторів після санітизації порожній -> умови нема."""
    assert build_match_clause("   |||   ", mysql_escape) == ""


# --- нормальні запити працюють як раніше ------------------------------------
# ПРИМІТКА: дефолтний перелік полів свідомо розширено до (title,summary,content)
# — див. test_default_fields_include_content. Це зміна ВИМОГИ (uk-recall), тому
# очікувані рядки нижче оновлено з @(title,summary) на @(title,summary,content).
def test_plain_word_query_kept():
    clause = build_match_clause("Cisco", mysql_escape)
    assert clause == "MATCH('@(title,summary,content) Cisco')", clause


def test_multiword_query_kept():
    clause = build_match_clause("root access", mysql_escape)
    assert clause == "MATCH('@(title,summary,content) root access')", clause


def test_cyrillic_query_kept():
    clause = build_match_clause("кібератака ЗСУ", mysql_escape)
    assert clause == "MATCH('@(title,summary,content) кібератака ЗСУ')", clause


def test_custom_fields_are_respected():
    clause = build_match_clause("Cisco", mysql_escape, fields=("title",))
    assert clause == "MATCH('@(title) Cisco')", clause


def test_default_fields_include_content():
    """Дефолтний пошук має бити і по полю content (оригінальний текст поста).

    ВИМОГА (uk-recall). title/summary — це англомовний переклад, який робить
    LLM. Тіло content зберігає ОРИГІНАЛ (здебільшого українською). Поки пошук
    ішов лише по title,summary, україномовний запит шукав у чужомовних полях і
    не знаходив нічого. Додавання content — головний важіль для українського
    пошуку (content TEXT indexed — повнотекстово шукається, хоч і не stored)."""
    clause = build_match_clause("Cisco", mysql_escape)
    assert "content" in clause, clause
    assert clause == "MATCH('@(title,summary,content) Cisco')", clause


# --- SQL-ін'єкція ----------------------------------------------------------
def test_single_quote_cannot_break_sql_literal():
    """Одинарна лапка з вводу не має завершувати SQL-літерал."""
    clause = build_match_clause("O'Brien", mysql_escape)
    body = sql_literal_body(clause)
    assert not has_unescaped(body, "'"), clause
    assert clause.startswith("MATCH('") and clause.endswith("')"), clause


def test_classic_sqli_payload_is_neutralised():
    """Пейлоад "') OR 1=1 -- " не має ані закрити літерал, ані закрити MATCH()."""
    clause = build_match_clause("') OR 1=1 -- ", mysql_escape)
    # SQL-рівень: лапка не виривається з літерала
    body = sql_literal_body(clause)
    assert not has_unescaped(body, "'"), clause
    # FTS-рівень: дужка не закриває вираз @(...)
    assert not has_unescaped(fts_user_part(clause), ")"), clause
    assert clause.count("MATCH(") == 1
    assert clause.startswith("MATCH('") and clause.endswith("')"), clause


def test_backslash_cannot_escape_closing_quote():
    """Ввід, що закінчується зворотним слешем, не має зʼїсти закриваючу лапку
    (класичний обхід наївного екранування лапок)."""
    clause = build_match_clause("payload\\", mysql_escape)
    body = sql_literal_body(clause)
    assert not has_unescaped(body, "'"), clause
    trailing = len(body) - len(body.rstrip("\\"))
    assert trailing % 2 == 0, clause


def test_semicolon_and_comment_stay_inside_literal():
    """Крапка з комою та -- лишаються всередині літерала (не стають окремим
    стейтментом)."""
    clause = build_match_clause("test; DROP TABLE events_enriched; --", mysql_escape)
    body = sql_literal_body(clause)
    assert not has_unescaped(body, "'"), clause
    assert clause.endswith("')"), clause

FTS_OPS_QUERY = 'a | b -c !d ~2 "e"/1 ^f$ =g << h >'
FTS_OPS_CHARS = '|!-~/^$="<>'


# --- FTS-ін'єкція (вихід за межі повнотекстового виразу) ---------------------
def test_field_operator_from_user_is_escaped():
    """Символ @ з вводу не має перевизначати область пошуку (напр. на @content)."""
    clause = build_match_clause("@(content) secret", mysql_escape)
    user_part = fts_user_part(clause)
    assert not has_unescaped(user_part, "@"), user_part
    assert not has_unescaped(user_part, "("), user_part
    assert not has_unescaped(user_part, ")"), user_part


def test_star_field_operator_is_escaped():
    """@* (усі поля) з вводу теж має бути літералом."""
    clause = build_match_clause("@* password", mysql_escape)
    user_part = fts_user_part(clause)
    assert not has_unescaped(user_part, "@"), user_part


def test_fts_boolean_operators_are_escaped():
    """Оператори extended-синтаксису з користувацького вводу мають лишатись
    літералами, а не керувати запитом."""
    clause = build_match_clause(FTS_OPS_QUERY, mysql_escape)
    # SQL-рівень цілий...
    assert not has_unescaped(sql_literal_body(clause), "'"), clause
    # ...і на FTS-рівні жоден оператор не лишився керуючим
    user_part = fts_user_part(clause)
    for ch in FTS_OPS_CHARS:
        assert not has_unescaped(user_part, ch), (ch, user_part)


# --- фолбек-екранувальник (коли escape_fn не передали) ---------------------
def test_default_escape_fn_escapes_quote():
    """_default_sql_escape має ЕКРАНУВАТИ лапку (слеш + лапка), а не повертати
    її сирою. Раніше out.append("\\'") був еквівалентний просто "'" — бо в
    Python-літералі "\\'" == "'" (сам пітонівський escape, не наш)."""
    from search_utils import _default_sql_escape

    out = _default_sql_escape("'")
    assert out == chr(92) + chr(39), repr(out)


def test_build_match_clause_without_escape_fn_is_safe():
    """build_match_clause БЕЗ escape_fn (фолбек-шлях) не має лишати
    неекранованих лапок у SQL-літералі — інакше "') OR 1=1 -- " виривається
    з MATCH('...') так само, як і без будь-якого екранування."""
    clause = build_match_clause("') OR 1=1 -- ")
    body = sql_literal_body(clause)
    assert not has_unescaped(body, "'"), clause
    assert clause.startswith("MATCH('") and clause.endswith("')"), clause


# --- OR-матчинг для чату (build_or_match_clause) ----------------------------
# МОТИВАЦІЯ. Агентський шлях будував запит як AND усіх слів довші 3 символів:
# «say about vulnerabilities» -> `about vulnerabilities`, тобто документ мусив
# містити ОБИДВА слова. Наслідок — з корпусу в контекст LLM потрапляло 1-2
# випадкові документи, і агент відповідав «джерела не містять релевантної
# інформації». Правильний IR-підхід: OR по термах + ранжування WEIGHT()
# (BM25) на стороні Manticore. Службові слова запиту («розкажи», «про»,
# «say», «about») відсіюємо — інакше вони тягнуть у топ шум.
def test_or_clause_joins_terms_with_or():
    clause = build_or_match_clause("root access vulnerability", mysql_escape)
    assert clause == "MATCH('@(title,summary,content) root | access | vulnerability')", clause


def test_or_clause_drops_query_stopwords():
    """«say about vulnerabilities» — реальний запит, що ламав агента."""
    clause = build_or_match_clause("say about vulnerabilities", mysql_escape)
    assert fts_user_part(clause) == "vulnerabilities", clause


def test_or_clause_drops_ukrainian_query_stopwords():
    """Ввід приходить УЖЕ лематизованим (ui лематизує до виклику), тому в
    списку службових слів мають бути саме леми: розкажи->розказати,
    останні->останній. Інакше вони проскакували б у пошук."""
    clause = build_or_match_clause("розказати про останній вразливість", mysql_escape)
    assert fts_user_part(clause) == "вразливість", clause


def test_or_clause_returns_empty_when_only_stopwords():
    """Запит із самих службових слів не має ставати умовою — хай спрацює
    семантичний KNN-фолбек, а не OR по сміттю."""
    assert build_or_match_clause("розказати про", mysql_escape) == ""
    assert build_or_match_clause("tell me about", mysql_escape) == ""


def test_or_clause_drops_too_short_tokens():
    clause = build_or_match_clause("a я ransomware", mysql_escape)
    assert fts_user_part(clause) == "ransomware", clause


def test_or_clause_empty_input():
    assert build_or_match_clause("", mysql_escape) == ""
    assert build_or_match_clause(None, mysql_escape) == ""


def test_or_clause_keeps_cve_identifier_intact():
    """CVE-2024-1234 містить дефіс — оператор FTS. Він має бути ЕКРАНОВАНИЙ,
    але сам терм не має розсипатись і не має зникнути."""
    clause = build_or_match_clause("CVE-2024-1234", mysql_escape)
    user_part = fts_user_part(clause)
    assert not has_unescaped(user_part, "-"), user_part
    assert "CVE" in user_part and "2024" in user_part and "1234" in user_part


# --- OR-матчинг: ті самі ін'єкції, що й для AND -----------------------------
def test_or_clause_neutralises_sqli_payload():
    clause = build_or_match_clause("') OR 1=1 -- ", mysql_escape)
    assert not has_unescaped(sql_literal_body(clause), "'"), clause
    assert clause.count("MATCH(") <= 1
    if clause:
        assert clause.startswith("MATCH('") and clause.endswith("')"), clause


def test_or_clause_escapes_field_operator():
    """@ і дужки з вводу не мають перевизначати область пошуку."""
    clause = build_or_match_clause("@(content) secret", mysql_escape)
    user_part = fts_user_part(clause)
    assert not has_unescaped(user_part, "@"), user_part
    assert not has_unescaped(user_part, "("), user_part
    assert not has_unescaped(user_part, ")"), user_part


def test_or_clause_escapes_user_supplied_or_operator():
    """Роздільник ' | ' між термами ставимо МИ (довірений), а той самий символ
    із вводу лишається екранованим літералом."""
    clause = build_or_match_clause("alpha|beta", mysql_escape)
    user_part = fts_user_part(clause)
    # один терм -> жодного нашого роздільника, отже неекранованого | бути не може
    assert not has_unescaped(user_part, "|"), user_part


def test_or_clause_escapes_fts_operators_in_terms():
    clause = build_or_match_clause(FTS_OPS_QUERY, mysql_escape)
    assert not has_unescaped(sql_literal_body(clause), "'"), clause
    user_part = fts_user_part(clause)
    for ch in FTS_OPS_CHARS.replace("|", ""):
        assert not has_unescaped(user_part, ch), (ch, user_part)


def test_or_clause_without_escape_fn_is_safe():
    clause = build_or_match_clause("') OR 1=1 -- ")
    if clause:
        assert not has_unescaped(sql_literal_body(clause), "'"), clause


def test_or_clause_respects_custom_fields():
    clause = build_or_match_clause("ransomware", mysql_escape, fields=("title",))
    assert clause == "MATCH('@(title) ransomware')", clause


def test_or_clause_limits_number_of_terms():
    """Довгий запит не має перетворюватись на OR із сотень термів — це важкий
    запит до Manticore з боку користувача UI."""
    clause = build_or_match_clause(" ".join(f"term{i}" for i in range(200)), mysql_escape)
    assert fts_user_part(clause).count("|") <= 24, clause


# --- фільтр мов --------------------------------------------------------------
# МОТИВАЦІЯ (реальний баг). Фільтр був жорстко зашитий як
# `language IN ('uk','ru','en')` з фолбеком на той самий перелік, коли нічого
# не вибрано. А в індексі 952 з 1495 документів (64%!) мають language='unknown'
# — детектор мови не впорався. Ці документи були НЕВИДИМІ і в стрічці, і в
# контексті агентів: пошук по всьому корпусу бачив лише третину корпусу.
def test_language_filter_builds_in_clause():
    assert build_language_filter(["uk", "en"]) == "language IN ('uk','en')"


def test_empty_selection_means_no_filter():
    """Порожній вибір = «усі мови», а не «повернутись до трьох зашитих».
    Інакше зняти фільтр було просто неможливо."""
    assert build_language_filter([]) == ""
    assert build_language_filter(None) == ""


def test_unknown_language_is_selectable():
    """Головне: 'unknown' має бути повноцінним значенням фільтра."""
    clause = build_language_filter(["uk", "unknown"])
    assert "unknown" in clause, clause


def test_language_codes_are_validated():
    """Значення приходять із мультиселекта (керується кодом), але підстановка
    йде в SQL — тож формат перевіряємо, а не довіряємо."""
    try:
        build_language_filter(["uk", "en'); DROP TABLE events_enriched; --"])
    except ValueError:
        return
    raise AssertionError("очікували ValueError на некоректному коді мови")


def test_language_filter_ignores_blank_entries():
    assert build_language_filter(["uk", "", "  "]) == "language IN ('uk')"


def test_language_filter_deduplicates():
    assert build_language_filter(["uk", "uk", "en"]) == "language IN ('uk','en')"


# --- санітизатор -----------------------------------------------------------
def test_sanitizer_is_idempotent_on_safe_input():
    """Безпечний ввід не мутує (жодних зайвих слешів)."""
    assert sanitize_match_query("Cisco IOS XE") == "Cisco IOS XE"


def test_whitespace_is_normalised():
    """Переводи рядка/таби згортаються в пробіли — запит лишається однорядковим."""
    out = sanitize_match_query("Cisco" + chr(10) + chr(9) + "IOS" + chr(13) + chr(10) + "  XE")
    assert out == "Cisco IOS XE", repr(out)
    assert "\n" not in out and "\r" not in out and "\t" not in out


def test_control_chars_are_stripped():
    """NUL та інші керуючі символи не потрапляють у запит."""
    out = sanitize_match_query("Cisco\x00\x07IOS")
    assert "\x00" not in out and "\x07" not in out, repr(out)


def test_overlong_query_is_truncated():
    """Надто довгий ввід обрізається (захист від важких запитів у Manticore)."""
    out = sanitize_match_query("A" * 5000)
    assert len(out) <= 512, len(out)


# --- ручний запуск без pytest ---------------------------------------------
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
