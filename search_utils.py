"""
Безпечна побудова повнотекстової умови MATCH() для Manticore.

НАВІЩО (security). Користувацький рядок пошуку потрапляє у SQL-запит двічі
вкладеним: спершу як full-text вираз, потім — усередині SQL-літерала:

    SELECT ... WHERE MATCH('@(title,summary) <ЗАПИТ>')

Тому екранувати треба ДВА рівні, і саме в такому порядку:

  1. Рівень full-text (extended syntax). Символи-оператори Manticore
     ( @ ( ) | - ! ~ " / ^ $ = ... ) дозволяють вийти за межі задуманого
     виразу: напр. @(content) або @* перевизначає область пошуку на поля,
     які UI навмисне не показує. Нейтралізуємо зворотним слешем.
  2. Рівень SQL-літерала. Одинарна лапка / зворотний слеш дозволяють вийти
     з літерала і дописати довільний SQL. Це робить escape_fn (у проді —
     pymysql `conn.escape_string`, який ще й подвоює слеші з кроку 1, як і
     треба: сервер зніме один рівень, FTS-парсер побачить екранований @).

Модуль НЕ імпортує streamlit / pymysql — тестується ізольовано.
"""

import re
import unicodedata

# Максимальна довжина повнотекстового запиту (після екранування). Захист від
# надважких запитів у Manticore з боку анонімного користувача UI.
MAX_QUERY_LEN = 512

# Символи-оператори full-text синтаксису Manticore + зворотний слеш і лапки.
# Перелік свідомо ширший за документований мінімум: зайве екранування
# нешкідливе (Manticore трактує \X як літерал X), недостатнє — це діра.
FTS_SPECIAL_CHARS = set(chr(92) + chr(39) + chr(34) + "!$()-/<>@^|~&=*?")

_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Максимум термів в OR-виразі. Без ліміту довгий запит перетворюється на OR із
# сотень слів — для Manticore це важкий запит, ініційований користувачем UI.
MAX_OR_TERMS = 25

# Мінімальна довжина терма (без урахування екрануючих слешів). Однолітерні
# токени в OR дають шум і майже не несуть сигналу.
MIN_TERM_LEN = 2

# Службові слова ЗАПИТУ (не тексту!) — те, чим людина обрамляє питання до
# агента: «розкажи про…», «say about…». У OR-виразі вони тягнуть у топ
# випадкові документи, тому відсіюються.
#
# ВАЖЛИВО: ввід сюди приходить УЖЕ лематизованим (ui.lemmatize_query), тож
# українські/російські форми перелічені саме як ЛЕМИ (розкажи -> розказати,
# останні -> останній). Поверхневі форми теж лишені — на випадок виклику без
# лематизації.
QUERY_STOPWORDS = {
    # EN
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "is",
    "are", "was", "were", "be", "been", "do", "does", "did", "me", "my", "we",
    "us", "you", "your", "it", "its", "this", "that", "these", "those",
    "say", "says", "said", "tell", "tells", "told", "show", "shows", "give",
    "gives", "please", "about", "what", "which", "who", "how", "why", "when",
    "where", "some", "any", "all", "more", "most", "latest", "recent", "new",
    "make", "made", "build", "generate", "create", "draw", "plot",
    # UK/RU — леми
    "розказати", "сказати", "показати", "дати", "написати", "зробити",
    "збудувати", "згенерувати", "побудувати", "намалювати", "вивести",
    "про", "мені", "нам", "ми", "ти", "ви", "він", "вона", "воно", "вони",
    "що", "який", "як", "де", "коли", "чому", "чи", "той", "цей", "це",
    "бути", "ласка", "останній", "новий", "свіжий", "весь", "усе", "все",
    "також", "дуже", "більш", "менш", "будь",
    # RU — поверхневі/леми
    "рассказать", "расскажи", "скажи", "покажи", "дай", "напиши", "сделай",
    "построй", "сгенерируй", "нарисуй", "мне", "нам", "про", "что", "какой",
    "какие", "как", "где", "когда", "почему", "это", "этот", "последний",
    "последние", "новый", "новые", "самый", "пожалуйста",
    # поверхневі UK-форми (якщо лематизацію не застосували)
    "розкажи", "розкажіть", "покажи", "дай", "напиши", "зроби", "збудуй",
    "згенеруй", "останні", "нові", "які", "яка", "яке", "мені",
}


def _has_alnum(token):
    """Чи є в токені бодай один буквено-цифровий символ. Токен на кшталт «\\|»
    (екранований оператор із вводу) сенсу як пошуковий терм не має."""
    return any(ch.isalnum() for ch in token)


def _term_len(token):
    """Довжина терма без екрануючих слешів — «\\-» це один символ, не два."""
    return sum(1 for ch in token if ch != "\\")


def _default_sql_escape(value):
    """Запасний SQL-екранувальник (коли escape_fn не передали). Поводиться як
    pymysql `conn.escape_string`: подвоює зворотний слеш, екранує лапки."""
    out = []
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "'":
            out.append("\\'")
        elif ch == '"':
            out.append('\\"')
        else:
            out.append(ch)
    return "".join(out)


def _strip_dangling_escape(value):
    """Прибирає непарний хвостовий зворотний слеш, щоб обрізаний запит не
    «з'їв» наступний символ (напр. закриваючу лапку літерала)."""
    trailing = len(value) - len(value.rstrip("\\"))
    if trailing % 2:
        return value[:-1]
    return value


def sanitize_match_query(raw):
    """Готує користувацький рядок до вставки у full-text вираз.

    Кроки: керуючі символи -> пробіл, згортання пробілів, відсів запитів без
    жодного словесного символу, екранування FTS-операторів, обрізання до
    MAX_QUERY_LEN. Повертає "" якщо шукати нема чого."""
    if not raw:
        return ""
    text = str(raw)

    # Керуючі символи (NUL, BEL, переводи рядка, таби) -> пробіл.
    text = "".join(" " if unicodedata.category(ch) == "Cc" else ch for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    # Запит із самих операторів (напр. "|||") сенсу не має — і після
    # екранування став би просто сміттям у виразі.
    if not any(ch.isalnum() for ch in text):
        return ""

    escaped = "".join(
        ("\\" + ch) if ch in FTS_SPECIAL_CHARS else ch
        for ch in text
    )

    if len(escaped) > MAX_QUERY_LEN:
        escaped = _strip_dangling_escape(escaped[:MAX_QUERY_LEN]).rstrip()
    return escaped


def build_match_clause(raw_query, escape_fn=None, fields=("title", "summary", "content")):
    """Повертає готову умову `MATCH('@(поля) запит')` або "" (нічого шукати).

    raw_query — сирий ввід користувача; escape_fn — SQL-екранувальник
    (`conn.escape_string`); fields — перелік полів індексу (керується кодом,
    не користувачем, але перевіряється на ідентифікатор про всяк випадок)."""
    if isinstance(fields, str):
        fields = [f.strip() for f in fields.split(",") if f.strip()]
    fields = list(fields)
    if not fields:
        raise ValueError("build_match_clause: порожній перелік полів")
    for f in fields:
        if not _FIELD_RE.match(f):
            raise ValueError("build_match_clause: некоректне ім'я поля: %r" % (f,))

    query = sanitize_match_query(raw_query)
    if not query:
        return ""

    escaped = (escape_fn or _default_sql_escape)(query)
    return "MATCH('@(%s) %s')" % (",".join(fields), escaped)


_LANG_RE = re.compile(r"^[a-z]{2,12}$")


def build_language_filter(languages):
    """Повертає `language IN ('uk','en')` або "" (фільтрувати не треба).

    ЧОМУ ПОРОЖНІЙ = «усі мови». Раніше фільтр був зашитий як
    IN ('uk','ru','en') з фолбеком на той самий перелік, коли нічого не
    вибрано. А детектор мови проставив 'unknown' 952 документам із 1495 (64%
    корпусу) — і ці документи були невидимі і в стрічці, і в контексті
    агентів. Зняти фільтр було неможливо в принципі. Тепер порожній вибір
    означає «без обмежень».

    Коди мов приходять із мультиселекта (керується кодом, не користувачем),
    але підставляються в SQL — тож формат перевіряємо явно."""
    if not languages:
        return ""
    codes = []
    for lang in languages:
        code = str(lang or "").strip().lower()
        if not code:
            continue
        if not _LANG_RE.match(code):
            raise ValueError("build_language_filter: некоректний код мови: %r" % (lang,))
        if code not in codes:
            codes.append(code)
    if not codes:
        return ""
    return "language IN (%s)" % ",".join("'%s'" % c for c in codes)


def build_or_match_clause(raw_query, escape_fn=None,
                          fields=("title", "summary", "content"),
                          stopwords=QUERY_STOPWORDS, max_terms=MAX_OR_TERMS):
    """Повертає `MATCH('@(поля) t1 | t2 | …')` — OR по значущих термах запиту.

    НАВІЩО окремо від build_match_clause. Той будує AND-вираз (усі слова
    поспіль) — це правильно для точкового пошуку в стрічці, але вбиває
    відкликання (recall) у чаті: питання «say about vulnerabilities» вимагало
    від документа ОБИДВА слова й давало 1-2 випадкові збіги. Тут навпаки:
    документ підходить, якщо містить БУДЬ-ЯКИЙ значущий терм, а порядок
    видачі визначає WEIGHT() (BM25) на боці Manticore — класична IR-схема
    «OR + ранжування».

    БЕЗПЕКА. Роздільник « | » вставляємо МИ і лише між уже санітизованими
    термами. Ввід користувача проходить той самий sanitize_match_query, що й
    в AND-гілці, тож його власні `|`, `@`, дужки лишаються екранованими
    літералами й не можуть перевизначити вираз."""
    if isinstance(fields, str):
        fields = [f.strip() for f in fields.split(",") if f.strip()]
    fields = list(fields)
    if not fields:
        raise ValueError("build_or_match_clause: порожній перелік полів")
    for f in fields:
        if not _FIELD_RE.match(f):
            raise ValueError("build_or_match_clause: некоректне ім'я поля: %r" % (f,))

    query = sanitize_match_query(raw_query)
    if not query:
        return ""

    # sanitize_match_query уже згорнув пробіли, тож split() безпечний.
    terms = []
    for token in query.split(" "):
        if not _has_alnum(token) or _term_len(token) < MIN_TERM_LEN:
            continue
        if token.lower().lstrip("\\") in stopwords:
            continue
        if token not in terms:            # дублі в OR нічого не додають
            terms.append(token)
        if len(terms) >= max_terms:
            break

    if not terms:
        return ""

    escaped = (escape_fn or _default_sql_escape)(" | ".join(terms))
    return "MATCH('@(%s) %s')" % (",".join(fields), escaped)
