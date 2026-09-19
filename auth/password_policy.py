"""
Вимоги до пароля при СТВОРЕННІ облікового запису.

Розвиток власного password_strength_checker з трьома змінами (кожна закріплена
тестом у tests/test_password_policy.py):

  1. Прибрано заборону двох однакових сусідніх символів. Композиційні правила
     такого штибу лише звужують простір паролів (а отже — допомагають перебору)
     і дратують користувача; NIST SP 800-63B радить їх не вводити. Ловимо те, що
     справді слабке: 3+ однакових поспіль і послідовності 1234 / abcd / qwerty.
  2. Порівняння з логіном — без урахування регістру (раніше `admin` не ловився
     в паролі `Admin12345!`).
  3. Доданий блоклист очевидних паролів і токенів проєкту.

Політика працює ЛИШЕ при заведенні/зміні пароля — у ONYX немає самореєстрації,
облікові записи створює адміністратор (tools/create_user.py). Тому суворість тут
дешева: незручність одноразова, а пароль живе довго.

Модуль чистий: без streamlit, без Mongo, без argon2.
"""

import re

MIN_LENGTH = 12
# Верхня межа не про «стійкість», а про доступність: argon2 рахує хеш від рядка
# будь-якої довжини, і мегабайтний «пароль» у формі логіну — це безкоштовний DoS.
MAX_LENGTH = 128

# Скільки однакових символів поспіль ще припустимо ("aa" — так, "aaa" — ні).
MAX_RUN = 2
# Довжина монотонної послідовності, з якої вона вважається слабкою.
MAX_SEQUENCE = 4

_SPECIAL_RE = re.compile(r"[^a-zA-Z0-9]")

# Ряди клавіатури й алфавіт — джерела «послідовностей», які перебирають першими.
_SEQUENCE_ALPHABETS = (
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789",
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
    "йцукенгшщзхї",
    "фівапролджє",
    "ячсмитьбю",
)

# Найпоширеніші паролі + токени, повʼязані з проєктом (їх атакер перебирає
# одразу після rockyou-топу). Перевірка — на входження підрядком.
_BLOCKLIST_TOKENS = (
    "password", "passwort", "qwerty", "123456", "letmein", "welcome",
    "admin", "iloveyou", "monkey", "dragon", "sunshine", "princess",
    "correcthorsebatterystaple", "changeme", "secret", "master",
    "onyx", "mongo", "manticore", "threatintel", "diplom", "kpi",
)


def check_length(password) -> bool:
    """Довжина в межах [MIN_LENGTH, MAX_LENGTH]."""
    if not isinstance(password, str):
        return False
    return MIN_LENGTH <= len(password) <= MAX_LENGTH


def check_specials(password) -> bool:
    """Є хоча б один символ поза [a-zA-Z0-9]. Повертає саме bool."""
    if not isinstance(password, str):
        return False
    return _SPECIAL_RE.search(password) is not None


def check_no_long_runs(password, max_run: int = MAX_RUN) -> bool:
    """Немає (max_run + 1) однакових символів поспіль."""
    if not isinstance(password, str):
        return False
    run = 1
    for prev, cur in zip(password, password[1:]):
        run = run + 1 if cur == prev else 1
        if run > max_run:
            return False
    return True


def check_no_sequences(password, max_len: int = MAX_SEQUENCE) -> bool:
    """Немає монотонного фрагмента довжиною max_len (в обидва боки)."""
    if not isinstance(password, str):
        return False
    low = password.lower()
    for alphabet in _SEQUENCE_ALPHABETS:
        reverse = alphabet[::-1]
        for start in range(len(alphabet) - max_len + 1):
            if alphabet[start:start + max_len] in low:
                return False
            if reverse[start:start + max_len] in low:
                return False
    return True


def check_login_mismatch(login, password) -> bool:
    """Пароль не містить логін (без урахування регістру). Порожній логін — ок."""
    if not isinstance(password, str):
        return False
    if not login:
        return True
    return str(login).lower() not in password.lower()


def check_not_common(password) -> bool:
    """Пароль не містить очевидних токенів із блоклиста."""
    if not isinstance(password, str):
        return False
    low = password.lower()
    return not any(token in low for token in _BLOCKLIST_TOKENS)


def validate_password(login, password) -> list:
    """Повертає СПИСОК усіх порушень (порожній = пароль приймається).

    Саме список, а не перша помилка: інакше користувач виправляє претензії по
    одній і кожного разу отримує нову.
    """
    errors = []
    if not check_length(password):
        errors.append(
            "Довжина має бути від %d до %d символів" % (MIN_LENGTH, MAX_LENGTH)
        )
    if not check_specials(password):
        errors.append("Потрібен хоча б один спецсимвол (не літера й не цифра)")
    if not check_no_long_runs(password):
        errors.append("Не можна %d+ однакових символів поспіль" % (MAX_RUN + 1))
    if not check_no_sequences(password):
        errors.append("Прибери послідовності на кшталт 1234, abcd, qwerty")
    if not check_login_mismatch(login, password):
        errors.append("Пароль не має містити логін")
    if not check_not_common(password):
        errors.append("Пароль містить надто очевидне слово (перебирається першим)")
    return errors
