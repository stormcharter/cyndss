"""
Маскування секретів перед логуванням.

НАВІЩО. Shipper на старті пише весь конфіг у лог. Доки Mongo працював без
автентифікації, це було нешкідливо. З `--auth` рядок підключення став
`mongodb://onyx_shipper:<пароль>@mongo:27017/...` — і пароль поїхав би в
`docker logs` відкритим текстом. Доступ до логів ширший за доступ до .env
(0600), тож лог — це окремий канал витоку, який треба закривати окремо.

Маскуємо, але не сліпо: схема, користувач і хост лишаються видимими — без них
з логу неможливо зрозуміти, КУДИ саме не вдалось підключитись.

Модуль навмисно не імпортує ні pydantic, ні config — тестується ізольовано
(інстанціювання Settings тягне за собою читання .env).
"""

import re

# Маркер у логах. Не "***", щоб у грепі було однозначно видно навмисне маскування.
REDACTED = "<redacted>"

# Поле вважаємо секретним за назвою — так нове поле (напр. mongo_app_password)
# маскується саме собою, без правок цього списку.
_SECRET_NAME_PARTS = ("password", "passwd", "pwd", "secret", "token",
                      "api_key", "apikey")

# scheme://user:password@... — пароль до першого '@' (у самому паролі '@' та '/'
# мають бути percent-encoded, тож межі однозначні).
_URI_CREDENTIALS_RE = re.compile(
    r"(?P<head>[a-zA-Z][a-zA-Z0-9+.\-]*://[^:@/\s]+:)(?P<pw>[^@/\s]*)(?P<tail>@)"
)

# Хмарні API часто передають секрети і в query-рядку URL.
_URI_QUERY_SECRET_RE = re.compile(
    r"(?P<head>[?&](?:api[_-]?key|access[_-]?token|token|password|secret)=)"
    r"(?P<value>[^&#\s]*)",
    re.IGNORECASE,
)


def redact_uri_credentials(value):
    """Замаскувати паролі з userinfo та секретоносні query-параметри URL."""
    if not isinstance(value, str):
        return value
    redacted = _URI_CREDENTIALS_RE.sub(
        lambda match: match.group("head") + REDACTED + match.group("tail"),
        value,
    )
    return _URI_QUERY_SECRET_RE.sub(
        lambda match: match.group("head") + REDACTED,
        redacted,
    )


def is_secret_field(name) -> bool:
    """Чи є поле секретним за іменем."""
    low = str(name).lower()
    return any(part in low for part in _SECRET_NAME_PARTS)


def safe_settings_dump(config: dict) -> dict:
    """Копія конфігу, придатна до логування.

    Секретні поля -> REDACTED (None лишається None: у логах має бути видно, що
    ключ НЕ заданий, це різні діагнози). У решті рядків гасяться креденшели URI.
    Вхідний словник не змінюється.
    """
    out = {}
    for key, value in config.items():
        if is_secret_field(key):
            out[key] = None if value is None else REDACTED
        else:
            out[key] = redact_uri_credentials(value)
    return out
