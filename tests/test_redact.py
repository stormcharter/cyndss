"""
Юніт-тести для shipper/redact.py — маскування секретів у логах.

Мотивація (security). Shipper на старті логує весь конфіг:

    logger.info("starting", config=settings.model_dump(...))

Доки Mongo працював без автентифікації, у `mongo_uri` не було чого ховати.
З увімкненням `--auth` рядок підключення став
`mongodb://onyx_shipper:<пароль>@mongo:27017/...`, і пароль поїхав у `docker logs`
відкритим текстом — його бачить будь-хто, хто має доступ до логів (а на
спільному боксі це ширше коло, ніж доступ до .env з правами 0600).

Тому конфіг перед логуванням проходить через `safe_settings_dump()`:
секретні поля замінюються на маркер, у будь-якому URI гаситься пароль.
Хост, користувач і схема лишаються видимими — саме вони потрібні для
діагностики «куди саме ми не змогли підключитись».

Запуск:
    pytest tests/test_redact.py -v
або:
    python tests/test_redact.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shipper.redact import (  # noqa: E402
    REDACTED,
    redact_uri_credentials,
    safe_settings_dump,
)

SECRET = "c58371a1c09ad03d70ef9568a50456476c826127f06f6c793a5acaab8db78c9b"


# --- redact_uri_credentials -------------------------------------------------
def test_uri_password_is_hidden():
    uri = "mongodb://onyx_shipper:%s@mongo:27017/?replicaSet=rs0&authSource=admin" % SECRET
    out = redact_uri_credentials(uri)
    assert SECRET not in out, out


def test_uri_keeps_scheme_user_host_and_options():
    uri = "mongodb://onyx_shipper:%s@mongo:27017/?replicaSet=rs0&authSource=admin" % SECRET
    out = redact_uri_credentials(uri)
    # діагностична цінність має лишитись: видно кого і куди пускали
    assert out.startswith("mongodb://onyx_shipper:"), out
    assert "@mongo:27017/?replicaSet=rs0&authSource=admin" in out, out
    assert REDACTED in out, out


def test_uri_without_credentials_unchanged():
    uri = "http://ollama:11434/v1"
    assert redact_uri_credentials(uri) == uri


def test_uri_with_user_but_no_password_unchanged():
    # `mongodb://host` без креденшелів не має перетворитись на щось із маскою
    uri = "mongodb://mongo:27017/?replicaSet=rs0"
    out = redact_uri_credentials(uri)
    assert out == uri, out


def test_uri_password_with_percent_encoded_at_sign():
    """Пароль із закодованим '@' (%40) теж має бути прихований повністю."""
    uri = "mongodb://user:p%40ss%2Fword@mongo:27017/?authSource=admin"
    out = redact_uri_credentials(uri)
    assert "p%40ss" not in out, out
    assert "%2Fword" not in out, out
    assert out == "mongodb://user:%s@mongo:27017/?authSource=admin" % REDACTED


def test_redacts_secret_query_parameters_in_service_urls():
    url = "https://api.example.test/v1?model=embed&api_key=top-secret&token=next"

    out = redact_uri_credentials(url)

    assert "top-secret" not in out
    assert "next" not in out
    assert out.count(REDACTED) == 2


def test_non_string_input_passes_through():
    assert redact_uri_credentials(None) is None
    assert redact_uri_credentials(42) == 42


# --- safe_settings_dump -----------------------------------------------------
def _sample_config():
    return {
        "mongo_uri": "mongodb://onyx_shipper:%s@mongo:27017/?replicaSet=rs0" % SECRET,
        "mongo_db": "threat_intel",
        "llm_base_url": "http://ollama:11434/v1",
        "llm_api_key": "sk-or-v1-SUPERSECRET",
        "llm_model": "hermes3:8b",
        "nvd_api_key": None,
        "manticore_url": "http://manticore:9308",
        "batch_size": 50,
        "dedup_similarity_threshold": 0.92,
    }


def test_secret_fields_are_masked():
    out = safe_settings_dump(_sample_config())
    assert out["llm_api_key"] == REDACTED, out["llm_api_key"]


def test_unset_secret_stays_none():
    """None -> None, а не маска: у логах має бути видно, що ключ НЕ заданий."""
    out = safe_settings_dump(_sample_config())
    assert out["nvd_api_key"] is None, out["nvd_api_key"]


def test_uri_fields_are_redacted():
    out = safe_settings_dump(_sample_config())
    assert SECRET not in out["mongo_uri"], out["mongo_uri"]
    assert REDACTED in out["mongo_uri"], out["mongo_uri"]


def test_non_secret_fields_survive_untouched():
    src = _sample_config()
    out = safe_settings_dump(src)
    for key in ("mongo_db", "llm_base_url", "llm_model", "manticore_url",
                "batch_size", "dedup_similarity_threshold"):
        assert out[key] == src[key], key


def test_no_secret_leaks_anywhere_in_the_dump():
    """Головний регрес: жодне значення конфігу не містить секретів цілком."""
    dumped = repr(safe_settings_dump(_sample_config()))
    assert SECRET not in dumped, dumped
    assert "SUPERSECRET" not in dumped, dumped


def test_input_dict_is_not_mutated():
    src = _sample_config()
    safe_settings_dump(src)
    assert SECRET in src["mongo_uri"], "вхідний конфіг не можна псувати на місці"
    assert src["llm_api_key"] == "sk-or-v1-SUPERSECRET"


def test_unknown_secret_looking_field_is_masked():
    """Нове поле з 'password'/'token'/'secret' у назві маскується без правок коду."""
    out = safe_settings_dump({
        "mongo_app_password": SECRET,
        "some_token": "abc",
        "webhook_secret": "xyz",
    })
    assert out["mongo_app_password"] == REDACTED
    assert out["some_token"] == REDACTED
    assert out["webhook_secret"] == REDACTED


# --- інтеграція з точкою входу shipper --------------------------------------
def test_main_logs_config_through_safe_dump():
    """Регрес на саму діру: main.py не має логувати model_dump() напряму."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "shipper", "main.py",
    )
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "safe_settings_dump" in src, "main.py має маскувати конфіг перед логуванням"
    assert 'logger.info("starting", config=settings.model_dump(' not in src, \
        "конфіг логується сирим — пароль Mongo поїде у docker logs"


# --- ручний запуск без pytest ----------------------------------------------
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
