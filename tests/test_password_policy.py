"""
Юніт-тести для auth/password_policy.py — вимоги до пароля при СТВОРЕННІ облікового запису.

Основа — власний password_strength_checker (github.com/stormcharter), але з трьома
свідомими змінами; кожна закріплена тестом, щоб не «відкотилась» назад:

  1. Заборона будь-яких двох однакових сусідніх символів (`check_no_repeats`) —
     шкідлива: вона викидає з простору паролів купу нормальних варіантів
     («passphrase» не пройде через 'ss'), тобто ЗМЕНШУЄ ентропію, а не збільшує.
     NIST SP 800-63B прямо радить не накручувати композиційні правила.
     Замість цього ловимо те, що справді слабке: 3+ однакових поспіль
     і послідовності на кшталт 1234 / abcd / qwerty.
  2. Перевірка «логін усередині пароля» була регістрозалежна: логін `admin`
     не ловився в пароді `Admin12345!`. Тепер порівняння без регістру.
  3. Додано блоклист очевидних паролів і токенів проєкту (onyx, mongo…):
     довжина не рятує, якщо пароль — «onyxonyxonyx».

Політика застосовується ЛИШЕ при заведенні/зміні пароля (реєстрації в UI немає —
облікові записи створює адміністратор через tools/create_user.py), тому вона може
бути суворою: незручність одноразова.

Запуск:
    pytest tests/test_password_policy.py -v
або:
    python tests/test_password_policy.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.password_policy import (  # noqa: E402
    MAX_LENGTH,
    MIN_LENGTH,
    check_length,
    check_login_mismatch,
    check_no_long_runs,
    check_no_sequences,
    check_not_common,
    check_specials,
    validate_password,
)

GOOD = "Kremenchuk!Fortec9"   # 17 символів, спецсимвол, без прогонів і послідовностей


def _ok(login, password):
    return validate_password(login, password) == []


# --- базовий сценарій -------------------------------------------------------
def test_good_password_has_no_errors():
    assert validate_password("mykyta", GOOD) == [], validate_password("mykyta", GOOD)


def test_errors_are_human_readable_strings():
    errors = validate_password("mykyta", "short")
    assert errors, "слабкий пароль має дати помилки"
    for e in errors:
        assert isinstance(e, str) and e.strip(), errors


# --- довжина ----------------------------------------------------------------
def test_min_length_is_at_least_12():
    assert MIN_LENGTH >= 12


def test_short_password_rejected():
    assert not check_length("Ab1!x")
    assert not _ok("mykyta", "Ab1!x")


def test_password_of_exactly_min_length_passes_length_check():
    assert check_length("A" * MIN_LENGTH)


def test_absurdly_long_password_rejected():
    """Верхня межа — щоб форма логіну не стала точкою DoS через argon2."""
    assert not check_length("Aa1!" * 1000)
    assert not _ok("mykyta", "Aa1!" * 1000)
    assert MAX_LENGTH >= 64


def test_empty_password_does_not_crash():
    errors = validate_password("mykyta", "")
    assert errors


# --- спецсимволи ------------------------------------------------------------
def test_special_char_required():
    assert not check_specials("Abcdefgh12345")
    assert not _ok("mykyta", "Abcdefgh12345")


def test_check_specials_returns_bool_not_match_object():
    """В оригіналі поверталось re.Match — у тестах це маскує помилки типізації."""
    assert check_specials("abc!") is True
    assert check_specials("abc") is False


# --- прогони й послідовності ------------------------------------------------
def test_two_identical_chars_in_a_row_are_allowed():
    """Свідома відмова від правила оригіналу: 'ss' — не вразливість."""
    assert check_no_long_runs("Passphrase!Kyiv7")
    assert _ok("mykyta", "Passphrase!Kyiv7")


def test_three_identical_chars_in_a_row_rejected():
    assert not check_no_long_runs("Paaassphrase!Kyiv")
    assert not _ok("mykyta", "Paaassphrase!Kyiv")


def test_ascending_letter_sequence_rejected():
    assert not check_no_sequences("Xabcdefghijk!7")


def test_ascending_digit_sequence_rejected():
    assert not check_no_sequences("Zorro!123456xy")


def test_descending_sequence_rejected():
    assert not check_no_sequences("Zorro!654321xy")


def test_keyboard_row_rejected():
    assert not check_no_sequences("Qwerty!Zorro77")


def test_short_incidental_sequence_allowed():
    """'ab' або '12' усередині слова — не привід відхиляти."""
    assert check_no_sequences(GOOD), GOOD


# --- логін усередині пароля -------------------------------------------------
def test_login_inside_password_rejected():
    assert not check_login_mismatch("mykyta", "mykyta!Secret42")


def test_login_check_is_case_insensitive():
    """Регрес на баг оригіналу: різний регістр обходив перевірку."""
    assert not check_login_mismatch("admin", "Admin!Secret4242")
    assert not check_login_mismatch("ADMIN", "xxadmin!Secret42")


def test_empty_login_does_not_reject_everything():
    assert check_login_mismatch("", GOOD)
    assert check_login_mismatch(None, GOOD)


# --- блоклист ---------------------------------------------------------------
def test_common_password_rejected_even_if_long():
    assert not check_not_common("correcthorsebatterystaple")
    assert not _ok("mykyta", "Password123!Password")


def test_project_token_rejected():
    """Назва проєкту — перше, що перебирає атакер."""
    assert not check_not_common("OnyxOnyx!Onyx12")


def test_blocklist_is_case_insensitive():
    assert not check_not_common("PASSWORD")


def test_normal_password_passes_blocklist():
    assert check_not_common(GOOD)


# --- сукупність -------------------------------------------------------------
def test_all_violations_are_reported_at_once():
    """Користувач має побачити ВСІ проблеми, а не виправляти їх по одній."""
    errors = validate_password("admin", "admin")
    assert len(errors) >= 2, errors


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
