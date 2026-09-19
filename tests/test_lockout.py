"""
Юніт-тести для auth/lockout.py — гальмування підбору пароля.

НАВІЩО. Форма логіну без обмежень — це офлайн-перебір у режимі онлайн: 4 відомі
логіни й необмежена кількість спроб. Argon2 сам по собі не рятує (він робить
дорогою перевірку, але атакер платить нашим CPU, а не своїм — це ще й вектор DoS).

Політика: після MAX_ATTEMPTS невдач обліковий запис блокується на час, що
подвоюється з кожною наступною невдачею, зі стелею. Стеля потрібна, щоб
зловмисник не міг ЗАБЛОКУВАТИ аналітика назавжди, просто вводячи дурню (DoS
через блокування — реальна проблема таких схем).

Модуль чистий (тільки арифметика над датами), тому перевіряється без Mongo.

Запуск:
    pytest tests/test_lockout.py -v
або:
    python tests/test_lockout.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.lockout import (  # noqa: E402
    BASE_LOCK_SECONDS,
    MAX_ATTEMPTS,
    MAX_LOCK_SECONDS,
    is_locked,
    lock_seconds,
    lock_until,
    seconds_remaining,
)

NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)


# --- тривалість блокування --------------------------------------------------
def test_no_lock_below_threshold():
    for attempts in range(MAX_ATTEMPTS):
        assert lock_seconds(attempts) == 0, attempts


def test_first_lock_equals_base():
    assert lock_seconds(MAX_ATTEMPTS) == BASE_LOCK_SECONDS


def test_lock_doubles_with_each_further_failure():
    assert lock_seconds(MAX_ATTEMPTS + 1) == BASE_LOCK_SECONDS * 2
    assert lock_seconds(MAX_ATTEMPTS + 2) == BASE_LOCK_SECONDS * 4


def test_lock_is_capped():
    """Стеля — щоб блокуванням не можна було вимкнути аналітика назавжди."""
    assert lock_seconds(MAX_ATTEMPTS + 50) == MAX_LOCK_SECONDS
    assert MAX_LOCK_SECONDS >= BASE_LOCK_SECONDS


def test_lock_seconds_never_overflows_on_huge_counter():
    """Лічильник з бази може бути будь-яким — 2**N не має рахуватись взагалі."""
    assert lock_seconds(10 ** 6) == MAX_LOCK_SECONDS


def test_lock_seconds_tolerates_bad_input():
    for bad in (None, -3, "5", 2.7):
        assert lock_seconds(bad) >= 0, bad


# --- момент розблокування ---------------------------------------------------
def test_lock_until_is_none_below_threshold():
    assert lock_until(1, NOW) is None


def test_lock_until_is_now_plus_duration():
    assert lock_until(MAX_ATTEMPTS, NOW) == NOW + timedelta(seconds=BASE_LOCK_SECONDS)


# --- перевірка стану --------------------------------------------------------
def test_not_locked_when_no_timestamp():
    assert is_locked(None, NOW) is False


def test_locked_while_timestamp_in_future():
    assert is_locked(NOW + timedelta(seconds=30), NOW) is True


def test_unlocked_when_timestamp_passed():
    assert is_locked(NOW - timedelta(seconds=1), NOW) is False


def test_naive_datetime_from_mongo_is_treated_as_utc():
    """pymongo без tz_aware віддає naive UTC — не можна падати на порівнянні."""
    naive_future = (NOW + timedelta(seconds=30)).replace(tzinfo=None)
    assert is_locked(naive_future, NOW) is True
    naive_past = (NOW - timedelta(seconds=30)).replace(tzinfo=None)
    assert is_locked(naive_past, NOW) is False


def test_garbage_timestamp_does_not_lock_out_forever():
    """Сміття в полі не має назавжди замкнути обліковий запис."""
    for bad in ("2026-07-22", 12345, object()):
        assert is_locked(bad, NOW) is False, bad


# --- залишок часу для повідомлення користувачу ------------------------------
def test_seconds_remaining_rounds_up():
    assert seconds_remaining(NOW + timedelta(seconds=30.2), NOW) == 31


def test_seconds_remaining_is_zero_when_not_locked():
    assert seconds_remaining(None, NOW) == 0
    assert seconds_remaining(NOW - timedelta(seconds=5), NOW) == 0


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
