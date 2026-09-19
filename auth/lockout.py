"""
Гальмування підбору пароля: експоненційне блокування облікового запису.

НАВІЩО. Форма логіну без обмежень перетворює онлайн-перебір на безкоштовну
операцію: логінів усього чотири, вони вгадуються, а спроб — скільки завгодно.
Argon2 тут не рятує: він робить кожну перевірку дорогою, але платить за неї НАШ
процесор — тобто перебір заодно стає DoS-ом.

ПОЛІТИКА. Перші MAX_ATTEMPTS невдач — без наслідків (людина справді може
помилитись). Далі кожна невдача подвоює паузу: 1 хв, 2, 4, 8… зі стелею
MAX_LOCK_SECONDS.

Стеля — не послаблення, а захист від зворотної атаки: без неї будь-хто, хто
знає логін аналітика, вимикав би йому доступ назавжди, просто вводячи дурню.
Година паузи вже робить перебір безглуздим, а робочий день — ні.

Лічильник обнуляється при успішному вході (це робить store, не цей модуль).
Модуль чистий: сама лише арифметика над датами, тестується без Mongo.
"""

from datetime import datetime, timedelta, timezone

MAX_ATTEMPTS = 5          # невдач до першого блокування
BASE_LOCK_SECONDS = 60    # перша пауза
MAX_LOCK_SECONDS = 3600   # стеля (1 година)

# Скільки подвоєнь має сенс рахувати: далі все одно впираємось у стелю, а
# 2**(велике число) з лічильника в базі — це зависання на рівному місці.
_MAX_DOUBLINGS = 20


def _as_utc(value):
    """datetime -> aware UTC. Не-дата або naive з Mongo обробляються тихо."""
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        # pymongo без tz_aware=True віддає naive UTC.
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def lock_seconds(failed_attempts) -> int:
    """Тривалість блокування для заданої кількості невдач (0 — без блокування)."""
    if not isinstance(failed_attempts, int) or isinstance(failed_attempts, bool):
        return 0
    if failed_attempts < MAX_ATTEMPTS:
        return 0
    doublings = min(failed_attempts - MAX_ATTEMPTS, _MAX_DOUBLINGS)
    return min(BASE_LOCK_SECONDS * (2 ** doublings), MAX_LOCK_SECONDS)


def lock_until(failed_attempts, now: datetime):
    """Момент, до якого діє блокування, або None (якщо блокувати ще рано)."""
    seconds = lock_seconds(failed_attempts)
    if seconds <= 0:
        return None
    return now + timedelta(seconds=seconds)


def is_locked(locked_until, now: datetime) -> bool:
    """Чи діє блокування зараз. Порожнє/сміттєве значення — не блокує."""
    moment = _as_utc(locked_until)
    if moment is None:
        return False
    return moment > _as_utc(now)


def seconds_remaining(locked_until, now: datetime) -> int:
    """Скільки ще чекати (округлення вгору) — для повідомлення користувачу."""
    moment = _as_utc(locked_until)
    if moment is None:
        return 0
    delta = (moment - _as_utc(now)).total_seconds()
    if delta <= 0:
        return 0
    return int(delta) + (1 if delta % 1 else 0)
