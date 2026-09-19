"""
Хешування паролів — argon2id (обгортка над argon2-cffi).

ЧОМУ argon2id, а не sha256 (як у прототипі password_strength_checker):
sha256 рахується мільярдами хешів за секунду на GPU, тому вивантажена база
хешів перебирається за години. Argon2 навмисно повільний І памʼятежерний —
memory-hard функція погано паралелиться на GPU/ASIC. Варіант id — гібрид:
стійкий і до GPU-перебору (argon2d), і до атак по сторонніх каналах (argon2i).
Сіль генерується автоматично й лежить усередині рядка хешу.

Параметри — трохи вище дефолтних: у ONYX буквально 4 облікові записи, вхід
рідкісний, тож ~0.1-0.3 с на перевірку користувач навіть не помітить, а вартість
перебору зростає лінійно.

Обгортка потрібна ще й тому, що argon2-cffi кидає винятки на будь-яку невдачу
(включно з пошкодженим хешем у базі) — а на рівні застосунку це має бути просто
«вхід не вдався», без 500-ки й без витоку деталей у стектрейс.
"""

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

# memory_cost у кібібайтах: 64 МіБ на перевірку. time_cost — кількість проходів.
_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=64 * 1024,
    parallelism=2,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    """Повертає argon2id-рядок (сіль і параметри — усередині нього)."""
    if not isinstance(password, str) or not password:
        raise ValueError("hash_password: порожній пароль")
    return _HASHER.hash(password)


def verify_password(stored_hash, password) -> bool:
    """Перевіряє пароль. Будь-яка проблема (не той пароль, побитий хеш,
    None) — це False, а не виняток."""
    if not isinstance(stored_hash, str) or not stored_hash:
        return False
    if not isinstance(password, str) or not password:
        return False
    try:
        return bool(_HASHER.verify(stored_hash, password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    except Exception:
        # argon2-cffi може кинути й інше на екзотично пошкодженому рядку.
        return False


def needs_rehash(stored_hash) -> bool:
    """Чи хеш зроблений слабшими параметрами, ніж поточні (тоді при
    успішному вході його варто перезаписати). Нерозпізнаний рядок — True."""
    if not isinstance(stored_hash, str) or not stored_hash:
        return True
    try:
        return bool(_HASHER.check_needs_rehash(stored_hash))
    except Exception:
        return True
