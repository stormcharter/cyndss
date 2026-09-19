"""
Облікові записи й довірені пристрої ONYX (колекції `users` і `devices` у Mongo).

МОДЕЛЬ ЗАГРОЗ, під яку це писалось:
  * бокс спільний, тому базу може прочитати не тільки застосунок → у Mongo
    лежать лише argon2-хеші паролів і sha256-відбитки токенів; ані пароля, ані
    самого токена там немає;
  * логінів усього чотири й вони вгадуються → онлайн-перебір гальмується
    експоненційним блокуванням (auth/lockout.py);
  * форма логіну не має підказувати, який логін існує → «немає такого
    користувача» і «невірний пароль» дають ІДЕНТИЧНУ відповідь, та ще й із
    однаковою вартістю (для неіснуючого користувача теж рахуємо argon2 —
    інакше атакер відрізнить їх за часом відповіді);
  * cookie «запамʼятати пристрій» можуть вкрасти (XSS, чужий комп) → токен
    ротується при КОЖНОМУ використанні, а список пристроїв видно й відкликається
    миттєво (сервер завжди звіряється з базою, токен нічого не «доводить» сам).

Самореєстрації немає: користувачів заводить адміністратор
(tools/create_user.py). Тому політика пароля застосовується саме тут, при
створенні/зміні, а не при вході.

Store навмисно користується лише простими операціями Mongo (точні фільтри,
$set/$inc) — щоб поведінку можна було чесно відтворити фейком у тестах.
"""

from collections import namedtuple
from datetime import datetime, timedelta, timezone

from .hashing import hash_password, needs_rehash, verify_password
from .lockout import is_locked, lock_until, seconds_remaining
from .password_policy import validate_password
from .tokens import new_device_token, token_fingerprint

# Скільки живе «запамʼятати цей пристрій».
DEVICE_TTL_DAYS = 30

# Результат спроби входу. reason: ok | invalid | locked | disabled.
# token заповнюється лише для входу по пристрою (це НОВИЙ токен після ротації).
AuthResult = namedtuple("AuthResult", "ok user reason retry_after token")


def _fail(reason, retry_after=0):
    return AuthResult(False, None, reason, retry_after, None)


def _utcnow():
    return datetime.now(timezone.utc)


# Хеш-заглушка: перевіряємо його для НЕІСНУЮЧОГО користувача, щоб час відповіді
# не видавав, існує логін чи ні. Рахується один раз при імпорті.
_DUMMY_HASH = hash_password("x" * 24)


def normalize_username(username) -> str:
    """Логіни порівнюються без регістру й країв — інакше зʼявляться двійники."""
    if username is None:
        return ""
    return str(username).strip().lower()


class AuthStore:
    def __init__(self, users, devices, now_fn=_utcnow):
        self.users = users
        self.devices = devices
        self._now = now_fn

    # --- індекси (викликається один раз при старті застосунку) --------------
    def ensure_indexes(self) -> None:
        self.users.create_index("username", unique=True)
        self.devices.create_index("fingerprint", unique=True)
        self.devices.create_index("username")

    # --- користувачі --------------------------------------------------------
    def create_user(self, username, password, full_name=""):
        """Заводить обліковий запис. Кидає ValueError на слабкий пароль,
        порожній чи зайнятий логін."""
        name = normalize_username(username)
        if not name:
            raise ValueError("Порожній логін")
        if self.users.find_one({"username": name}) is not None:
            raise ValueError("Логін %r вже зайнятий" % (name,))
        errors = validate_password(name, password)
        if errors:
            raise ValueError("; ".join(errors))
        doc = {
            "username": name,
            "full_name": str(full_name or ""),
            "password_hash": hash_password(password),
            "is_active": True,
            "created_at": self._now(),
            "failed_attempts": 0,
            "locked_until": None,
            "last_login_at": None,
        }
        self.users.insert_one(doc)
        return doc

    def set_password(self, username, password) -> None:
        """Змінює пароль і знімає блокування (старий пароль більше не діє —
        тримати паузу після цього немає сенсу)."""
        name = normalize_username(username)
        if self.users.find_one({"username": name}) is None:
            raise ValueError("Немає користувача %r" % (name,))
        errors = validate_password(name, password)
        if errors:
            raise ValueError("; ".join(errors))
        self.users.update_one({"username": name}, {"$set": {
            "password_hash": hash_password(password),
            "failed_attempts": 0,
            "locked_until": None,
        }})

    def set_active(self, username, active: bool) -> None:
        self.users.update_one(
            {"username": normalize_username(username)},
            {"$set": {"is_active": bool(active)}},
        )

    def unlock(self, username) -> None:
        """Знімає блокування, НЕ чіпаючи пароль.

        Без цього єдиним способом розблокувати людину була зміна пароля — тобто
        кожне «забув і натиснув пʼять разів» коштувало б нового пароля.
        """
        self.users.update_one(
            {"username": normalize_username(username)},
            {"$set": {"failed_attempts": 0, "locked_until": None}},
        )

    def is_active(self, username) -> bool:
        """Чи існує обліковий запис і чи він увімкнений ЗАРАЗ.

        Потрібне для періодичної переперевірки вже відкритої вкладки: інакше
        вимкнений адміністратором доступ продовжує працювати, доки людина не
        закриє браузер.
        """
        name = normalize_username(username)
        if not name:
            return False
        user = self.users.find_one({"username": name})
        return bool(user and user.get("is_active", True))

    def list_users(self):
        return [
            {k: v for k, v in doc.items() if k != "password_hash"}
            for doc in self.users.find({})
        ]

    def authenticate(self, username, password) -> AuthResult:
        """Перевіряє логін+пароль з урахуванням блокування."""
        name = normalize_username(username)
        user = self.users.find_one({"username": name}) if name else None

        if user is None:
            # Рахуємо хеш-пустушку: без цього неіснуючий логін відповідав би
            # помітно швидше, і форма стала б перелічувачем користувачів.
            verify_password(_DUMMY_HASH, password if isinstance(password, str) else "")
            return _fail("invalid")

        now = self._now()
        if is_locked(user.get("locked_until"), now):
            return _fail("locked", seconds_remaining(user.get("locked_until"), now))

        if not user.get("is_active", True):
            verify_password(_DUMMY_HASH, password if isinstance(password, str) else "")
            return _fail("invalid")

        if not verify_password(user.get("password_hash"), password):
            attempts = int(user.get("failed_attempts") or 0) + 1
            until = lock_until(attempts, now)
            self.users.update_one({"username": name}, {"$set": {
                "failed_attempts": attempts,
                "locked_until": until,
            }})
            if until is not None:
                return _fail("locked", seconds_remaining(until, now))
            return _fail("invalid")

        updates = {"failed_attempts": 0, "locked_until": None, "last_login_at": now}
        # Параметри argon2 з часом посилюються — переписуємо хеш, поки пароль
        # у нас «в руках» (іншої нагоди не буде).
        if needs_rehash(user.get("password_hash")):
            updates["password_hash"] = hash_password(password)
        self.users.update_one({"username": name}, {"$set": updates})

        user.update(updates)
        user.pop("password_hash", None)
        return AuthResult(True, user, "ok", 0, None)

    # --- довірені пристрої --------------------------------------------------
    def remember_device(self, username, label="", ttl_days=DEVICE_TTL_DAYS) -> str:
        """Створює запис пристрою й повертає СИРИЙ токен (єдиний раз — далі в
        базі лише його відбиток)."""
        name = normalize_username(username)
        if self.users.find_one({"username": name}) is None:
            raise ValueError("Немає користувача %r" % (name,))
        token = new_device_token()
        now = self._now()
        self.devices.insert_one({
            "username": name,
            "fingerprint": token_fingerprint(token),
            "label": str(label or "")[:64],
            "created_at": now,
            "last_seen_at": now,
            "expires_at": now + timedelta(days=ttl_days),
        })
        return token

    def authenticate_device(self, token) -> AuthResult:
        """Вхід за cookie. При успіху токен РОТУЄТЬСЯ: новий повертається в
        result.token, старий одразу перестає діяти."""
        if not isinstance(token, str) or not token:
            return _fail("invalid")

        device = self.devices.find_one({"fingerprint": token_fingerprint(token)})
        if device is None:
            return _fail("invalid")

        now = self._now()
        expires_at = device.get("expires_at")
        if not isinstance(expires_at, datetime):
            return _fail("invalid")
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            self.devices.delete_one({"_id": device["_id"]})
            return _fail("invalid")

        user = self.users.find_one({"username": device.get("username")})
        if user is None or not user.get("is_active", True):
            return _fail("invalid")
        # Заблокований підбором обліковий запис не обходиться старою cookie.
        if is_locked(user.get("locked_until"), now):
            return _fail("locked", seconds_remaining(user.get("locked_until"), now))

        fresh = new_device_token()
        self.devices.update_one({"_id": device["_id"]}, {"$set": {
            "fingerprint": token_fingerprint(fresh),
            "last_seen_at": now,
        }})
        user.pop("password_hash", None)
        return AuthResult(True, user, "ok", 0, fresh)

    def list_devices(self, username):
        """Пристрої користувача без відбитків (їх нема навіщо показувати)."""
        name = normalize_username(username)
        return [
            {k: v for k, v in doc.items() if k != "fingerprint"}
            for doc in self.devices.find({"username": name})
        ]

    def revoke_device(self, username, device_id) -> bool:
        """Видаляє пристрій. username у фільтрі — щоб не можна було відкликати
        чужий пристрій, підставивши його id."""
        result = self.devices.delete_one({
            "_id": device_id,
            "username": normalize_username(username),
        })
        return getattr(result, "deleted_count", 0) > 0

    def revoke_all_devices(self, username) -> int:
        name = normalize_username(username)
        count = 0
        while self.devices.delete_one({"username": name}).deleted_count:
            count += 1
        return count
