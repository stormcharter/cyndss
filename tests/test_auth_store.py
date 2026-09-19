"""
Юніт-тести для auth/store.py — облікові записи й довірені пристрої в Mongo.

ЯК ТЕСТУЄМО. Замість реальної Mongo — мінімальний фейк колекції. Щоб він не
брехав про поведінку справжньої бази, store навмисно обмежений простими
операціями: find_one/find з ТОЧНИМИ фільтрами, insert_one, update_one з
$set/$inc, delete_one. Жодних $expr, агрегацій чи хитрих операторів — усе, що
фейк не відтворить чесно, у store просто не використовується.

ЩО САМЕ ТУТ ЗАХИЩАЄМО:
  * пароль не потрапляє в базу навіть у вигляді «зворотного» перетворення;
  * невідомий логін і невірний пароль НЕРОЗРІЗНЮВАНІ для атакера (інакше форма
    перетворюється на перелічувач користувачів);
  * заблокований обліковий запис не пускає навіть з правильним паролем;
  * токен пристрою одноразовий: при кожному використанні ротується, тож
    вкрадена cookie перестає діяти, щойно нею скористався хтось один;
  * відкликання пристрою діє миттєво (сервер завжди звіряється з базою).

Запуск:
    pytest tests/test_auth_store.py -v
або:
    python tests/test_auth_store.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.store import AuthStore, DEVICE_TTL_DAYS  # noqa: E402
from auth.lockout import MAX_ATTEMPTS  # noqa: E402

GOOD_PASSWORD = "Kremenchuk!Fortec9"
OTHER_PASSWORD = "Vinnytsia?Bastion4"


# --- мінімальний фейк колекції Mongo ---------------------------------------
class FakeCollection:
    """Підтримує рівно те, чим користується store (точні фільтри + $set/$inc)."""

    def __init__(self):
        self.docs = []
        self._next_id = 1
        self.indexes = []

    # -- допоміжне --
    def _matches(self, doc, flt):
        return all(doc.get(k) == v for k, v in (flt or {}).items())

    # -- API --
    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def find_one(self, flt=None, *args, **kwargs):
        for doc in self.docs:
            if self._matches(doc, flt):
                return dict(doc)
        return None

    def find(self, flt=None, *args, **kwargs):
        return [dict(d) for d in self.docs if self._matches(d, flt)]

    def insert_one(self, doc):
        doc = dict(doc)
        doc.setdefault("_id", self._next_id)
        self._next_id += 1
        self.docs.append(doc)
        return type("R", (), {"inserted_id": doc["_id"]})()

    def update_one(self, flt, update):
        for doc in self.docs:
            if self._matches(doc, flt):
                for key, value in update.get("$set", {}).items():
                    doc[key] = value
                for key, value in update.get("$inc", {}).items():
                    doc[key] = doc.get(key, 0) + value
                return type("R", (), {"modified_count": 1})()
        return type("R", (), {"modified_count": 0})()

    def delete_one(self, flt):
        for i, doc in enumerate(self.docs):
            if self._matches(doc, flt):
                del self.docs[i]
                return type("R", (), {"deleted_count": 1})()
        return type("R", (), {"deleted_count": 0})()

    def count_documents(self, flt=None):
        return len(self.find(flt))


class Clock:
    """Керований годинник — щоб перевіряти терміни без sleep()."""

    def __init__(self, start=None):
        self.now = start or datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now = self.now + timedelta(**kwargs)


def _store(clock=None):
    clock = clock or Clock()
    store = AuthStore(FakeCollection(), FakeCollection(), now_fn=clock)
    return store, clock


def _store_with_user(clock=None):
    store, clock = _store(clock)
    store.create_user("mykyta", GOOD_PASSWORD)
    return store, clock


# --- створення облікових записів -------------------------------------------
def test_create_user_stores_no_plaintext_password():
    store, _ = _store_with_user()
    doc = store.users.find_one({"username": "mykyta"})
    assert GOOD_PASSWORD not in repr(doc), doc
    assert "password" not in doc, doc
    assert doc["password_hash"].startswith("$argon2id$")


def test_create_user_rejects_weak_password():
    store, _ = _store()
    try:
        store.create_user("mykyta", "123")
    except ValueError as e:
        assert "довжин" in str(e).lower() or "символ" in str(e).lower(), str(e)
    else:
        raise AssertionError("слабкий пароль мав бути відхилений")


def test_create_user_rejects_password_containing_login():
    store, _ = _store()
    try:
        store.create_user("mykyta", "mykyta!Secret42")
    except ValueError:
        pass
    else:
        raise AssertionError("пароль із логіном мав бути відхилений")


def test_create_user_rejects_duplicate_username():
    store, _ = _store_with_user()
    try:
        store.create_user("mykyta", OTHER_PASSWORD)
    except ValueError:
        pass
    else:
        raise AssertionError("дублікат логіну мав бути відхилений")


def test_username_is_normalised():
    """'Mykyta ' і 'mykyta' — той самий обліковий запис, інакше зʼявляться двійники."""
    store, _ = _store()
    store.create_user("  Mykyta ", GOOD_PASSWORD)
    assert store.users.find_one({"username": "mykyta"}) is not None
    assert store.authenticate("MYKYTA", GOOD_PASSWORD).ok is True


def test_create_user_rejects_empty_username():
    store, _ = _store()
    for bad in ("", "   ", None):
        try:
            store.create_user(bad, GOOD_PASSWORD)
        except ValueError:
            pass
        else:
            raise AssertionError("порожній логін мав бути відхилений: %r" % (bad,))


# --- автентифікація ---------------------------------------------------------
def test_correct_password_authenticates():
    store, _ = _store_with_user()
    result = store.authenticate("mykyta", GOOD_PASSWORD)
    assert result.ok is True, result
    assert result.user["username"] == "mykyta"


def test_wrong_password_rejected():
    store, _ = _store_with_user()
    assert store.authenticate("mykyta", OTHER_PASSWORD).ok is False


def test_unknown_user_is_indistinguishable_from_wrong_password():
    """Інакше форма логіну стає перелічувачем облікових записів."""
    store, _ = _store_with_user()
    unknown = store.authenticate("nobody", GOOD_PASSWORD)
    wrong = store.authenticate("mykyta", OTHER_PASSWORD)
    assert unknown.ok is False and wrong.ok is False
    assert unknown.reason == wrong.reason, (unknown.reason, wrong.reason)


def test_inactive_user_cannot_log_in():
    store, _ = _store_with_user()
    store.set_active("mykyta", False)
    assert store.authenticate("mykyta", GOOD_PASSWORD).ok is False


def test_failed_attempts_are_counted():
    store, _ = _store_with_user()
    store.authenticate("mykyta", OTHER_PASSWORD)
    store.authenticate("mykyta", OTHER_PASSWORD)
    assert store.users.find_one({"username": "mykyta"})["failed_attempts"] == 2


def test_account_locks_after_max_attempts():
    store, _ = _store_with_user()
    for _ in range(MAX_ATTEMPTS):
        store.authenticate("mykyta", OTHER_PASSWORD)
    result = store.authenticate("mykyta", OTHER_PASSWORD)
    assert result.reason == "locked", result
    assert result.retry_after > 0


def test_locked_account_rejects_even_correct_password():
    """Головна властивість блокування — інакше воно нічого не гальмує."""
    store, _ = _store_with_user()
    for _ in range(MAX_ATTEMPTS):
        store.authenticate("mykyta", OTHER_PASSWORD)
    result = store.authenticate("mykyta", GOOD_PASSWORD)
    assert result.ok is False, result
    assert result.reason == "locked"


def test_lock_expires_with_time():
    store, clock = _store_with_user()
    for _ in range(MAX_ATTEMPTS):
        store.authenticate("mykyta", OTHER_PASSWORD)
    clock.advance(hours=2)
    assert store.authenticate("mykyta", GOOD_PASSWORD).ok is True


def test_successful_login_resets_counter():
    store, _ = _store_with_user()
    store.authenticate("mykyta", OTHER_PASSWORD)
    store.authenticate("mykyta", GOOD_PASSWORD)
    doc = store.users.find_one({"username": "mykyta"})
    assert doc["failed_attempts"] == 0, doc
    assert doc.get("locked_until") is None, doc


def test_successful_login_records_timestamp():
    store, clock = _store_with_user()
    store.authenticate("mykyta", GOOD_PASSWORD)
    assert store.users.find_one({"username": "mykyta"})["last_login_at"] == clock.now


def test_set_password_applies_policy_and_takes_effect():
    store, _ = _store_with_user()
    try:
        store.set_password("mykyta", "weak")
    except ValueError:
        pass
    else:
        raise AssertionError("слабкий пароль мав бути відхилений")
    store.set_password("mykyta", OTHER_PASSWORD)
    assert store.authenticate("mykyta", OTHER_PASSWORD).ok is True
    assert store.authenticate("mykyta", GOOD_PASSWORD).ok is False


def test_set_password_unlocks_account():
    """Адмін скинув пароль — блокування має зникнути разом зі старим паролем."""
    store, _ = _store_with_user()
    for _ in range(MAX_ATTEMPTS):
        store.authenticate("mykyta", "WrongPassword!42")
    store.set_password("mykyta", OTHER_PASSWORD)
    assert store.authenticate("mykyta", OTHER_PASSWORD).ok is True


def test_unlock_releases_the_account_without_changing_password():
    """Адміну потрібен спосіб зняти блокування, не змінюючи пароль людині:
    інакше кожне «забув і натиснув 5 разів» коштує нового пароля."""
    store, _ = _store_with_user()
    for _ in range(MAX_ATTEMPTS):
        store.authenticate("mykyta", OTHER_PASSWORD)
    assert store.authenticate("mykyta", GOOD_PASSWORD).reason == "locked"
    store.unlock("mykyta")
    assert store.authenticate("mykyta", GOOD_PASSWORD).ok is True


def test_unlock_resets_the_attempt_counter():
    store, _ = _store_with_user()
    for _ in range(MAX_ATTEMPTS):
        store.authenticate("mykyta", OTHER_PASSWORD)
    store.unlock("mykyta")
    assert store.users.find_one({"username": "mykyta"})["failed_attempts"] == 0


def test_unlock_unknown_user_is_harmless():
    store, _ = _store_with_user()
    store.unlock("nobody")   # не має кидати


def test_is_active_reflects_current_state():
    """Потрібне для періодичної переперевірки вже відкритої вкладки: без неї
    вимкнений обліковий запис працює, доки людина не закриє браузер."""
    store, _ = _store_with_user()
    assert store.is_active("mykyta") is True
    store.set_active("mykyta", False)
    assert store.is_active("mykyta") is False


def test_is_active_false_for_unknown_user():
    store, _ = _store_with_user()
    assert store.is_active("nobody") is False
    assert store.is_active(None) is False


def test_is_active_normalises_username():
    store, _ = _store_with_user()
    assert store.is_active("  MYKYTA ") is True


# --- довірені пристрої ------------------------------------------------------
def test_remember_device_returns_token_and_stores_only_fingerprint():
    store, _ = _store_with_user()
    token = store.remember_device("mykyta", label="ноутбук")
    doc = store.devices.find_one({"username": "mykyta"})
    assert token, "має повернутись сирий токен для cookie"
    assert token not in repr(doc), doc
    assert len(doc["fingerprint"]) == 64


def test_device_token_authenticates():
    store, _ = _store_with_user()
    token = store.remember_device("mykyta", label="ноутбук")
    result = store.authenticate_device(token)
    assert result.ok is True, result
    assert result.user["username"] == "mykyta"


def test_device_token_rotates_on_use():
    """Одноразовість: вкрадена cookie перестає діяти, щойно нею скористались."""
    store, _ = _store_with_user()
    token = store.remember_device("mykyta", label="ноутбук")
    result = store.authenticate_device(token)
    assert result.token and result.token != token, result
    assert store.authenticate_device(token).ok is False, "старий токен має вмерти"
    assert store.authenticate_device(result.token).ok is True


def test_device_token_expires():
    store, clock = _store_with_user()
    token = store.remember_device("mykyta", label="ноутбук")
    clock.advance(days=DEVICE_TTL_DAYS + 1)
    assert store.authenticate_device(token).ok is False


def test_revoked_device_token_stops_working_immediately():
    store, _ = _store_with_user()
    token = store.remember_device("mykyta", label="ноутбук")
    device = store.list_devices("mykyta")[0]
    store.revoke_device("mykyta", device["_id"])
    assert store.authenticate_device(token).ok is False


def test_cannot_revoke_another_users_device():
    store, _ = _store_with_user()
    store.create_user("olena", OTHER_PASSWORD)
    token = store.remember_device("mykyta", label="ноутбук")
    device = store.list_devices("mykyta")[0]
    store.revoke_device("olena", device["_id"])
    assert store.authenticate_device(token).ok is True, "чужий пристрій не мав зникнути"


def test_device_token_of_inactive_user_rejected():
    store, _ = _store_with_user()
    token = store.remember_device("mykyta", label="ноутбук")
    store.set_active("mykyta", False)
    assert store.authenticate_device(token).ok is False


def test_garbage_device_token_rejected():
    store, _ = _store_with_user()
    for bad in (None, "", "not-a-token", "x" * 43):
        assert store.authenticate_device(bad).ok is False, bad


def test_revoke_all_devices():
    store, _ = _store_with_user()
    first = store.remember_device("mykyta", label="ноутбук")
    second = store.remember_device("mykyta", label="телефон")
    store.revoke_all_devices("mykyta")
    assert store.authenticate_device(first).ok is False
    assert store.authenticate_device(second).ok is False


def test_list_devices_shows_labels_but_not_fingerprints():
    store, _ = _store_with_user()
    store.remember_device("mykyta", label="ноутбук")
    devices = store.list_devices("mykyta")
    assert devices[0]["label"] == "ноутбук"
    assert "fingerprint" not in devices[0], devices[0]


def test_list_devices_is_per_user():
    store, _ = _store_with_user()
    store.create_user("olena", OTHER_PASSWORD)
    store.remember_device("mykyta", label="ноутбук")
    assert store.list_devices("olena") == []


def test_device_use_updates_last_seen():
    store, clock = _store_with_user()
    token = store.remember_device("mykyta", label="ноутбук")
    clock.advance(days=1)
    result = store.authenticate_device(token)
    assert store.list_devices("mykyta")[0]["last_seen_at"] == clock.now
    assert result.ok is True


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
