"""
Юніт-тести для auth/hashing.py та auth/tokens.py.

ЧОМУ ДВА РІЗНІ ПІДХОДИ ДО ХЕШУВАННЯ (це не непослідовність, а розрахунок):

  * Пароль вигадує людина — його ентропія низька, тож хеш має бути НАВМИСНО
    повільним і памʼятежерним: argon2id. Інакше вивантажена база хешів
    перебирається на GPU за години.
  * Токен «запамʼятати пристрій» генерує CSPRNG — 256 біт справжньої
    випадковості. Перебирати таке безглуздо незалежно від швидкості хешу, тому
    достатньо sha256: він дешевий, а нам треба шукати по відбитку на кожному
    відкритті сторінки.

У Mongo в обох випадках лягає ЛИШЕ похідна: ані пароля, ані самого токена в
базі немає. Хто прочитає базу (напр. сусід по боксу) — не зможе увійти.

Запуск:
    pytest tests/test_auth_crypto.py -v
або:
    python tests/test_auth_crypto.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.hashing import (  # noqa: E402
    hash_password,
    needs_rehash,
    verify_password,
)
from auth.tokens import (  # noqa: E402
    TOKEN_BYTES,
    new_device_token,
    token_fingerprint,
    tokens_match,
)

PASSWORD = "Kremenchuk!Fortec9"


# --- hashing ----------------------------------------------------------------
def test_hash_does_not_contain_password():
    h = hash_password(PASSWORD)
    assert PASSWORD not in h, h


def test_hash_uses_argon2id():
    """Саме id-варіант: гібрид, стійкий і до GPU, і до side-channel."""
    assert hash_password(PASSWORD).startswith("$argon2id$")


def test_same_password_gives_different_hashes():
    """Різна сіль — інакше однакові паролі видно по однакових хешах."""
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_verify_accepts_correct_password():
    assert verify_password(hash_password(PASSWORD), PASSWORD) is True


def test_verify_rejects_wrong_password():
    assert verify_password(hash_password(PASSWORD), PASSWORD + "x") is False


def test_verify_rejects_empty_password():
    assert verify_password(hash_password(PASSWORD), "") is False


def test_verify_returns_false_on_garbage_hash():
    """Пошкоджений/чужий формат у базі — це відмова у вході, а не 500-ка."""
    for bad in ("", "not-a-hash", "$argon2id$broken", None):
        assert verify_password(bad, PASSWORD) is False, bad


def test_verify_returns_false_on_none_password():
    assert verify_password(hash_password(PASSWORD), None) is False


def test_needs_rehash_false_for_fresh_hash():
    assert needs_rehash(hash_password(PASSWORD)) is False


def test_needs_rehash_true_for_weak_legacy_hash():
    """Хеш зі слабшими параметрами треба переписати при наступному вході."""
    weak = ("$argon2id$v=19$m=8,t=1,p=1$"
            "c29tZXNhbHRzb21lc2FsdA$YnJva2VuaGFzaHZhbHVlMDAwMDAwMDA")
    assert needs_rehash(weak) is True


def test_needs_rehash_true_for_garbage():
    assert needs_rehash("not-a-hash") is True


# --- tokens -----------------------------------------------------------------
def test_token_has_at_least_256_bits_of_entropy():
    assert TOKEN_BYTES >= 32


def test_tokens_are_unique():
    assert len({new_device_token() for _ in range(200)}) == 200


def test_token_is_url_safe():
    """Токен їде в cookie — жодних '=', ';', пробілів чи не-ASCII."""
    for _ in range(20):
        token = new_device_token()
        assert token.isascii(), token
        assert not (set(token) - set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        )), token


def test_fingerprint_hides_the_token():
    token = new_device_token()
    fp = token_fingerprint(token)
    assert token not in fp
    assert len(fp) == 64  # sha256 hex


def test_fingerprint_is_deterministic():
    token = new_device_token()
    assert token_fingerprint(token) == token_fingerprint(token)


def test_different_tokens_give_different_fingerprints():
    assert token_fingerprint(new_device_token()) != token_fingerprint(new_device_token())


def test_tokens_match_accepts_own_fingerprint():
    token = new_device_token()
    assert tokens_match(token, token_fingerprint(token)) is True


def test_tokens_match_rejects_foreign_fingerprint():
    assert tokens_match(new_device_token(), token_fingerprint(new_device_token())) is False


def test_tokens_match_handles_missing_values():
    token = new_device_token()
    assert tokens_match(None, token_fingerprint(token)) is False
    assert tokens_match(token, None) is False
    assert tokens_match("", "") is False


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
