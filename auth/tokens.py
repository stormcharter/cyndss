"""
Опакові токени для «запамʼятати цей пристрій».

СХЕМА. Браузеру віддається випадковий токен (cookie), у Mongo лягає лише його
sha256-відбиток. Наслідки:
  * читання бази не дає увійти — з відбитка токен не відновити;
  * пристрій можна відкликати, видаливши запис (cookie стане недійсною);
  * термін дії й мітка пристрою зберігаються поруч із відбитком.

ЧОМУ ТУТ sha256, А НЕ argon2 (як для паролів). Argon2 захищає СЛАБКИЙ секрет —
той, що вигадала людина, і який реально перебирається. Тут секрет генерує
CSPRNG: 256 біт. Перебір такого не залежить від швидкості хешу — його просто
не існує як загрози. Натомість відбиток рахується на КОЖНОМУ відкритті
сторінки, тож дорога функція тут коштувала б затримки на рівному місці.

Токен НЕ несе в собі даних (не JWT): жодного підпису, жодного payload —
сервер завжди звіряється з базою, тому відкликання діє миттєво.
"""

import hashlib
import hmac
import secrets

# 32 байти -> 256 біт ентропії, ~43 символи в base64url.
TOKEN_BYTES = 32


def new_device_token() -> str:
    """Новий опаковий токен. URL-safe: їде в cookie як є."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def token_fingerprint(token: str) -> str:
    """sha256-відбиток токена (hex). Саме він зберігається в Mongo."""
    if not isinstance(token, str):
        raise ValueError("token_fingerprint: очікується рядок")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_match(token, fingerprint) -> bool:
    """Чи відповідає токен збереженому відбитку.

    Порівняння constant-time: звичайне '==' завершується на першому
    розбіжному байті, і за часом відповіді відбиток можна відновити
    посимвольно.
    """
    if not isinstance(token, str) or not token:
        return False
    if not isinstance(fingerprint, str) or not fingerprint:
        return False
    return hmac.compare_digest(token_fingerprint(token), fingerprint)
