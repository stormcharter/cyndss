"""
Керування обліковими записами ONYX з командного рядка.

Самореєстрації в UI немає навмисно: сервіс дивиться в інтернет, а користувачів
рівно четверо — форма реєстрації дала б стороннім спосіб завести собі доступ.
Тому облікові записи заводяться тільки звідси, з shell-доступом до сервера.

Приклади:
    python tools/create_user.py add mykyta --full-name "Микита"
    python tools/create_user.py list
    python tools/create_user.py passwd mykyta
    python tools/create_user.py disable mykyta
    python tools/create_user.py enable mykyta
    python tools/create_user.py revoke-devices mykyta

Пароль вводиться інтерактивно (getpass) і НЕ передається аргументом: аргументи
командного рядка видно в `ps` іншим користувачам спільного бокса, а ще вони
осідають у ~/.bash_history.

Адреса Mongo береться з COLLECTOR_MONGO_URI (як у колекторів і UI на хості).
"""

import argparse
import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymongo  # noqa: E402

from auth.password_policy import MIN_LENGTH, validate_password  # noqa: E402
from auth.store import AuthStore, normalize_username  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _store() -> AuthStore:
    uri = os.getenv("COLLECTOR_MONGO_URI") or os.getenv("MONGO_URI")
    if not uri:
        print("Немає COLLECTOR_MONGO_URI / MONGO_URI (перевір .env)", file=sys.stderr)
        sys.exit(2)
    db = pymongo.MongoClient(uri)[os.getenv("MONGO_DB", "threat_intel")]
    store = AuthStore(db["users"], db["devices"])
    store.ensure_indexes()
    return store


def _ask_password(login: str) -> str:
    """Питає пароль двічі й перевіряє політикою до запису в базу."""
    print("Вимоги: від %d символів, спецсимвол, без 1234/qwerty, без логіну."
          % MIN_LENGTH)
    for attempt in range(3):
        password = getpass.getpass("Пароль: ")
        errors = validate_password(login, password)
        if errors:
            print("Слабкий пароль:")
            for error in errors:
                print("  - " + error)
            print("Лишилось спроб: %d" % (2 - attempt))
            continue
        if password != getpass.getpass("Ще раз: "):
            print("Паролі не збігаються.")
            continue
        return password
    print("Відмова.", file=sys.stderr)
    sys.exit(1)


def cmd_add(args) -> None:
    store = _store()
    login = normalize_username(args.username)
    password = _ask_password(login)
    store.create_user(login, password, full_name=args.full_name)
    print("Створено: %s" % login)


def cmd_passwd(args) -> None:
    store = _store()
    login = normalize_username(args.username)
    password = _ask_password(login)
    store.set_password(login, password)
    print("Пароль змінено: %s (блокування знято)" % login)


def cmd_list(args) -> None:
    users = _store().list_users()
    if not users:
        print("Користувачів немає. Заведи: python tools/create_user.py add <логін>")
        return
    print("%-16s %-24s %-8s %-20s %s" % ("ЛОГІН", "ІМʼЯ", "АКТИВ", "ОСТАННІЙ ВХІД", "НЕВДАЧ"))
    for user in users:
        last = user.get("last_login_at")
        print("%-16s %-24s %-8s %-20s %s" % (
            user.get("username", ""),
            (user.get("full_name") or "")[:24],
            "так" if user.get("is_active", True) else "НІ",
            last.strftime("%d.%m.%Y %H:%M") if last else "—",
            user.get("failed_attempts", 0),
        ))


def cmd_unlock(args) -> None:
    _store().unlock(args.username)
    print("Блокування знято: %s (пароль не змінювався)"
          % normalize_username(args.username))


def cmd_disable(args) -> None:
    _store().set_active(args.username, False)
    print("Вимкнено: %s" % normalize_username(args.username))


def cmd_enable(args) -> None:
    _store().set_active(args.username, True)
    print("Увімкнено: %s" % normalize_username(args.username))


def cmd_revoke_devices(args) -> None:
    count = _store().revoke_all_devices(args.username)
    print("Відкликано пристроїв: %d" % count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Облікові записи ONYX")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="створити обліковий запис")
    add.add_argument("username")
    add.add_argument("--full-name", default="")
    add.set_defaults(func=cmd_add)

    passwd = sub.add_parser("passwd", help="змінити пароль")
    passwd.add_argument("username")
    passwd.set_defaults(func=cmd_passwd)

    listing = sub.add_parser("list", help="перелік користувачів")
    listing.set_defaults(func=cmd_list)

    unlock = sub.add_parser("unlock", help="зняти блокування після невдалих спроб")
    unlock.add_argument("username")
    unlock.set_defaults(func=cmd_unlock)

    disable = sub.add_parser("disable", help="вимкнути доступ")
    disable.add_argument("username")
    disable.set_defaults(func=cmd_disable)

    enable = sub.add_parser("enable", help="увімкнути доступ")
    enable.add_argument("username")
    enable.set_defaults(func=cmd_enable)

    revoke = sub.add_parser("revoke-devices", help="відкликати всі довірені пристрої")
    revoke.add_argument("username")
    revoke.set_defaults(func=cmd_revoke_devices)

    args = parser.parse_args()
    try:
        args.func(args)
    except ValueError as e:
        print("Помилка: %s" % e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
