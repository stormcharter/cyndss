"""
Єдина точка запуску пакетних колекторів.

    python collect.py youtube     # зібрати з YouTube-джерел
    python collect.py websites    # зібрати з веб-сайтів та RSS-стрічок
    python collect.py all         # і те, і інше (за замовчуванням)

Список джерел береться з channels.json (ключі "youtube" і "websites").
Залежності: pip install -r requirements-collectors.txt

Telegram збирається окремо:
    python tg.py               # живий режим (слухає нові пости)
    python migrate_history.py  # історія каналів
"""

import sys

USAGE = "Використання: python collect.py [youtube|websites|all]"


def main() -> None:
    target = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    if target not in ("youtube", "websites", "web", "all"):
        print(USAGE)
        sys.exit(1)

    if target in ("youtube", "all"):
        from collectors import youtube
        youtube.collect()

    if target in ("websites", "web", "all"):
        from collectors import website
        website.collect()


if __name__ == "__main__":
    main()
