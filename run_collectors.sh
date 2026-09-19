#!/usr/bin/env bash
# Пакетні колектори OSINT (RSS/веб) на хості, у venv.
#
# НАВІЩО ЦЕЙ ФАЙЛ. У channels.json давно прописані TheHackersNews і
# BleepingComputer, але в проді вони не збиралися ЖОДНОГО разу — корпус на 100%
# telegram. Це не дрібниця: у телеграм-постах ідентифікатори CVE майже ніколи не
# стоять у тексті, а в стрічках вендорських новин — стоять дослівно. Тобто це
# найдешевший спосіб дати конвеєру матеріал, який проходить заземлення
# (shipper/grounding.py) не з натяжкою, а по-справжньому.
#
# Запуск вручну:
#   bash run_collectors.sh            # websites (RSS + сторінки)
#   bash run_collectors.sh youtube    # youtube-транскрипти
#   bash run_collectors.sh all
#
# За розкладом (user-crontab, sudo НЕ потрібен — `crontab -e`):
#   17 */6 * * * cd ~/onyx && bash run_collectors.sh >> ~/onyx/logs/collect.log 2>&1
# Далі нові документи підхоплює shipper через Mongo Change Stream — окремо
# нічого штовхати не треба.
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-websites}"

# Замок: стрічки тягнуться довго, а cron не питає, чи попередній запуск скінчився.
# Без цього два колектори товчуться в одну Mongo і дублюють роботу (дедуп їх
# відсіє, але LLM-виклики на дублі витратяться).
LOCK="/tmp/onyx-collect-${TARGET}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[skip] $(date -Is): попередній запуск '$TARGET' ще працює — виходжу."
  exit 0
fi

# .env читає сам collectors/base.py (load_dotenv), тож нічого не експортуємо —
# інакше два джерела правди розʼїдуться. Тут лише перевіряємо, що потрібний ключ
# на місці: Mongo на сервері з --auth, і без COLLECTOR_MONGO_URI колектор упаде
# аж на вставці, витративши перед тим увесь обхід стрічок.
if ! grep -qE "^COLLECTOR_MONGO_URI=." .env 2>/dev/null; then
  echo "run_collectors.sh: у .env немає COLLECTOR_MONGO_URI — Mongo вимагає автентифікації, зупиняюсь" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[i] $(date -Is): збираю '$TARGET'"
python collect.py "$TARGET"
echo "[i] $(date -Is): готово"
