#!/usr/bin/env bash
# Підняти серверний стек ONYX: Mongo + Manticore + Ollama(ембеддер) + shipper.
# Hermes 3 тут НЕ запускається — він у хмарі (OpenRouter), див. .env.
#
# Запуск:  bash start_server.sh
set -euo pipefail
cd "$(dirname "$0")"

COMPOSE="docker compose -f docker-compose.server.yml"

if [ ! -f .env ]; then
  echo "[!] Немає .env. Створи його:  cp .env.example .env  &&  nano .env"
  exit 1
fi

echo "[1/3] Піднімаємо контейнери (mongo, manticore, ollama, shipper)..."
$COMPOSE up -d --build

echo "[2/3] Чекаємо Ollama і тягнемо ембеддер nomic-embed-text (~270 МБ)..."
OLLAMA_CID="$($COMPOSE ps -q ollama)"
until docker exec "$OLLAMA_CID" ollama list >/dev/null 2>&1; do
  echo "    ...ollama ще піднімається"
  sleep 2
done
docker exec "$OLLAMA_CID" ollama pull nomic-embed-text

echo "[3/3] Готово. Стан сервісів:"
$COMPOSE ps
echo
echo "Логи shipper:   $COMPOSE logs -f shipper"
echo "Далі — UI:      bash run_ui.sh   (доступ через SSH-тунель, див. DEPLOY_SERVER.md)"
