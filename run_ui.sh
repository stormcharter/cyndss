#!/usr/bin/env bash
# Запустити UI ONYX (Streamlit) на хості, у venv. Слухає ЛИШЕ localhost.
# Доступ з браузера — через SSH-тунель з твоєї машини:
#   ssh -p <SSH_PORT> -L 8501:127.0.0.1:8501 <USER>@<SERVER_IP>
#   потім відкрити http://localhost:8501
#
# Запуск:  bash run_ui.sh
set -euo pipefail
cd "$(dirname "$0")"

# Дістати значення ключа з .env (не source-имо файл цілком: там є LLM_MODEL, який
# перебив би значення з systemd-юніта onyx-ui).
env_get() { grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- || true; }

# --- Хостові ендпойнти (контейнери прокинуто на localhost сервера) ---
# UI живе на ХОСТІ, тож у Mongo ходить тим самим рядком, що й колектори: порт 27018
# + креденшели onyx_app. Хардкодити тут не можна — з --auth підключення без пароля відлетить.
export MONGO_URI="${MONGO_URI:-$(env_get COLLECTOR_MONGO_URI)}"
if [ -z "${MONGO_URI}" ]; then
  echo "run_ui.sh: у .env немає COLLECTOR_MONGO_URI — Mongo вимагає автентифікації, зупиняюсь" >&2
  exit 1
fi
export MONGO_DB="threat_intel"
export MONGO_COLLECTION="raw_events"
export MANTICORE_HOST="127.0.0.1"
export MANTICORE_PORT="9999"
export OLLAMA_URL="http://127.0.0.1:11434"        # ембеддинги (nomic, локально на CPU)
export OLLAMA_EMBED_MODEL="nomic-embed-text"

# --- LLM (Hermes 3) ---
# ЗА ЗАМОВЧУВАННЯМ — ЛОКАЛЬНА Ollama (без хмари, без ключів, без rate-limit).
# UI на хості → ендпойнт 127.0.0.1, а НЕ ollama:11434 (docker-ім'я бачить лише shipper).
# Модель hermes3:3b вже pull'нуто. Розумніший варіант для аналітика — hermes3 (8B):
#   docker exec <ollama-контейнер> ollama pull hermes3   →   LLM_MODEL=hermes3 (+ підняти mem_limit).
# Щоб перемкнути UI назад у хмару (OpenRouter), експортни ПЕРЕД запуском:
#   export LLM_BASE_URL=https://openrouter.ai/api/v1
#   export LLM_MODEL=nousresearch/hermes-3-llama-3.1-405b:free
#   export LLM_API_KEY=sk-or-...        (або поклади рядок LLM_API_KEY= у .env)
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434/v1}"
export LLM_MODEL="${LLM_MODEL:-hermes3:3b}"
# Ключ для локалі не потрібен (Ollama його ігнорує). Якщо не заданий, але у .env є рядок
# LLM_API_KEY (для хмарного режиму) — підхопимо звідти; інакше фолбек 'ollama', щоб не падати.
if [ -z "${LLM_API_KEY:-}" ] && [ -f .env ]; then
  export LLM_API_KEY="$(grep -E '^LLM_API_KEY=' .env | head -1 | cut -d= -f2-)"
fi
export LLM_API_KEY="${LLM_API_KEY:-ollama}"

# --- Додаткові LLM-бекенди для агентів (llm_backends.py) ---
# Без них усі агенти йдуть в ОДНУ модель і «незалежність» зводиться до різних
# промптів. Кожна змінна пробрасується з .env, ЛИШЕ якщо там непорожня, — інакше
# відповідний бекенд просто не зʼявиться в селекторі агента (і UI не зміниться).
for var in GROQ_API_KEY GROQ_BASE_URL GROQ_MODEL \
           OPENROUTER_API_KEY OPENROUTER_BASE_URL OPENROUTER_MODEL \
           OLLAMA_LLM_URL OLLAMA_LLM_MODEL; do
  if [ -z "${!var:-}" ]; then
    value="$(env_get "$var")"
    # Явний if, а не `[ -n ... ] && export`: під `set -e` AND-список, що впав
    # на порожньому значенні, може завалити весь запуск UI.
    if [ -n "$value" ]; then
      export "$var=$value"
    fi
  fi
done
unset var value

# --- venv ---
# shellcheck disable=SC1091
source .venv/bin/activate

# Слухаємо ЛИШЕ localhost — назовні нічого не світимо (доступ через SSH-тунель).
exec streamlit run ui.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
