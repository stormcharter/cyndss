#!/usr/bin/env bash
# Увімкнення автентифікації MongoDB для ONYX (ідемпотентно).
#
# НАВІЩО. Бокс спільний: бінд на 127.0.0.1 захищає лише від мережі ззовні, але будь-хто,
# хто має shell на машині, ходить у loopback вільно — тобто читає весь CTI-корпус і, що
# гірше, ПИШЕ в нього (напр. вставляє собі обліковий запис у колекцію користувачів UI).
#
# ЩО РОБИТЬ:
#   1. генерує keyFile (внутрішня автентифікація replica set — без нього mongod з --auth
#      не стартує навіть для одновузлового rs0);
#   2. створює onyxroot (адміністрування) і onyx_app (застосунок: readWrite@threat_intel
#      + clusterMonitor для healthcheck);
#   3. вписує креденшели у .env (MONGO_APP_USER/PASSWORD + обидва рядки підключення).
#
# Скрипт працює і ДО, і ПІСЛЯ увімкнення --auth: якщо анонімний доступ уже закритий,
# він автентифікується як onyxroot з secrets/mongo-root.pw. Повторний запуск = РОТАЦІЯ
# пароля застосунку (root-пароль і keyFile зберігаються).
# Паролі НЕ друкуються; вони лишаються у .env (0600) і secrets/ (0600).
#
# Запуск:  bash tools/setup_mongo_auth.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"
COMPOSE="docker compose -f docker-compose.server.yml"

[ -f .env ] || { echo "немає .env — спершу cp .env.example .env" >&2; exit 1; }

# --- 1. keyFile ---
mkdir -p secrets
if [ ! -s secrets/mongo-keyfile ]; then
  ( umask 077; openssl rand -base64 756 > secrets/mongo-keyfile )
  # chown на uid 999 (користувач mongo у контейнері) — через контейнер, щоб не смикати sudo.
  docker run --rm -v "$ROOT_DIR/secrets:/s" --entrypoint sh mongo:7 \
    -c 'chown 999:999 /s/mongo-keyfile && chmod 400 /s/mongo-keyfile'
  echo "keyfile: створено"
else
  echo "keyfile: вже існує, лишаю як є"
fi

# --- 2. Паролі (hex → URI-safe, percent-encoding не потрібен) ---
if [ ! -s secrets/mongo-root.pw ]; then
  ( umask 077; openssl rand -hex 32 > secrets/mongo-root.pw )
fi
chmod 600 secrets/mongo-root.pw
ROOT_PW=$(cat secrets/mongo-root.pw)
APP_PW=$(openssl rand -hex 32)

# --- 3. Користувачі ---
# Як заходимо: доки жодного користувача немає, діє localhost exception і креденшели не
# потрібні; після увімкнення --auth — йдемо під onyxroot.
if $COMPOSE exec -T mongo mongosh --quiet --host 127.0.0.1 \
     --eval 'db.adminCommand({listDatabases: 1})' </dev/null >/dev/null 2>&1; then
  AUTH_ARGS=()
  echo "mongosh: анонімний доступ ще відкритий (перший запуск)"
else
  AUTH_ARGS=(--username onyxroot --password "$ROOT_PW" --authenticationDatabase admin)
  echo "mongosh: заходжу під onyxroot (auth уже увімкнено)"
fi

# Паролі йдуть через env контейнера, а не argv (щоб не світились у ps).
# `</dev/null` — інакше `exec` зʼїсть stdin скрипта, коли той запущено як `bash -s`.
$COMPOSE exec -T -e RPW="$ROOT_PW" -e APW="$APP_PW" mongo mongosh --quiet \
  --host 127.0.0.1 "${AUTH_ARGS[@]}" --eval '
  const admin = db.getSiblingDB("admin");
  function upsert(name, roles, pwd) {
    if (admin.getUser(name) === null) {
      admin.createUser({user: name, pwd: pwd, roles: roles});
      print(name + ": створено");
    } else {
      admin.updateUser(name, {pwd: pwd, roles: roles});
      print(name + ": оновлено");
    }
  }
  upsert("onyxroot", [{role: "root", db: "admin"}], process.env.RPW);
  upsert("onyx_app", [{role: "readWrite", db: "threat_intel"},
                      {role: "clusterMonitor", db: "admin"}], process.env.APW);
' </dev/null

# --- 4. .env ---
APP_PW="$APP_PW" ROOT_DIR="$ROOT_DIR" python3 - <<'PY'
import os, re, pathlib
pw = os.environ["APP_PW"]
p = pathlib.Path(os.environ["ROOT_DIR"]) / ".env"
t = p.read_text(encoding="utf-8")

def upsert(text, key, value, anchor="MONGO_URI="):
    pat = re.compile(r'(?m)^%s=.*$' % re.escape(key))
    if pat.search(text):
        return pat.sub(lambda _: f'{key}={value}', text)
    return re.sub(r'(?m)^(%s)' % re.escape(anchor), f'{key}={value}\n\\1', text, count=1)

t = upsert(t, "MONGO_APP_USER", "onyx_app")
t = upsert(t, "MONGO_APP_PASSWORD", pw)
t = upsert(t, "MONGO_URI",
           f'mongodb://onyx_app:{pw}@mongo:27017/?replicaSet=rs0&authSource=admin')
t = upsert(t, "COLLECTOR_MONGO_URI",
           f'mongodb://onyx_app:{pw}@127.0.0.1:27018/?directConnection=true&authSource=admin')
p.write_text(t, encoding="utf-8")
os.chmod(p, 0o600)
print("env: креденшели Mongo записано")
PY

echo
echo "Готово. Далі:"
echo "  1) переконайся, що в docker-compose.server.yml у сервісу mongo є --auth та --keyFile"
echo "  2) docker compose -f docker-compose.server.yml up -d mongo shipper"
echo "     (mongo теж перестворюється: його healthcheck бере пароль з .env на етапі create)"
echo "  3) sudo systemctl restart onyx-ui"
