# Розгортання ONYX на сервері (Ubuntu 22.04, без GPU)

Цільовий сервер: `<SERVER_IP>`, SSH порт **<SSH_PORT>**, користувач `<USER>` (у прикладах нижче — саме ці плейсхолдери).
Бокс **спільний** — поряд працюють чужі Elasticsearch / Kibana / Neo4j / системний MongoDB
(порт 27017) / Apache (порт 80). **Нічого чужого не чіпаємо.** Усі наші сервіси слухають лише
`127.0.0.1`, наш Mongo — на порту **27018**.

Архітектура інференсу:
- **Ембеддинги** (`nomic-embed-text`) — локальна Ollama на CPU (легкий енкодер, GPU не потрібен).
- **LLM** (`Hermes 3`) — у хмарі через **OpenRouter** (OpenAI-сумісний API), бо на CPU генерація надто повільна.

---

## 1. Docker (узгоджено з власником бокса)
```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER     # потім перелогінься (exit -> знову ssh) або: newgrp docker
docker --version && docker compose version
```
> ⚠️ НЕ робимо `apt full-upgrade`, `reboot`, `ufw enable` — бокс спільний, це зона власника.

## 2. Код на сервер
```bash
git clone <repo> ~/onyx && cd ~/onyx
# або зі своєї машини:  scp -P <SSH_PORT> -r <локальна_тека_проєкту> <USER>@<SERVER_IP>:~/onyx
```

## 3. .env (секрети)
```bash
cp .env.example .env
nano .env     # вписати API_ID/API_HASH, LLM_API_KEY (ключ OpenRouter з openrouter.ai/keys), за бажанням NVD_API_KEY
```

## 3.1 Автентифікація Mongo (обовʼязково — бокс спільний)
Бінд на `127.0.0.1` закриває лише мережу ззовні. Будь-хто з shell-доступом на машині
ходить у loopback вільно: читає весь CTI-корпус і **пише** в нього. Тому Mongo працює з
`--auth`.

Порядок має значення: користувачів створюємо, **поки `--auth` ще вимкнено** (localhost
exception діє лише доки в базі немає жодного облікового запису).
```bash
# 1. підняти стек БЕЗ --auth (розділ 5), потім:
bash tools/setup_mongo_auth.sh        # keyFile + onyxroot/onyx_app + креденшели у .env
# 2. увімкнути --auth (він уже прописаний у docker-compose.server.yml) і перестворити:
docker compose -f docker-compose.server.yml up -d mongo shipper
sudo systemctl restart onyx-ui
```
Деталі:
- **keyFile обовʼязковий.** `--auth` на replica set вимагає внутрішньої автентифікації між
  учасниками — навіть коли учасник один. Без keyFile mongod просто не стартує.
  Файл має бути `400` і належати `uid 999` (користувач `mongo` у контейнері); скрипт
  виставляє це через одноразовий контейнер, щоб не потребувати `sudo` на хості.
- `secrets/` (keyFile + root-пароль) — у `.gitignore`, як і `.env`.
- Ролі `onyx_app`: `readWrite@threat_intel` (Change Stream входить у `read`) +
  `clusterMonitor@admin` — рівно щоб healthcheck міг викликати `rs.status()`.
- Перевірка, що діра закрита:
  ```bash
  mongosh "mongodb://127.0.0.1:27018/?directConnection=true" --quiet \
    --eval 'db.getSiblingDB("threat_intel").raw_events.countDocuments({})'   # має впасти: Unauthorized
  ```

## 3.2 Облікові записи UI
Сервіс дивиться в інтернет, тож вхід закритий паролем. **Самореєстрації немає
навмисно** — форма реєстрації дала б стороннім спосіб завести собі доступ.
Користувачів заводить адміністратор із shell:
```bash
source .venv/bin/activate
python tools/create_user.py add <login> --full-name "<Ім'я>"   # пароль спитає інтерактивно
python tools/create_user.py list
python tools/create_user.py passwd <login>        # зміна пароля (знімає блокування)
python tools/create_user.py disable <login>       # вимкнути доступ
python tools/create_user.py revoke-devices <login>
```
Як влаштовано (деталі — у docstring-ах `auth/`):
- пароль — argon2id (memory-hard; sha256 перебирається на GPU мільярдами хешів/с);
- після 5 невдач обліковий запис блокується з подвоєнням паузи (стеля — година,
  щоб блокуванням не можна було вимкнути аналітика назавжди);
- «невірний пароль» і «немає такого логіна» відповідають однаково й за однаковий
  час — інакше форма перелічує облікові записи;
- «запамʼятати пристрій»: у cookie опаковий токен, у базі лише його sha256;
  токен ротується при кожному використанні, пристрої видно й можна відкликати.

> **Обмеження, про яке треба знати:** cookie ставиться з JavaScript (Streamlit не
> дає доступу до заголовків відповіді), тож HttpOnly недосяжний — XSS в UI
> означає крадіжку токена пристрою. Через це токен одноразовий і відкликається.

## 4. Python venv (для UI та колекторів — вони працюють на хості, поза Docker)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-ui.txt -r requirements-collectors.txt
```

## 5. Підняти стек (Mongo + Manticore + Ollama-ембеддер + shipper)
```bash
bash start_server.sh      # підніме контейнери і стягне nomic-embed-text (~270 МБ)
docker compose -f docker-compose.server.yml ps
docker compose -f docker-compose.server.yml logs -f shipper   # перевірити, що збагачення йде
```
Mongo-replica set `rs0` ініціюється автоматично (healthcheck при першому старті).

## 5.1 Перерахунок векторів індексу (один раз, при оновленні)

Потрібен, якщо індекс зібрала версія shipper, що рахувала вектор від сирого
оригіналу посту. Нова рахує його від англійського «title. summary»
(`embedding_text.py`), і змішаний індекс — це вектори двох різних просторів в
одній таблиці. Тому shipper **відмовиться стартувати**, поки індекс не
перераховано: позначка схеми лежить у `threat_intel.shipper_state`
(`_id: "index_embedding"`). На чистому, порожньому індексі він ставить її сам.

Порядок важливий — старий shipper не повинен писати в індекс, поки той
перераховується:

```bash
docker compose -f docker-compose.server.yml stop shipper     # 1. зупинити старий (і будь-який replay)
git pull                                                     # 2. новий код
docker compose -f docker-compose.server.yml build shipper    # 3. новий образ
# 4. резервна копія індексу: старі вектори інакше не відновити
docker compose -f docker-compose.server.yml stop manticore
docker run --rm -v onyx_manticore-data:/d:ro -v "$PWD":/b alpine \
    tar czf /b/manticore-before-reembed.tgz -C /d .
docker compose -f docker-compose.server.yml up -d manticore
source .venv/bin/activate
python tools/reembed_index.py                                # 5. сухий прогін
python tools/reembed_index.py --apply                        # 6. перерахунок + позначка схеми
docker compose -f docker-compose.server.yml up -d shipper    # 7. новий shipper стартує
```

Ім'я тому (`onyx_manticore-data`) — за назвою теки проєкту; перевірити:
`docker volume ls | grep manticore`.

Сухий прогін друкує дві схеми — ту, яку запише інструмент, і ту, яку чекає
shipper. Вони мусять збігатися, інакше shipper не стартує й після перерахунку.

Інструмент відмовиться писати, поки оренду конвеєра тримає shipper. Replay
оренди не бере, тому на кроці 1 його теж треба зупинити: інакше він може
дописувати документи, поки індекс перераховується. Повторний запуск нічого не
ламає — вектори просто перерахуються наново.

Якщо в індексі є рядки, для яких у Mongo немає оригіналу, їх неможливо
перерахувати, і позначку схеми інструмент не запише. Такі рядки й так зламані
(UI бере текст із Mongo за `mongo_id`). Інструмент виведе їх повним списком;
прибрати — тим самим запуском із `--apply --drop-orphans`.

Якщо крок 5 забули, новий shipper під `restart: unless-stopped` перезапускається
по колу з `index_scheme_mismatch` у логах. Це безпечно: перевірка стоїть до
оренди, до запису і до завантаження MITRE, тож він нічого не пише й нічого не
тягне з мережі. Лікується кроком 5.

**Повернути загублене.** Стара версія відкидала як «дублі» різні пости (на
локальному корпусі — 462 з 1486) і губила довгі (15). Усі вони лишились у Mongo.
Переграти події, яких немає в індексі:

```bash
docker compose -f docker-compose.server.yml exec -T shipper python -m shipper.replay --missing --limit 5
nohup docker compose -f docker-compose.server.yml exec -T shipper \
    python -m shipper.replay --missing > replay_missing.log 2>&1 &
```

Спершу п'ять подій — переконатися, що конвеєр живий. Далі всі: кожна подія
проходить повний конвеєр із викликом LLM, локально це ~5 с на подію, тож на
сотнях — година-дві. Прогрес і підсумок (`replay_missing_complete`) — у
`replay_missing.log`. Два replay одночасно не запускати: вони не блокують один
одного й оброблятимуть ті самі події.

## 6. Колектори (за потреби наповнити базу)
```bash
source .venv/bin/activate
python tg.py                 # живий Telegram-збір; перший запуск — інтерактивна TG-авторизація
# або історичний бекфіл:  python migrate_history.py
```
> `*.session`-файл можна перенести зі своєї машини (`scp`), щоб не проходити авторизацію заново.

## 7. UI + доступ через SSH-тунель
На сервері:
```bash
bash run_ui.sh               # Streamlit слухає лише 127.0.0.1:8501
```
На **твоїй** машині (окремий термінал):
```bash
ssh -p <SSH_PORT> -L 8501:127.0.0.1:8501 <USER>@<SERVER_IP>
```
Потім у браузері: **http://localhost:8501**. Назовні нічого не світиться.

---

## Нотатки
- **Безкоштовний OpenRouter (405B:free)** має жорсткі rate-limit і може бути недоступний пік-годинами.
  Коли зʼявиться бюджет — у `.env` заміни `LLM_MODEL` на `nousresearch/hermes-3-llama-3.1-70b`
  (платний, стабільний). Більше нічого міняти не треба.
- **Памʼять:** на контейнерах стоять `mem_limit` (compose), щоб OOM-killer не зачепив сусідній ES.
- **Оренда конвеєра (lease).** Шиппер бере ексклюзивну оренду в `shipper_state`, щоб два інстанси
  не обробляли той самий потік. При *чистій* зупинці оренда знімається одразу. При `SIGKILL` /
  OOM-килі — ні: контейнер з `restart: unless-stopped` буде рестартувати й падати з
  `another Shipper worker owns the pipeline lease`, поки оренда не спливе (`SHIPPER_LEASE_SECONDS`,
  за замовчуванням 60 с). Це очікувано і самозцілюється — до хвилини crash-loop у логах після
  жорсткого вбивства. Панікувати треба, тільки якщо повідомлення тримається довше за
  `SHIPPER_STALE_SECONDS` (120 с).
- **Не записалось у Manticore** — подія паркується в `threat_intel.shipper_dead_letters`, а не
  зникає. Переграти після інциденту: `docker compose -f docker-compose.server.yml exec shipper
  python -m shipper.replay` (див. `shipper/replay.py --help`).
- **Дебаг-скрипти** (`check_queue.py`, `drop_marker.py`, `raw_events.py`, `reset_manticore.py`,
  `backfill_dedup.py`) усе ще хардкодять `127.0.0.1:27017` / `9306`. На сервері перед використанням
  правити порт на **27018** / **9999** або не запускати — на основний конвеєр вони не впливають.
- **Публічний доступ для демонстрацій** (пізніше, з дозволу власника): зупинити дефолтний Apache,
  поставити nginx reverse-proxy на 80/443 з basic-auth + TLS, відкрити порт у firewall. Поки —
  лише SSH-тунель.
