import os
import json
import requests
import datetime
import re
from dotenv import load_dotenv
from telethon import TelegramClient, events

# 1. Завантаження налаштувань
load_dotenv()

API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "hermes3"

# Шлях до сховища Obsidian (за замовчуванням — тека Obsidian_Vault/Events поруч зі скриптом)
OBSIDIAN_VAULT_PATH = os.getenv(
    "OBSIDIAN_VAULT_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Obsidian_Vault", "Events"),
)

# 2. Ініціалізація списку каналів
with open('channels.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    CHANNELS_TO_MONITOR = tuple(config.get('channels', []))

client = TelegramClient('bot_session', API_ID, API_HASH)

# 3. Функція аналізу (Ollama)
def analyze_with_llm(text):
    """Аналізує текст за допомогою Ollama для виявлення кібербезпекових подій."""
    prompt = f"""
    Ти експерт із кібербезпеки. Проаналізуй наступне повідомлення і виділи ключову інформацію.
    Відповідь надай СУВОРО у форматі JSON українською мовою.
    
    Структура JSON:
    {{
      "summary": "короткий опис події",
      "cve": ["список CVE, якщо є"],
      "mitre": ["техніки MITRE ATT&CK, якщо згадуються"],
      "severity": "рівень важливості (Критичний — якщо є RCE або атака на держсектор, Високий — якщо активний фішинг або CVE, Середній — для інших вразливостей)",
      "threat_actor": "назва угруповання, якщо відомо"
    }}

    Текст повідомлення:
    {text}
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        return json.loads(response.json().get('response'))
    except Exception as e:
        return {"error": f"Помилка Ollama: {str(e)}"}

# 4. Функція збереження в Obsidian
def save_to_obsidian(analysis_res, original_text, channel_name):
    """Створює Markdown-файл у сховищі Obsidian."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Очищення назви файлу від заборонених символів
    summary_safe = re.sub(r'[\\/*?:"<>|]', "", analysis_res.get('summary', 'event'))[:50]
    filename = f"{timestamp}_{summary_safe}.md"
    filepath = os.path.join(OBSIDIAN_VAULT_PATH, filename)

    content = f"""---
date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
source: "{channel_name}"
severity: "{analysis_res.get('severity')}"
cve: {analysis_res.get('cve', [])}
mitre: {analysis_res.get('mitre', [])}
tags:
  - cyber_event
  - {str(analysis_res.get('severity', 'unknown')).lower()}
---

# {analysis_res.get('summary')}

## Аналіз нейромережі
- **Важливість:** {analysis_res.get('severity')}
- **CVE:** {', '.join(analysis_res.get('cve', [])) if analysis_res.get('cve') else 'Відсутні'}
- **MITRE ATT&CK:** {', '.join(analysis_res.get('mitre', [])) if analysis_res.get('mitre') else 'Не вказано'}
- **Суб\'єкт загрози:** {analysis_res.get('threat_actor', 'Невідомо')}

## Оригінальний текст
> {original_text}

---
#auto_generated
"""
    try:
        if not os.path.exists(OBSIDIAN_VAULT_PATH):
            os.makedirs(OBSIDIAN_VAULT_PATH)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Замітку збережено: {filename}")
    except Exception as e:
        print(f"[-] Помилка збереження в Obsidian: {e}")

# 5. Обробник повідомлень
@client.on(events.NewMessage(chats=CHANNELS_TO_MONITOR))
async def handler(event):
    text = event.raw_text
    if not text or len(text) < 10: 
        return

    chat = await event.get_chat()
    print(f"\n[!] Аналіз допису з каналу: {chat.title}")

    res = analyze_with_llm(text)
    
    if "error" in res:
        print(f"[-] {res['error']}")
    else:
        # Автоматичне збереження
        save_to_obsidian(res, text, chat.title)
        
        print(f"Результат аналізу:")
        print(f"🔹 Подія: {res.get('summary')}")
        print(f"🔹 Важливість: {res.get('severity')}")
        print("-" * 50)

# 6. Запуск
print(f"Юзербот запущений. Очікую повідомлення...")
client.start()
client.run_until_disconnected()