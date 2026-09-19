import streamlit as st
import pymysql
import pymongo
from bson import ObjectId
import pandas as pd
import plotly.express as px
from datetime import datetime
from pyvis.network import Network
import streamlit.components.v1 as components
import httpx
import re
import os
import html
import uuid
import networkx as nx
from collections import Counter
from itertools import combinations
from twin_scenarios import build_scenarios, parse_infrastructure_model
from twin_defense import rank_defenses
from semantic_reasoner import build_reasoner_from_records, format_hypotheses_for_llm
from hypothesis_labels import build_mitre_label_map, annotate_chain_for_display
from hypothesis_graph import VIRTUAL_START_NODE, build_hypothesis_graph
from search_utils import build_language_filter, build_match_clause, build_or_match_clause
from lemmatizer import lemmatize_query
from chat_intents import (
    INTENT_ANALYSIS, INTENT_CTI_GRAPH, INTENT_HYPOTHESIS_GRAPH,
    INTENT_SEMANTIC_NETWORK, detect_intent,
)
from rag_context import (
    build_hypothesis_document, build_rag_document, decode_upload,
    hypothesis_stats, make_upload_doc, rag_stats,
)
from retrieval import merge_hits
from ui_html import build_verified_tags_html
from prompt_guard import ISOLATION_INSTRUCTION_UK, format_untrusted_context, wrap_untrusted
from auth.streamlit_gate import require_login
from llm_backends import available_backends, resolve_backend
import json

# Налаштування сторінки
st.set_page_config(
    page_title="Threat Intelligence Center",
    page_icon="🛡️",
    layout="wide"
)

# --- ВОРОТА АВТЕНТИФІКАЦІЇ ---
# Стоїть тут навмисно: ДО будь-якого підключення до Mongo/Manticore і до
# першого віджета з даними. Якщо користувач не впізнаний, require_login()
# малює форму входу і зупиняє скрипт (st.stop()) — далі цей файл не виконується.
CURRENT_USER = require_login()

# --- ГЛОБАЛЬНА КОНФІГУРАЦІЯ (env-driven, дефолти — під локальний запуск) ---
# Ембеддинги — локальна Ollama (nomic, CPU). LLM (Hermes 3) — зовнішній
# OpenAI-сумісний API (OpenRouter), бо на сервері без GPU генерація на CPU надто повільна.
# На сервері всі ці значення задаються через env (див. run_ui.sh): MONGO_URI на порт 27018,
# LLM_API_KEY/LLM_BASE_URL тощо. Локально працює без env — на дефолтах.
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")   # лише ембеддинги
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# LLM (Hermes 3) через OpenAI-сумісний chat/completions.
# Локально для роботи через Ollama: LLM_BASE_URL=http://127.0.0.1:11434/v1, LLM_MODEL=hermes3.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "nousresearch/hermes-3-llama-3.1-405b:free")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")


def _llm_headers(api_key: str = LLM_API_KEY) -> dict:
    """Заголовки для OpenAI-сумісного API (з Bearer-ключем, якщо заданий).

    За замовчуванням — глобальний LLM_API_KEY (поточна поведінка для
    перекладача запитів тощо); окремі агенти передають ключ свого бекенду."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-Title"] = "ONYX-CTI"
    return headers


# Функція підключення до Manticore Search
def get_manticore_connection():
    return pymysql.connect(
        host=os.getenv("MANTICORE_HOST", "127.0.0.1"),
        port=int(os.getenv("MANTICORE_PORT", "9999")),
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor
    )

# --- MongoDB як сховище сирих текстів (source of truth) ---
# Архітектура B2: Manticore тримає лише вектор + summary/title для пошуку,
# а повний оригінальний текст не зберігає (content TEXT indexed, без stored).
# Тому повний текст для LLM і для перегляду тягнемо звідси по mongo_id.
# На сервері Mongo прокинуто на 27018 (27017 зайнятий системним mongod) — задається через env.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/?directConnection=true")
MONGO_DB = os.getenv("MONGO_DB", "threat_intel")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "raw_events")


@st.cache_resource
def get_mongo_collection():
    """Кешоване підключення до Mongo (одне на сесію Streamlit)."""
    client = pymongo.MongoClient(MONGO_URI)
    return client[MONGO_DB][MONGO_COLLECTION]


def fetch_original_texts(mongo_ids) -> dict:
    """Повертає {mongo_id: оригінальний_текст}, тягнучи з Mongo одним запитом ($in)."""
    object_ids = []
    for mid in mongo_ids:
        if not mid:
            continue
        try:
            object_ids.append(ObjectId(mid))
        except Exception:
            continue  # пропускаємо нестандартні id (історичні/мігровані документи)
    if not object_ids:
        return {}
    try:
        col = get_mongo_collection()
        cursor = col.find({"_id": {"$in": object_ids}}, {"text": 1})
        return {str(doc["_id"]): doc.get("text", "") for doc in cursor}
    except Exception as e:
        st.warning(f"⚠️ Не вдалося отримати оригінали з MongoDB: {e}")
        return {}

# 1. Отримання вектора для запитання користувача через Ollama
def get_query_embedding(query_text: str) -> list[float] | None:
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": query_text},
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        st.error(f"⚠️ Помилка генерації ембеддинга (Перевірте роботу Ollama): {e}")
        return None

# 2. Синтез відповіді через Hermes 3 на основі реального контексту.
# Промпт більше НЕ захардкоджений — кожен ШІ-агент отримує власний системний промпт.

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

# Базовий промпт CTI-аналітика винесено у prompts/cti_analyst.txt, щоб не загубити
# та використовувати для одного з агентів. Якщо файлу немає — резервна копія нижче.
_FALLBACK_ANALYST_PROMPT = (
    "Ти — провідний аналітик кіберрозвідки (CTI Expert).\n"
    "Твоє завдання — проаналізувати надані документи (Контекст) та відповісти на запитання користувача.\n"
    "ПРАВИЛА АНАЛІЗУ:\n"
    "1. Спирайся ТІЛЬКИ на факти з наданих документів.\n"
    "2. Тобі дозволено робити логічні висновки, якщо вони ПРЯМО випливають із текстів подій.\n"
    "3. КАТЕГОРИЧНО ЗАБОРОНЕНО вигадувати сторонні інциденти, дати чи статистику.\n"
    "4. Якщо документи не стосуються теми, напиши: 'Знайдені джерела не містять релевантної інформації для відповіді.'"
)

# Промпт за замовчуванням для агента-синтезатора (ONYX Synthesizer).
DEFAULT_SYNTH_PROMPT = (
    "Ти — ONYX Synthesizer, головний редактор-синтезатор кіберрозвідки.\n"
    "Тобі надано відповіді кількох незалежних ШІ-агентів на одне й те саме запитання.\n"
    "Твоє завдання — скласти ЄДИНИЙ зведений висновок:\n"
    "1. Виділи спільні факти та точки згоди між агентами.\n"
    "2. Окремо познач розбіжності або унікальні спостереження окремих агентів.\n"
    "3. Не додавай нічого, чого немає у відповідях агентів.\n"
    "4. Дай стислий структурований підсумок для аналітика SOC."
)


def load_prompt_file(filename: str, fallback: str) -> str:
    """Читає системний промпт із prompts/<filename>; повертає fallback, якщо файлу немає."""
    try:
        with open(os.path.join(PROMPTS_DIR, filename), encoding="utf-8") as f:
            text = f.read().strip()
            return text or fallback
    except Exception:
        return fallback


def _format_context(context_docs: list[dict]) -> str:
    """Формує блок контексту з документів для підстановки у системний промпт.

    SECURITY: вміст документів — НЕДОВІРЕНИЙ OSINT (Telegram/YouTube/сайти).
    Кожен документ ізолюється делімітерами (prompt_guard), інакше допис
    зловмисника міг би перехопити керування агентом (indirect prompt injection)."""
    return format_untrusted_context(context_docs)


def _llm_generate(system_prompt: str, user_prompt: str,
                  temperature: float = 0.1, num_predict: int = 500,
                  backend=None) -> str:
    """Виклик LLM через OpenAI-сумісний chat/completions.

    backend=None → поточна поведінка байт-у-байт (глобальні LLM_BASE_URL /
    LLM_API_KEY / LLM_MODEL, тобто OpenRouter/Hermes 3 за замовчуванням).
    Якщо переданий Backend — URL/ключ/модель беруться з нього, щоб різні
    агенти реально ходили в різні моделі різних провайдерів."""
    base_url = backend.base_url if backend is not None else LLM_BASE_URL
    api_key = backend.api_key if backend is not None else LLM_API_KEY
    model = backend.model if backend is not None else LLM_MODEL
    try:
        custom_timeout = httpx.Timeout(180.0, connect=10.0)
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers=_llm_headers(api_key),
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": num_predict,
                "stream": False,
            },
            timeout=custom_timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or [{}]
        content = (choices[0].get("message") or {}).get("content")
        return content or "⚠️ Отримано порожню відповідь від моделі."
    except httpx.TimeoutException:
        return "❌ Помилка: модель Hermes думає занадто довго (таймаут). Спробуйте ще раз."
    except Exception as e:
        return f"❌ Помилка генерації аналітичного звіту: {e}"


def run_agent(query_text: str, context_docs: list[dict], agent_prompt: str, backend=None) -> str:
    """Запускає одного ШІ-агента: його власний промпт + спільний контекст із джерел.

    backend — Backend цього агента (свій base_url/ключ/модель); None → дефолт."""
    system_prompt = (
        agent_prompt.strip()
        + "\n\n" + ISOLATION_INSTRUCTION_UK
        + "\n\nНАДАНИЙ КОНТЕКСТ ДЛЯ АНАЛІЗУ:\n"
        + _format_context(context_docs)
    )
    return _llm_generate(system_prompt, f"Запитання: {query_text}\nАналітичний звіт:", backend=backend)


def run_synthesizer(query_text: str, agent_outputs: list[tuple], synth_prompt: str, backend=None) -> str:
    """Агент-синтезатор: зводить відповіді інших агентів у єдиний висновок."""
    joined = ""
    for name, answer in agent_outputs:
        # Відповіді агентів похідні від недовіреного OSINT — ізолюємо їх так само.
        joined += f"### Відповідь агента «{name}»:\n{wrap_untrusted(answer)}\n\n"
    system_prompt = (synth_prompt.strip() + "\n\n" + ISOLATION_INSTRUCTION_UK
                     + "\n\nВІДПОВІДІ АГЕНТІВ ДЛЯ УЗАГАЛЬНЕННЯ:\n" + joined)
    return _llm_generate(
        system_prompt,
        f"Запитання користувача: {query_text}\nЗведений висновок:",
        temperature=0.2,
        num_predict=600,
        backend=backend,
    )

# 3. Автоматичний технічний перекладач запитів
def translate_query_to_eng(user_query: str) -> str:
    if all(ord(char) < 128 for char in user_query.replace(" ", "")):
        return user_query
        
    try:
        response = httpx.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers=_llm_headers(),
            json={
                "model": LLM_MODEL,
                "messages": [{
                    "role": "user",
                    "content": f"Translate these cyber security search terms into precise English technical keywords. Output ONLY the translation, nothing else: '{user_query}'",
                }],
                "temperature": 0.0,
                "max_tokens": 30,
                "stream": False,
            },
            timeout=20.0
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or [{}]
        translated = (choices[0].get("message") or {}).get("content") or user_query
        return translated.strip().strip("'\"")
    except Exception:
        return user_query
    
# Функція генерації CTI-графа взаємозв'язків
def _split_ids(raw):
    """Розбиває рядок 'CVE-1, CVE-2' у список, безпечно обробляючи None/порожнечу."""
    if not isinstance(raw, str):
        return []
    return [item.strip() for item in raw.split(',') if item.strip()]


def _net_to_html(net):
    """HTML графа РЯДКОМ, без проміжного файлу.

    Навіщо рядок: у режимі «чат» граф має жити в історії повідомлень.
    Перемальовувати pyvis на кожному ререні для кожного повідомлення — це
    секунди затримки, що ростуть із довжиною діалогу. Тому граф будується один
    раз, а в стані сесії лежить готовий HTML.

    Навіщо generate_html, а не звичний net.show(file): show() пише у ФІКСОВАНЕ
    імʼя файлу поруч зі скриптом. На спільному сервері (кілька аналітиків або
    просто кілька вкладок) дві сесії, що будують граф одночасно, ганяються за
    той самий файл — і одна може прочитати чужий HTML. Плюс купа згенерованих
    .html у робочій теці. generate_html() віддає той самий вміст рядком."""
    return net.generate_html(notebook=True)


def build_cti_graph_html(records):
    """Граф зв'язків (пост ↔ CVE / MITRE) -> HTML-рядок або None.

    records — ітерований набір словників із ключами id, title, verified_cve_ids,
    verified_ttp_ids. Формат однаковий і для повнотекстового пошуку, і для RAG —
    тож одна функція обслуговує обидва місця виклику. None означає «немає
    жодного верифікованого звʼязку», а не помилку."""
    net = Network(height="450px", width="100%", bgcolor="#ffffff", font_color="#000000", notebook=True)
    net.barnes_hut(gravity=-1500, central_gravity=0.15, spring_length=120)

    used_nodes = set()
    has_links = False

    for row in records:
        post_id = str(row.get('id') or row.get('mongo_id') or id(row))
        title = row.get('title') or "Без заголовка"
        post_title = title[:30] + "..." if len(title) > 30 else title

        cves = _split_ids(row.get('verified_cve_ids'))
        ttps = _split_ids(row.get('verified_ttp_ids'))

        if not (cves or ttps):
            continue
        has_links = True

        if post_id not in used_nodes:
            net.add_node(post_id, label=post_title, color="#0068c9", size=20, shape="ellipse")
            net.get_node(post_id)["font"] = {"color": "#ffffff", "size": 14}
            used_nodes.add(post_id)

        for cve in cves:
            if cve not in used_nodes:
                net.add_node(cve, label=cve, color="#ff4b4b", size=22, shape="box")
                net.get_node(cve)["font"] = {"color": "#ffffff", "size": 14}
                used_nodes.add(cve)
            net.add_edge(post_id, cve, color="#a0a0a0", weight=2)

        for ttp in ttps:
            ttp_label = f"MITRE {ttp}"
            if ttp_label not in used_nodes:
                net.add_node(ttp_label, label=ttp_label, color="#ffa500", size=22, shape="dot")
                net.get_node(ttp_label)["font"] = {"color": "#1b1b2b", "size": 14}
                used_nodes.add(ttp_label)
            net.add_edge(post_id, ttp_label, color="#a0a0a0", weight=2)

    if not has_links:
        return None

    try:
        return _net_to_html(net)
    except Exception as e:
        st.error(f"⚠️ Не вдалося побудувати граф взаємозв'язків: {e}")
        return None


def _build_cti_graph(records):
    """Сумісна обгортка: будує граф і одразу малює його. Повертає True/False."""
    html_data = build_cti_graph_html(records)
    if not html_data:
        return False
    components.html(html_data, height=470, scrolling=False)
    return True


def render_filtered_cti_graph(dataframe):
    """Будує зведений граф зв'язків тільки для результатів поточного пошуку."""
    records = dataframe.head(20).to_dict("records")
    if not _build_cti_graph(records):
        st.caption("ℹ️ Немає верифікованих CVE/TTP для побудови графа за цим запитом.")


# --- СЕМАНТИЧНА МЕРЕЖА ПОНЯТЬ (горизонтальна мережа співзустрічальності термінів) ---
# Реалізація за методологією горизонтальних семантичних мереж (Д.В. Ланде):
# вузли — значущі терміни/сутності, ребра — семантичні асоціації за спільною
# появою у межах одного документа, вага ребра — частота співзустрічальності.

# Службові слова (UK / RU / EN) та технічний шум, що не несуть смислового навантаження.
SEMANTIC_STOPWORDS = {
    # EN
    "about", "above", "after", "again", "against", "also", "although", "always",
    "among", "another", "because", "been", "before", "being", "below", "between",
    "both", "came", "come", "could", "does", "doing", "done", "down", "during",
    "each", "either", "else", "even", "ever", "every", "from", "further", "have",
    "having", "here", "however", "into", "itself", "just", "like", "made", "make",
    "many", "might", "more", "most", "much", "must", "never", "only", "other",
    "over", "said", "same", "should", "since", "some", "such", "than", "that",
    "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "thus", "under", "until", "upon", "used", "using", "very", "well", "were",
    "what", "when", "where", "which", "while", "whom", "whose", "will", "with",
    "within", "without", "would", "your", "been", "they", "this", "also",
    # RU / UK
    "после", "через", "которые", "более", "может", "можно", "если", "чтобы",
    "этого", "этом", "были", "было", "есть", "тому", "після", "через", "щодо",
    "який", "яка", "яке", "які", "було", "були", "також", "більше", "менше",
    "коли", "цього", "цьому", "проти", "своїх", "своєї", "дуже", "лише",
    # технічний шум
    "https", "http", "www", "com", "net", "org", "amp", "quot", "mitre", "cve",
    "року", "року", "users", "user", "channel", "telegram", "post", "report",
    # загальні дієслова/зв'язки, що не є поняттями
    "various", "discusses", "discuss", "including", "include", "includes",
    "provided", "provides", "provide", "appears", "appear", "article",
    "online", "gain", "gains", "related", "regarding", "involving", "mentions",
    "mentioned", "claims", "claimed", "reportedly", "allegedly", "potential",
    "potentially", "several", "recent", "recently", "highlights", "describes",
    "according", "however", "based", "users",
    # generic-шум (спостережені порушники: found/files рвуть ланцюги гіпотез
    # через побутову лексику, не CTI-концепти — див. приклад "MITRE T1494 →
    # found → ransomware" з поста, ніяк не повʼязаного з ransomware) + весь
    # клас: медіа/файли, час, побутові іменники/дієслова-звʼязки, оцінки.
    # (дубль semantic_reasoner.SEMANTIC_STOPWORDS — тримати синхронними.)
    "found", "find", "finds", "files", "file", "video", "videos", "photo",
    "photos", "image", "images", "anomalies", "anomaly", "unexplained",
    "mysterious", "story", "stories", "news", "today", "yesterday",
    "tomorrow", "data", "info", "information", "thing", "things", "people",
    "world", "place", "places", "point", "points", "case", "cases", "part",
    "parts", "week", "weeks", "month", "months", "year", "years", "hour",
    "hours", "time", "times", "first", "last", "next", "good", "better",
    "best", "real", "main", "full", "open", "close", "high", "long",
    "small", "large", "back", "kind", "type", "need", "needs", "want",
    "wants", "know", "knows", "look", "looks", "show", "shows", "shown",
    "seen", "tell", "tells", "told", "call", "calls", "called", "help",
    "helps", "keep", "kept", "turn", "turns", "move", "moves", "work",
    "works", "worked", "working", "went", "gone", "took", "taken", "give",
    "gives", "given", "way", "ways", "lot", "lots",
    # UK/RU generic-аналоги
    "відео", "фото", "новини", "новина", "сьогодні", "вчора", "завтра",
    "тепер", "дуже",
}


def _extract_terms(text):
    """Витягує значущі терміни-кандидати з тексту: нормалізація + фільтр стоп-слів.

    Лишаються слова довжиною >= 4 (латиниця/кирилиця), без службових слів, чисел
    та CVE-ідентифікаторів (їх додаємо окремо як верифіковані сутності).
    """
    if not text:
        return []
    tokens = re.findall(r"[A-Za-zА-Яа-яЁёЇїІіЄєҐґ][A-Za-zА-Яа-яЁёЇїІіЄєҐґ0-9\-]{3,}", text.lower())
    out = []
    for tok in tokens:
        tok = tok.strip("-")
        if len(tok) < 4:
            continue
        if tok in SEMANTIC_STOPWORDS:
            continue
        if tok.startswith("cve-"):          # верифіковані CVE додаємо окремо
            continue
        if tok.replace("-", "").isdigit():
            continue
        out.append(tok)
    return out


def build_graph_exports(freq, doc_terms, export_terms=120):
    """Будує ПОВНИЙ граф співзустрічальності (networkx) і повертає його у
    форматах для Gephi: {"graphml": …, "gexf": …, "nodes": n, "edges": m}.

    Повний (а не обрізаний до топ-40, як на екрані) — щоб у Gephi можна було
    аналізувати всі звʼязки абстрактно. None, якщо звʼязків не знайшлось."""
    top = {t for t, _ in freq.most_common(export_terms)}
    cooc = Counter()
    for terms in doc_terms:
        present = sorted(terms & top)
        for a, b in combinations(present, 2):
            cooc[(a, b)] += 1
    if not cooc:
        return None

    G = nx.Graph()
    for (a, b), w in cooc.items():
        for node in (a, b):
            if node not in G:
                kind = ("cve" if node.startswith("CVE-")
                        else "ttp" if node.startswith("MITRE ") else "term")
                G.add_node(node, label=node, kind=kind, frequency=int(freq.get(node, 1)))
        G.add_edge(a, b, weight=int(w))

    try:
        return {
            "graphml": "\n".join(nx.generate_graphml(G)),
            "gexf": "\n".join(nx.generate_gexf(G)),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }
    except Exception as e:
        st.caption(f"⚠️ Не вдалося підготувати експорт графа: {e}")
        return None


def render_graph_export_buttons(exports, key_suffix):
    """Кнопки завантаження GraphML / GEXF під графом."""
    if not exports:
        return
    st.caption(
        f"📤 Експорт мережі для Gephi: {exports['nodes']} вузлів, {exports['edges']} звʼязків "
        "(повний граф співзустрічальності — для абстрактного аналізу всіх звʼязків)."
    )
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button(
        "⬇️ GraphML (XML)", exports["graphml"], file_name="onyx_semantic_network.graphml",
        mime="application/xml", key=f"dl_graphml_{key_suffix}", use_container_width=True,
    )
    c2.download_button(
        "⬇️ GEXF (Gephi)", exports["gexf"], file_name="onyx_semantic_network.gexf",
        mime="application/gexf+xml", key=f"dl_gexf_{key_suffix}", use_container_width=True,
    )




def build_semantic_network_artifacts(records, top_terms=40, min_edge=2,
                                     max_docs=200, max_edges=250):
    """Семантична мережа понять -> {"html", "exports", "nodes", "edges"} або None.

    records — набір словників (title, summary, verified_cve_ids, verified_ttp_ids).
    Для документів, докинутих користувачем у чат, title/summary порожні, тому
    терміни беремо ще й із content — інакше файл користувача не потрапив би в
    мережу взагалі."""
    doc_terms = []          # множина термінів на кожен документ
    freq = Counter()        # сумарна частота терміна (для розміру вузла)

    for row in list(records)[:max_docs]:
        text = f"{row.get('title', '') or ''} {row.get('summary', '') or ''}"
        if not text.strip():
            # документ без перекладу (напр. файл користувача) — беремо оригінал
            text = (row.get('content') or '')[:4000]
        terms = set(_extract_terms(text))
        # верифіковані сутності — це «сильні» вузли мережі
        for cve in _split_ids(row.get('verified_cve_ids')):
            terms.add(cve.upper())
        for ttp in _split_ids(row.get('verified_ttp_ids')):
            terms.add(f"MITRE {ttp.upper()}")
        if not terms:
            continue
        doc_terms.append(terms)
        for t in terms:
            freq[t] += 1

    if not freq:
        return None

    # лишаємо лише топ-N найчастотніших термінів — щоб мережа була читабельною
    top = {t for t, _ in freq.most_common(top_terms)}

    cooc = Counter()
    for terms in doc_terms:
        present = sorted(terms & top)
        for a, b in combinations(present, 2):
            cooc[(a, b)] += 1

    if not cooc:
        return None

    edges = [(a, b, w) for (a, b), w in cooc.items() if w >= min_edge]
    if not edges:                                   # малий корпус — лишаємо всі зв'язки
        edges = [(a, b, w) for (a, b), w in cooc.items()]
    edges.sort(key=lambda e: e[2], reverse=True)
    edges = edges[:max_edges]
    if not edges:
        return None

    net = Network(height="600px", width="100%", bgcolor="#ffffff",
                  font_color="#000000", notebook=True)
    net.barnes_hut(gravity=-2200, central_gravity=0.1, spring_length=160,
                   spring_strength=0.02)

    nodes_in_edges = set()
    for a, b, _ in edges:
        nodes_in_edges.add(a)
        nodes_in_edges.add(b)

    for t in nodes_in_edges:
        if t.startswith("CVE-"):
            color, shape, font_color = "#ff4b4b", "box", "#ffffff"      # підпис усередині вузла → білий
        elif t.startswith("MITRE "):
            color, shape, font_color = "#ffa500", "dot", "#1b1b2b"      # підпис під вузлом (на світлому тлі)
        else:
            color, shape, font_color = "#0068c9", "ellipse", "#ffffff"  # синій вузол-термін → білий текст
        size = 12 + min(freq[t] * 4, 40)
        net.add_node(t, label=t, color=color, size=size, shape=shape,
                     title=f"Частота згадувань: {freq[t]}")
        net.get_node(t)["font"] = {"color": font_color, "size": 15}

    for a, b, w in edges:
        net.add_edge(a, b, value=int(w), color="#c8c8c8",
                     title=f"Спільних згадувань: {w}")

    try:
        return {
            "html": _net_to_html(net),
            "exports": build_graph_exports(freq, doc_terms),
            "nodes": len(nodes_in_edges),
            "edges": len(edges),
        }
    except Exception as e:
        st.error(f"⚠️ Не вдалося побудувати семантичну мережу: {e}")
        return None


def _build_semantic_network(records, key, top_terms=40, min_edge=2,
                            max_docs=200, max_edges=250):
    """Сумісна обгортка «побудувати + намалювати» (стрічковий режим).

    key — суфікс ключів віджетів завантаження (Streamlit вимагає унікальні)."""
    art = build_semantic_network_artifacts(records, top_terms,
                                           min_edge, max_docs, max_edges)
    if not art:
        return False
    components.html(art["html"], height=620, scrolling=False)
    render_graph_export_buttons(art["exports"], key)
    return True


# --- ГІПОТЕЗИ / ПРЕДИКТ (traversable-підкладка над семантичною мережею) ---
# Та сама мережа співзустрічальності, що й у візуалізації вище, але тут нею
# МОЖНА ходити: від цільового результату (goal) обхід іде назад до наявних
# свідчень (логіка Ланде «будувати рішення з кінця»). Візуалізацію не чіпаємо —
# це окремий шар інференсу поверх того самого корпусу.
HYPOTHESIS_PROMPT = load_prompt_file(
    "hypothesis_reasoner.txt",
    "Ти — ONYX Predict, аналітик-прогнозист CTI. Поясни надані ланцюги "
    "міркування (start→goal), познач, що спирається на верифіковані факти, і "
    "назви недостаючі ланки. Не вигадуй фактів поза наданими даними.",
)


def _get_reasoner_for_rows(rows):
    """Будує (і кешує в сесії) резонер для поточного корпусу.

    Кеш інвалідовується за сигнатурою (розмір + найновіший collected_at) — тож
    граф не перебудовується на кожен ререн (напр. при кліку кнопки)."""
    sig = (len(rows), rows[0].get("collected_at") if rows else 0)
    cache = st.session_state.get("_reasoner_cache")
    if cache and cache[0] == sig:
        return cache[1]
    reasoner = build_reasoner_from_records(rows)
    st.session_state["_reasoner_cache"] = (sig, reasoner)
    return reasoner


@st.cache_data
def _load_mitre_names() -> dict:
    """Читає плоский словник {id_техніки: назва} з mitre_names.json поряд з
    ui.py (без префікса "MITRE "). Кешується на процес; якщо ассету немає
    або він битий — повертає {} (панель гіпотез не повинна падати без нього)."""
    path = os.path.join(os.path.dirname(__file__), "mitre_names.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data
def _get_mitre_label_map() -> dict:
    """Готовий "MITRE "-map (id вузла графа → читабельна назва) для
    format_hypotheses_for_llm та екранного рендеру ланцюгів. Кешується
    окремо, щоб не пересобирати на кожен ререн панелі."""
    return build_mitre_label_map(_load_mitre_names())


def build_hypothesis_graph_html(hyps, goal_node, label_map):
    """Візуалізує ланцюги гіпотез як напрямлений граф -> HTML-рядок або None.

    Той самий візуальний словник, що й у семантичній мережі: ціль — велика
    фіолетова зірка (тема Amethyst), якірні старти — золото, CVE — червоний
    box, MITRE — золота точка, звичайні терміни — синій. Ребра напрямлені ДО
    цілі, товщина й підпис = вага звʼязку.

    Окремо — ФІКТИВНИЙ вузол «Теперішній стан» (kind="start"): без нього обхід
    від цілі назад малював віяло «5 початків → 1 кінець». Він службовий, тому
    й виглядає інакше: пунктирне сіре ребро без підпису ваги (ваги там нема —
    це не звʼязок із корпусу, а точка відліку)."""
    if not hyps:
        return None
    try:
        G = build_hypothesis_graph(hyps, goal_node, label_map=label_map)
    except Exception as e:
        st.warning(f"⚠️ Не вдалося побудувати граф гіпотез: {e}")
        return None
    if G.number_of_nodes() == 0:
        return None

    net = Network(height="520px", width="100%", bgcolor="#ffffff",
                  font_color="#000000", notebook=True, directed=True)
    net.barnes_hut(gravity=-2600, central_gravity=0.2, spring_length=170,
                   spring_strength=0.03)

    for node, data in G.nodes(data=True):
        kind = data.get("kind", "term")
        label = data.get("label", node)
        is_anchor = data.get("is_anchor", False)
        if kind == "start":
            # службова точка відліку — навмисно нейтральний сірий трикутник,
            # щоб її не сплутали з реальним свідченням із корпусу
            color, shape, font_color, size = "#6b7280", "triangle", "#ffffff", 28
        elif kind == "goal":
            color, shape, font_color, size = "#7b2ff7", "star", "#ffffff", 34
        elif kind == "cve":
            color, shape, font_color, size = "#ff4b4b", "box", "#ffffff", 24
        elif kind == "mitre":
            color, shape, font_color, size = "#ffa500", "dot", "#1b1b2b", 24
        else:
            color, shape, font_color, size = "#0068c9", "ellipse", "#ffffff", 20
        title = ("🎯 Ціль" if kind == "goal" else
                 "🕘 Фіктивний початок: точка відліку, у корпусі такого вузла немає"
                 if kind == "start" else
                 "🔗 Якір (верифікований код MITRE/NVD)" if is_anchor else kind.upper())
        net.add_node(node, label=label, color=color, size=size, shape=shape, title=title)
        font = {"color": font_color, "size": 15}
        if is_anchor and kind != "goal":
            # золоту рамку/жирність якорю — щоб було видно верифіковані старти
            net.get_node(node)["borderWidth"] = 3
            net.get_node(node)["color"] = {"background": color, "border": "#b8860b"}
        net.get_node(node)["font"] = font

    for a, b, data in G.edges(data=True):
        if data.get("virtual"):
            # ребро від фіктивного початку: без ваги й підпису, пунктиром —
            # інакше «0.00» на ребрі читалось би як нульовий реальний звʼязок
            net.add_edge(a, b, value=0.4, title="точка відліку ланцюга",
                         color="#8a8a99", arrows="to", dashes=True)
            continue
        w = data.get("weight", 0.0)
        net.add_edge(a, b, value=float(max(w, 0.1)), title=f"вага звʼязку: {w:.3f}",
                     label=f"{w:.2f}", color="#b0b0b0", arrows="to")

    try:
        return _net_to_html(net)
    except Exception as e:
        st.warning(f"⚠️ Не вдалося відобразити граф гіпотез: {e}")
        return None








# =========================================================================
#  ONYX · багатоагентна оболонка (тема, брендинг, керування ШІ-агентами)
# =========================================================================

# Дві палітри: фірмова темна (Amethyst) та класична світла — перемикач у сайдбарі.
ONYX_PALETTES = {
    "amethyst": {
        "bg": "#0b0b12", "bg2": "#0e0e18",
        "panel": "rgba(20,20,31,0.72)", "panel_solid": "#14141f",
        "border": "rgba(124,58,237,0.28)", "border_strong": "rgba(124,58,237,0.55)",
        "text": "#e7e7f2", "muted": "#9a9ab0",
        "field": "#171723", "btn": "#1d1d2c",
        "accent": "#7c3aed", "accent2": "#4f46e5", "accent3": "#a855f7",
        "glow": "rgba(124,58,237,0.35)",
        "g1": "rgba(124,58,237,0.18)", "g2": "rgba(79,70,229,0.15)",
    },
    "light": {
        "bg": "#f5f5fb", "bg2": "#ffffff",
        "panel": "rgba(255,255,255,0.9)", "panel_solid": "#ffffff",
        "border": "rgba(124,58,237,0.22)", "border_strong": "rgba(124,58,237,0.42)",
        "text": "#1b1b2b", "muted": "#5b5b70",
        "field": "#ffffff", "btn": "#efeafb",
        "accent": "#6d28d9", "accent2": "#4f46e5", "accent3": "#7c3aed",
        "glow": "rgba(124,58,237,0.16)",
        "g1": "rgba(124,58,237,0.10)", "g2": "rgba(99,102,241,0.08)",
    },
}

# Статична частина стилів (літеральні { } — тому НЕ f-рядок; кольори через var()).
ONYX_CSS = """
.stApp{
  background:
    radial-gradient(1100px 520px at 18% -8%, var(--g1), transparent 60%),
    radial-gradient(820px 460px at 102% 0%, var(--g2), transparent 55%),
    var(--onyx-bg);
  color:var(--onyx-text);
}
[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{background:var(--onyx-panel-solid);border-right:1px solid var(--onyx-border);}
[data-testid="stHorizontalBlock"],[data-testid="column"],
[data-testid="stVerticalBlockBorderWrapper"]{overflow:visible !important;}

/* ===== ONYX HERO ===== */
.onyx-hero{position:relative;padding:22px 4px 4px;}
.onyx-logo{font-size:60px;font-weight:800;letter-spacing:.18em;line-height:1;
  background:linear-gradient(92deg,var(--onyx-accent3),var(--onyx-accent2) 60%,#c4b5fd);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  filter:drop-shadow(0 0 22px var(--onyx-glow));}
.onyx-gem{font-size:38px;margin-right:14px;vertical-align:7px;
  background:linear-gradient(180deg,#e9d5ff,var(--onyx-accent));
  -webkit-background-clip:text;background-clip:text;color:transparent;
  animation:onyxpulse 3.4s ease-in-out infinite;}
@keyframes onyxpulse{0%,100%{filter:drop-shadow(0 0 6px var(--onyx-glow));}
  50%{filter:drop-shadow(0 0 18px var(--onyx-accent));}}
.onyx-tag{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.34em;
  text-transform:uppercase;color:var(--onyx-accent3);margin-top:10px;opacity:.92;}
.onyx-sub{color:var(--onyx-muted);font-size:14.5px;margin-top:8px;max-width:780px;line-height:1.5;}
.onyx-rule{height:1px;border:0;margin:16px 0 6px;
  background:linear-gradient(90deg,transparent,var(--onyx-border-strong),transparent);}

.onyx-hero-compact{padding:8px 4px 0;}
.onyx-hero-compact .onyx-logo{font-size:34px;}
.onyx-hero-compact .onyx-gem{font-size:22px;vertical-align:4px;margin-right:9px;}
.onyx-hero-compact .onyx-tag{font-size:10px;letter-spacing:.28em;margin-top:5px;}

/* ===== SIDEBAR BRAND + MODE SWITCH ===== */
.onyx-side-brand{font-size:26px;font-weight:800;letter-spacing:.16em;padding:4px 0 10px;
  background:linear-gradient(92deg,var(--onyx-accent3),var(--onyx-accent2));
  -webkit-background-clip:text;background-clip:text;color:transparent;
  filter:drop-shadow(0 0 14px var(--onyx-glow));}

/* ===== CHAT ===== */
.onyx-chat-hint{border:1px dashed var(--onyx-border-strong);border-radius:14px;
  padding:16px 18px;margin:10px 0 18px;color:var(--onyx-muted);font-size:13.5px;
  background:linear-gradient(120deg,var(--g1),transparent 70%);}
.onyx-chat-hint b{color:var(--onyx-text);}
.onyx-chat-hint code{color:var(--onyx-accent3);background:var(--onyx-field);
  padding:1px 6px;border-radius:6px;}
.onyx-chat-hint ul{margin:9px 0 9px 18px;padding:0;}
.onyx-chat-hint li{margin:3px 0;}
[data-testid="stChatMessage"]{background:var(--onyx-panel);border:1px solid var(--onyx-border);
  border-radius:14px;}
[data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li{color:var(--onyx-text);}
[data-testid="stChatInput"],[data-testid="stChatInput"] textarea{
  background:var(--onyx-field) !important;color:var(--onyx-text) !important;}
[data-testid="stChatInput"]{border:1px solid var(--onyx-border-strong) !important;
  border-radius:14px !important;box-shadow:0 0 22px var(--onyx-glow);}

/* ===== SECTION TITLES ===== */
.onyx-section-title{font-size:21px;font-weight:800;letter-spacing:.02em;margin:8px 0 2px;
  background:linear-gradient(90deg,var(--onyx-text),var(--onyx-accent3));
  -webkit-background-clip:text;background-clip:text;color:transparent;}

/* ===== AGENT CARD ===== */
.onyx-card{position:relative;}
.onyx-card::before{content:"";position:absolute;left:-13px;top:-2px;bottom:-2px;width:3px;
  border-radius:3px;background:linear-gradient(var(--onyx-accent),var(--onyx-accent2));
  box-shadow:0 0 10px var(--c);}
.onyx-card-top{display:flex;align-items:center;gap:9px;}
.onyx-card-icon{font-size:22px;filter:drop-shadow(0 0 8px var(--c));}
.onyx-card-name{font-weight:700;font-size:16px;color:var(--onyx-text);}
.onyx-dot{width:9px;height:9px;border-radius:50%;margin-left:2px;}
.onyx-dot.on{background:#34d399;box-shadow:0 0 8px #34d399;}
.onyx-dot.off{background:#6b7280;}
.onyx-info{margin-left:auto;position:relative;cursor:help;color:var(--onyx-accent3);
  font-size:17px;outline:none;}
.onyx-pop{position:absolute;top:135%;right:0;width:300px;z-index:9999;
  background:var(--onyx-panel-solid);border:1px solid var(--onyx-border-strong);border-radius:12px;
  padding:12px 13px;box-shadow:0 18px 50px rgba(0,0,0,.5),0 0 0 1px var(--onyx-border);
  opacity:0;visibility:hidden;transform:translateY(-6px);transition:.16s ease;
  display:flex;flex-direction:column;gap:5px;text-align:left;}
.onyx-info:hover .onyx-pop,.onyx-info:focus .onyx-pop{opacity:1;visibility:visible;transform:translateY(0);}
.onyx-pop-title{font-weight:700;font-size:14px;color:var(--onyx-text);}
.onyx-pop-desc{font-size:12.5px;color:var(--onyx-muted);}
.onyx-pop-label{font-family:ui-monospace,monospace;font-size:10px;letter-spacing:.2em;
  color:var(--onyx-accent3);margin-top:4px;}
.onyx-pop-prompt{font-size:12px;color:var(--onyx-text);opacity:.85;line-height:1.45;
  max-height:170px;overflow:auto;}
.onyx-card-desc{color:var(--onyx-muted);font-size:12.5px;margin-top:8px;min-height:34px;}

/* ===== SYNTH CARD ===== */
.onyx-synth{position:relative;border-radius:14px;padding:15px 18px;margin-top:8px;
  background:linear-gradient(120deg,rgba(124,58,237,.16),rgba(79,70,229,.10));
  border:1px solid var(--onyx-border-strong);box-shadow:0 0 30px var(--onyx-glow) inset;}
.onyx-synth-top{display:flex;align-items:center;gap:10px;}
.onyx-synth-name{font-weight:800;font-size:17px;
  background:linear-gradient(90deg,var(--onyx-accent3),#c4b5fd);
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.onyx-synth-badge{margin-left:auto;font-family:ui-monospace,monospace;font-size:10.5px;
  letter-spacing:.12em;padding:3px 10px;border-radius:999px;
  border:1px solid var(--onyx-border-strong);color:var(--onyx-accent3);}
.onyx-synth-desc{color:var(--onyx-muted);font-size:13px;margin-top:7px;}

/* ===== RESULT CARDS ===== */
.onyx-result-bar{height:3px;border-radius:3px;margin:-4px 0 10px;
  background:linear-gradient(90deg,var(--c),transparent);}
.onyx-result-head{display:flex;align-items:center;gap:9px;font-weight:700;font-size:15.5px;
  color:var(--onyx-text);margin-bottom:2px;}
.onyx-result-tag{margin-left:auto;font-family:ui-monospace,monospace;font-size:10px;
  letter-spacing:.14em;color:var(--onyx-muted);}
.onyx-summary-head{display:flex;align-items:center;gap:9px;font-weight:800;font-size:16px;margin-bottom:6px;
  background:linear-gradient(90deg,var(--onyx-accent3),#c4b5fd);
  -webkit-background-clip:text;background-clip:text;color:transparent;}

/* ===== BUTTONS ===== */
.stButton>button{border:1px solid var(--onyx-border-strong);background:var(--onyx-btn);
  color:var(--onyx-text);border-radius:10px;transition:.15s;}
.stButton>button:hover{border-color:var(--onyx-accent);filter:brightness(1.06);
  box-shadow:0 0 14px var(--onyx-glow);}
.stButton>button p,.stButton>button div{color:var(--onyx-text);}
.stButton>button[kind="primary"],[data-testid="stBaseButton-primary"]{
  background:linear-gradient(92deg,var(--onyx-accent),var(--onyx-accent2)) !important;
  border:none !important;box-shadow:0 6px 20px var(--onyx-glow);font-weight:700;}
[data-testid="stBaseButton-primary"] p,[data-testid="stBaseButton-primary"] span,
[data-testid="stBaseButton-primary"] div,
[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] p{
  color:#fff !important;-webkit-text-fill-color:#fff !important;}
[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.08);box-shadow:0 8px 26px var(--onyx-glow);}
/* тригери popover (Редагувати / Додати агента / Налаштувати синтезатор) */
[data-testid="stPopoverButton"]{background:var(--onyx-btn) !important;color:var(--onyx-text) !important;
  border:1px solid var(--onyx-border-strong) !important;border-radius:10px !important;}
[data-testid="stPopoverButton"]:hover{border-color:var(--onyx-accent) !important;
  box-shadow:0 0 14px var(--onyx-glow) !important;}
[data-testid="stPopoverButton"] p,[data-testid="stPopoverButton"] div{color:var(--onyx-text) !important;}

/* ===== STREAMLIT WIDGETS (обидві теми — через var()) ===== */
/* підписи / markdown / текст */
label,[data-testid="stWidgetLabel"] p,.stMarkdown,.stMarkdown p,
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
[data-testid="stHeadingWithActionElements"]{color:var(--onyx-text) !important;}
/* капшени та метрики — приглушений колір */
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p{color:var(--onyx-muted) !important;}
[data-testid="stMetricValue"]{color:var(--onyx-text) !important;}
/* поля вводу / textarea / select */
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="base-input"],
[data-baseweb="select"]>div{background:var(--onyx-field) !important;
  border-color:var(--onyx-border) !important;}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea,
[data-baseweb="select"] div{color:var(--onyx-text) !important;}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder{color:var(--onyx-muted) !important;}
/* випадні меню select / multiselect */
[data-baseweb="menu"],[data-baseweb="popover"] [role="listbox"]{background:var(--onyx-field) !important;}
[data-baseweb="menu"] li,[role="option"]{color:var(--onyx-text) !important;}
/* заголовки (вкл. сайдбар «Налаштування ONYX») */
h1,h2,h3,h4,h5,h6{color:var(--onyx-text) !important;}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"]{
  color:var(--onyx-text) !important;}
/* expander: і шапка (summary), і вміст — у кольорах теми */
[data-testid="stExpander"]{background:var(--onyx-field);border-color:var(--onyx-border) !important;}
[data-testid="stExpander"] details{background:var(--onyx-field) !important;}
[data-testid="stExpander"] summary{background:var(--onyx-field) !important;color:var(--onyx-text) !important;}
[data-testid="stExpander"] summary:hover{color:var(--onyx-accent) !important;}
[data-testid="stExpander"] summary svg{fill:var(--onyx-text) !important;}
[data-testid="stPopoverBody"]{background:var(--onyx-panel-solid);}
/* disabled textarea/input (оригінальний текст OSINT-події) — читабельний текст */
textarea:disabled,input:disabled{color:var(--onyx-text) !important;
  -webkit-text-fill-color:var(--onyx-text) !important;opacity:1 !important;}
[data-testid="stTextArea"] textarea:disabled{background:var(--onyx-field) !important;}
"""


def inject_onyx_theme(mode: str):
    """Інʼєктує CSS обраної палітри ONYX (фірмова темна або класична світла)."""
    p = ONYX_PALETTES.get(mode, ONYX_PALETTES["amethyst"])
    root_vars = (
        ":root{"
        f"--onyx-bg:{p['bg']};--onyx-bg2:{p['bg2']};"
        f"--onyx-panel:{p['panel']};--onyx-panel-solid:{p['panel_solid']};"
        f"--onyx-border:{p['border']};--onyx-border-strong:{p['border_strong']};"
        f"--onyx-text:{p['text']};--onyx-muted:{p['muted']};"
        f"--onyx-field:{p['field']};--onyx-btn:{p['btn']};"
        f"--onyx-accent:{p['accent']};--onyx-accent2:{p['accent2']};--onyx-accent3:{p['accent3']};"
        f"--onyx-glow:{p['glow']};--g1:{p['g1']};--g2:{p['g2']};"
        "}"
    )
    st.markdown("<style>" + root_vars + ONYX_CSS + "</style>", unsafe_allow_html=True)


def render_onyx_hero(compact: bool = False):
    """Брендинговий блок ONYX (огранений «онікс» з аметистовим сяйвом).

    compact=True — стисла шапка для режиму чату: там екран має належати
    діалогу, а не банеру на пів-екрана."""
    if compact:
        st.markdown(
            '<div class="onyx-hero onyx-hero-compact">'
            '<div class="onyx-logo"><span class="onyx-gem">&#9670;</span>ONYX</div>'
            '<div class="onyx-tag">Orchestrated Neuro-symbolic Yield &middot; eXpert agents</div>'
            '<hr class="onyx-rule">'
            '</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        '<div class="onyx-hero">'
        '<div class="onyx-logo"><span class="onyx-gem">&#9670;</span>ONYX</div>'
        '<div class="onyx-tag">Orchestrated Neuro-symbolic Yield &middot; eXpert agents</div>'
        '<div class="onyx-sub">Багатоагентна нейро-символічна платформа кіберрозвідки. '
        'Рада незалежних ШІ-агентів аналізує верифіковані джерела, а синтезатор ONYX '
        'зводить їхні висновки в один — кожен агент є окремою <b>гранню</b> онікса.</div>'
        '<hr class="onyx-rule">'
        '</div>',
        unsafe_allow_html=True,
    )


# Готові шаблони ролей для кнопки «+ агент».
AGENT_PRESETS = {
    "🛡️ CTI-аналітик": {
        "icon": "🛡️", "color": "#7c3aed",
        "desc": "Базовий аналіз загроз за джерелами (промпт із cti_analyst.txt)",
        "prompt": load_prompt_file("cti_analyst.txt", _FALLBACK_ANALYST_PROMPT),
    },
    "🔮 Прогнозист загроз": {
        "icon": "🔮", "color": "#a855f7",
        "desc": "Прогноз подальших дій зловмисника та векторів розвитку атаки",
        "prompt": ("Ти — аналітик-прогнозист кіберзагроз. На основі наданих документів спрогнозуй "
                   "найбільш імовірні подальші дії зловмисника, потенційні цілі та вектори розвитку "
                   "атаки. Чітко відокремлюй факти з джерел від обґрунтованих припущень. Не вигадуй "
                   "даних, яких немає в контексті."),
    },
    "🎯 MITRE-тактик": {
        "icon": "🎯", "color": "#6366f1",
        "desc": "Розкладка дій на тактики й техніки MITRE ATT&CK",
        "prompt": ("Ти — експерт із матриці MITRE ATT&CK. Розклади описані в документах дії на тактики "
                   "та техніки ATT&CK (з ідентифікаторами Txxxx, де доречно). Поясни логіку зіставлення. "
                   "Спирайся лише на факти з контексту."),
    },
    "🧩 IOC-екстрактор": {
        "icon": "🧩", "color": "#4f46e5",
        "desc": "Витяг індикаторів компрометації (IP, домени, URL, хеші, CVE)",
        "prompt": ("Ти — інженер з обробки індикаторів компрометації. Витягни з наданих документів усі "
                   "IoC: IP-адреси, домени, URL, хеші, CVE. Згрупуй за типом. Якщо індикаторів немає — "
                   "прямо вкажи це. Нічого не вигадуй."),
    },
    "⚔️ Red Team": {
        "icon": "⚔️", "color": "#ec4899",
        "desc": "Погляд атакуючого: що експлуатувати (навчально-оборонний аналіз)",
        "prompt": ("Ти — оператор Red Team. Поглянь на описані події очима атакуючого: які слабкі місця "
                   "та можливості для експлуатації випливають із джерел. Це навчально-оборонний аналіз. "
                   "Спирайся лише на факти з контексту, не додавай зовнішніх сценаріїв."),
    },
    "🤖 Порожній агент": {
        "icon": "🤖", "color": "#7c3aed",
        "desc": "Власна роль — опиши промпт самостійно",
        "prompt": "Опиши тут роль цього агента та правила аналізу. Агент відповість на запит користувача, спираючись на надані документи.",
    },
}


def init_agent_state():
    """Ініціалізує стан ШІ-агентів: 1 базовий CTI-аналітик + автоматичний синтезатор."""
    if "agents" not in st.session_state:
        st.session_state.agents = [{
            "id": "agent-cti",
            "name": "CTI-аналітик",
            "icon": "🛡️",
            "color": "#7c3aed",
            "desc": "Базовий аналіз загроз за джерелами (промпт із cti_analyst.txt)",
            "prompt": load_prompt_file("cti_analyst.txt", _FALLBACK_ANALYST_PROMPT),
            "enabled": True,
            "backend": "default",
        }]
    if "synth_agent" not in st.session_state:
        st.session_state.synth_agent = {
            "name": "ONYX Synthesizer", "icon": "🧬", "prompt": DEFAULT_SYNTH_PROMPT,
            "backend": "default",
        }


def add_agent(preset_key: str):
    """Додає нового агента за обраним шаблоном (максимум 5)."""
    if len(st.session_state.agents) >= 5:
        return
    p = AGENT_PRESETS[preset_key]
    name = preset_key.split(" ", 1)[1] if " " in preset_key else preset_key
    st.session_state.agents.append({
        "id": "agent-" + uuid.uuid4().hex[:8],
        "name": name, "icon": p["icon"], "color": p["color"],
        "desc": p["desc"], "prompt": p["prompt"], "enabled": True,
        "backend": "default",
    })


def remove_agent(agent_id: str):
    st.session_state.agents = [a for a in st.session_state.agents if a["id"] != agent_id]


def _esc(text: str, limit: int = None) -> str:
    """HTML-екранує текст для вспливних підказок; обрізає до limit символів."""
    text = text or ""
    if limit and len(text) > limit:
        text = text[:limit] + "…"
    return html.escape(text).replace("\n", "<br>")


def _render_backend_selector(current, key: str):
    """Селектор LLM-бекенду для агента: своя модель+провайдер, а не спільна на всіх.

    Якщо в оточенні налаштований лише "default" (немає GROQ/OPENROUTER/OLLAMA
    ключів) — селектор не показуємо, щоб не смітити UI зайвим вибором з одного
    пункту. Повертає ключ обраного бекенду (для збереження в стані агента)."""
    backends = available_backends()
    if len(backends) <= 1:
        return "default"
    names = list(backends.keys())
    if current not in names:
        current = "default"
    return st.selectbox(
        "LLM-бекенд", names, index=names.index(current), key=key,
        format_func=lambda n: backends[n].label,
    )


def render_agent_card(agent: dict):
    """Карточка одного агента: заголовок з hover-поповером + редактор промпта."""
    status_cls = "on" if agent["enabled"] else "off"
    st.markdown(
        f'<div class="onyx-card" style="--c:{agent["color"]}">'
        f'  <div class="onyx-card-top">'
        f'    <span class="onyx-card-icon">{agent["icon"]}</span>'
        f'    <span class="onyx-card-name">{html.escape(agent["name"])}</span>'
        f'    <span class="onyx-dot {status_cls}"></span>'
        f'    <span class="onyx-info" tabindex="0">&#9432;'
        f'      <span class="onyx-pop">'
        f'        <span class="onyx-pop-title">{agent["icon"]} {html.escape(agent["name"])}</span>'
        f'        <span class="onyx-pop-desc">{html.escape(agent["desc"])}</span>'
        f'        <span class="onyx-pop-label">СИСТЕМНИЙ ПРОМПТ</span>'
        f'        <span class="onyx-pop-prompt">{_esc(agent["prompt"], 360)}</span>'
        f'      </span>'
        f'    </span>'
        f'  </div>'
        f'  <div class="onyx-card-desc">{html.escape(agent["desc"])}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with st.popover("✏️ Редагувати", use_container_width=True):
        agent["name"] = st.text_input("Назва", value=agent["name"], key=f"nm_{agent['id']}")
        c1, c2 = st.columns([1, 2])
        agent["icon"] = c1.text_input("Іконка", value=agent["icon"], key=f"ic_{agent['id']}", max_chars=4)
        agent["desc"] = c2.text_input("Опис", value=agent["desc"], key=f"ds_{agent['id']}")
        agent["prompt"] = st.text_area("Системний промпт", value=agent["prompt"], height=200, key=f"pr_{agent['id']}")
        agent["backend"] = _render_backend_selector(agent.get("backend"), key=f"bk_{agent['id']}")
        agent["enabled"] = st.toggle("Активний", value=agent["enabled"], key=f"en_{agent['id']}")
        if st.button("🗑 Видалити агента", key=f"rm_{agent['id']}", use_container_width=True):
            remove_agent(agent["id"])
            st.rerun()


def render_agent_grid():
    """Сітка карток агентів (по 3 у рядку) + кнопка додавання нового."""
    agents = st.session_state.agents
    per_row = 3
    for i in range(0, len(agents), per_row):
        cols = st.columns(per_row, gap="medium")
        for col, agent in zip(cols, agents[i:i + per_row]):
            with col:
                with st.container(border=True):
                    render_agent_card(agent)

    if len(agents) < 5:
        with st.popover(f"➕ Додати агента  ·  {len(agents)}/5", use_container_width=False):
            preset = st.selectbox("Шаблон ролі", list(AGENT_PRESETS.keys()), key="add_preset")
            st.caption(AGENT_PRESETS[preset]["desc"])
            if st.button("Створити агента", key="add_btn", type="primary"):
                add_agent(preset)
                st.rerun()
    else:
        st.caption("Досягнуто максимум — 5 агентів. Видаліть одного, щоб додати іншого.")


def render_synth_card():
    """Карточка агента-синтезатора (ONYX Synthesizer) — завжди присутній, окремо."""
    synth = st.session_state.synth_agent
    n_enabled = sum(1 for a in st.session_state.agents if a["enabled"])
    active = n_enabled > 1
    badge = "ГОТОВИЙ" if active else "ОЧІКУЄ ≥2 АГЕНТІВ"
    st.markdown(
        f'<div class="onyx-synth">'
        f'  <div class="onyx-synth-top">'
        f'    <span style="font-size:22px">{synth["icon"]}</span>'
        f'    <span class="onyx-synth-name">{html.escape(synth["name"])}</span>'
        f'    <span class="onyx-synth-badge">{badge}</span>'
        f'  </div>'
        f'  <div class="onyx-synth-desc">Автоматичний агент-синтезатор. Зводить відповіді всіх '
        f'активних агентів у єдиний висновок — запускається, лише коли відповіли понад один агент.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    with st.popover("✏️ Налаштувати синтезатор", use_container_width=False):
        synth["name"] = st.text_input("Назва", value=synth["name"], key="synth_nm")
        synth["icon"] = st.text_input("Іконка", value=synth["icon"], key="synth_ic", max_chars=4)
        synth["prompt"] = st.text_area("Промпт синтезатора", value=synth["prompt"], height=200, key="synth_pr")
        synth["backend"] = _render_backend_selector(synth.get("backend"), key="synth_bk")




# =========================================================================
#  РЕЖИМ «ЧАТ» — єдиний вхід замість трьох окремих полів і купи кнопок
# =========================================================================
# Було: окреме поле під запит до ради агентів, окреме поле під ціль гіпотез,
# окремі кнопки експорту GraphML/GEXF, і все це одночасно на одному екрані.
# Стало: одне поле вводу. Намір розпізнає chat_intents (без LLM — миттєво й
# детерміновано), а результат лягає в історію діалогу.

# Поля, які тягнемо з індексу для будь-якої вибірки подій (щоб стрічка, чат і
# графи оперували ОДНАКОВИМ набором полів і не розходились).
_EVENT_FIELDS = ("id, mongo_id, title, summary, source_type, channel_name, "
                 "language, verified_cve_ids, verified_ttp_ids, collected_at")

# Поріг косинусної відстані для векторного (KNN) складника гібридного пошуку.
# Свідомо суворий: KNN тут ДОПОВНЮЄ повнотекстовий пошук, а не замінює його —
# усе, що далі за цей поріг, у контексті LLM буде шумом і провокує «галюцинації
# за асоціацією».
KNN_MAX_DISTANCE = 0.30

# Скільки документів іде в контекст однієї відповіді.
CHAT_CONTEXT_LIMIT = 8


def fetch_events(lang_filter, search_query=None, limit=5000):
    """Вибірка подій зі стрічки: фільтр мов + (опційно) повнотекстовий пошук.

    Спільна для режиму «стрічка» і для графових намірів у чаті — щоб «мережа
    понять» будувалась рівно на тому корпусі, який користувач бачить."""
    conn = get_manticore_connection()
    try:
        with conn.cursor() as cursor:
            where_clause = build_language_filter(lang_filter) or "1=1"
            # Лематизація укр. запиту: «вразливості» -> «вразливість», так само як
            # проіндексований content. Без цього україномовний запит в одній формі
            # не знаходив текст в іншій (Manticore не має uk-стемера). Латиниця
            # проходить як є (її морфологію стягує stem_en). Лематизуємо ДО escape.
            # lemmatize_QUERY (не _text): ріже ввід ДО морфоаналізу, щоб довгий
            # запит не ганявся через pymorphy3 цілком (обрізання в search_utils
            # настає вже після — запізно). Тіла документів у shipper НЕ ріжуться.
            # SECURITY: ввід користувача НЕ можна класти в MATCH() сирим — потрібне
            # екранування і SQL-літерала, і full-text операторів (див. search_utils).
            match_clause = build_match_clause(lemmatize_query(search_query),
                                              conn.escape_string)
            if match_clause:
                where_clause += f" AND {match_clause}"

            cursor.execute(f"""
                SELECT {_EVENT_FIELDS}
                FROM events_enriched
                WHERE {where_clause}
                ORDER BY collected_at DESC
                LIMIT {int(limit)}
                OPTION max_matches = {int(limit)}
            """)
            return cursor.fetchall()
    finally:
        conn.close()


def retrieve_context(query_text, lang_filter, limit=CHAT_CONTEXT_LIMIT):
    """Гібридний пошук контексту під запит у чаті.

    ЧОМУ ГІБРИД. Стара агентська гілка робила AND-матчинг усіх слів довших за
    3 символи: «say about vulnerabilities» вимагало від документа ОБИДВА слова,
    давало 1-2 випадкові збіги, і агент чесно писав «джерела не містять
    релевантної інформації». Тут три незалежні стратегії:
      1) OR-матчинг по ЛЕМАТИЗОВАНОМУ оригіналу — б'є в поле content (оригінал,
         переважно українською), ранжування BM25 на боці Manticore;
      2) OR-матчинг по англійському перекладу — б'є в title/summary (їх пише
         LLM англійською), тож укр. запит дістає й англомовні поля;
      3) векторний KNN — ловить перефразування, яких немає дослівно.
    Видачі зливаються круговим обходом (retrieval.merge_hits), щоб жодна
    стратегія не з'їла ліміт цілком.

    Повертає (documents, meta) — meta потрібна UI, щоб чесно показати, який
    саме запит пішов у сховище."""
    meta = {"lemma": "", "translated": "", "strategies": []}
    if not (query_text or "").strip():
        return [], meta

    lemma_q = lemmatize_query(query_text)
    translated = translate_query_to_eng(query_text)
    meta["lemma"] = lemma_q
    meta["translated"] = translated

    groups = []
    conn = get_manticore_connection()
    try:
        with conn.cursor() as cursor:
            base_filter = build_language_filter(lang_filter) or "1=1"

            seen_clauses = set()
            for label, text in (("fts_uk", lemma_q), ("fts_en", translated)):
                clause = build_or_match_clause(text, conn.escape_string)
                if not clause or clause in seen_clauses:
                    continue
                seen_clauses.add(clause)
                try:
                    cursor.execute(f"""
                        SELECT {_EVENT_FIELDS}, 0.0000 AS distance, WEIGHT() AS relevance
                        FROM events_enriched
                        WHERE {clause} AND {base_filter}
                        ORDER BY relevance DESC
                        LIMIT {int(limit)}
                        OPTION max_matches = 1000
                    """)
                    hits = cursor.fetchall()
                except Exception as e:
                    st.caption(f"⚠️ Стратегія {label} не спрацювала: {e}")
                    hits = []
                if hits:
                    meta["strategies"].append(f"{label}:{len(hits)}")
                    groups.append(hits)

            # KNN шукаємо за англійським варіантом: ембеддинги корпусу рахувались
            # по тексту, який здебільшого проходив через той самий переклад.
            vector = get_query_embedding(translated or query_text)
            if vector:
                vec_str = ",".join(map(str, vector))
                try:
                    cursor.execute(f"""
                        SELECT {_EVENT_FIELDS}, knn_dist() AS distance
                        FROM events_enriched
                        WHERE knn(embedding, {int(limit)}, ({vec_str})) AND {base_filter}
                    """)
                    knn_hits = [d for d in cursor.fetchall()
                                if d.get("distance", 1.0) < KNN_MAX_DISTANCE]
                except Exception as e:
                    st.caption(f"⚠️ Векторний пошук не спрацював: {e}")
                    knn_hits = []
                if knn_hits:
                    meta["strategies"].append(f"knn:{len(knn_hits)}")
                    groups.append(knn_hits)
    finally:
        conn.close()

    docs = merge_hits(groups, limit=limit)
    # Повний оригінальний текст живе лише в Mongo (архітектура B2) — тягнемо
    # одним запитом на всю вибірку.
    mongo_texts = fetch_original_texts([d.get("mongo_id") for d in docs])
    for d in docs:
        d["content"] = mongo_texts.get(d.get("mongo_id")) or d.get("summary") or ""
    return docs, meta


# --- стан чату ---------------------------------------------------------------
def init_chat_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []


def read_uploaded_file(file) -> str:
    """Текст із завантаженого файлу (декодування й обрізання — у rag_context)."""
    try:
        raw = file.read()
    except Exception as e:
        return f"[не вдалося прочитати файл: {e}]"
    return decode_upload(raw)


def add_uploaded_files(files):
    """Докидає файли користувача у спільний пул документів сесії.

    Пул зберігається між повідомленнями: типовий сценарій — «ось звіт» а потім
    кілька уточнювальних питань по ньому. Список і кнопка очищення — у сайдбарі,
    щоб було видно, що саме модель бачить."""
    added = []
    known = {d["title"] for d in st.session_state.uploaded_docs}
    for f in files or []:
        name = getattr(f, "name", "файл")
        if name in known:
            continue
        st.session_state.uploaded_docs.append(make_upload_doc(name, read_uploaded_file(f)))
        added.append(name)
    return added


def _chat_history_for_llm(max_turns=3):
    """Стисла історія діалогу для моделі: останні N пар «питання → відповідь».

    Без неї «а що по цьому CVE?» після попереднього питання не має сенсу.
    Відповіді агентів похідні від недовіреного OSINT, тому йдуть у тій самій
    ізоляції, що й документи (wrap_untrusted)."""
    turns, pending_q = [], None
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            pending_q = msg.get("content", "")
        elif pending_q is not None:
            answer = msg.get("summary") or ""
            if not answer and msg.get("agents"):
                answer = msg["agents"][0].get("answer", "")
            if answer:
                turns.append((pending_q, answer[:600]))
            pending_q = None
    if not turns:
        return ""
    block = ""
    for q, a in turns[-max_turns:]:
        block += f"Питання: {q}\nВідповідь ONYX: {wrap_untrusted(a)}\n\n"
    return "ПОПЕРЕДНІЙ ХІД ДІАЛОГУ (для розуміння контексту питання):\n" + block


# --- обробка одного повідомлення --------------------------------------------
def handle_analysis(query, docs, meta):
    """Звичайне питання: рада агентів + синтезатор + граф звʼязків по джерелах."""
    if not docs:
        return {"role": "assistant", "kind": INTENT_ANALYSIS, "query": query,
                "no_results": True, "meta": meta}

    history = _chat_history_for_llm()
    active_agents = [a for a in st.session_state.agents if a["enabled"]]
    agent_outputs = []
    if active_agents:
        with st.status(f"🧬 ONYX запускає {len(active_agents)} аген(тів)…", expanded=True) as status:
            for a in active_agents:
                status.write(f"{a['icon']} Агент «{a['name']}» аналізує джерела…")
                question = (history + f"Запитання: {query}") if history else query
                agent_backend = resolve_backend(a.get("backend"))
                agent_outputs.append((a, run_agent(question, docs, a["prompt"], backend=agent_backend)))
            status.update(label=f"✅ Відповіли {len(agent_outputs)} аген(тів)",
                          state="complete", expanded=False)

    summary, synth_meta = None, None
    if len(agent_outputs) > 1:
        synth = st.session_state.synth_agent
        with st.status(f"{synth['icon']} {synth['name']} зводить відповіді…", expanded=False):
            summary = run_synthesizer(query, [(a["name"], ans) for a, ans in agent_outputs],
                                      synth["prompt"], backend=resolve_backend(synth.get("backend")))
        synth_meta = {"icon": synth["icon"], "name": synth["name"]}

    return {
        "role": "assistant", "kind": INTENT_ANALYSIS, "query": query, "meta": meta,
        "docs": docs,
        "rag_doc": build_rag_document(docs), "stats": rag_stats(docs),
        "agents": [{"name": a["name"], "icon": a["icon"], "color": a["color"], "answer": ans}
                   for a, ans in agent_outputs],
        "no_agents": not active_agents,
        "summary": summary, "synth": synth_meta,
        "cti_html": build_cti_graph_html(docs),
    }


def handle_semantic_network(query, docs, meta, lang_filter):
    """«згенуй мені семантичну мережу понять» — мережа + експорти в чат."""
    # Якщо в запиті була тема — будуємо по знайдених документах; якщо ні
    # («просто побудуй мережу») — по свіжому зрізу корпусу.
    records = docs
    scope = f"за знайденими джерелами ({len(docs)})"
    if len(records) < 5:
        records = list(st.session_state.uploaded_docs) + list(fetch_events(lang_filter, limit=300))
        scope = f"за останніми {len(records)} подіями корпусу"
    art = build_semantic_network_artifacts(records, top_terms=40, min_edge=1)
    return {
        "role": "assistant", "kind": INTENT_SEMANTIC_NETWORK, "query": query, "meta": meta,
        "docs": docs, "scope": scope,
        "rag_doc": build_rag_document(docs), "stats": rag_stats(docs),
        "html": (art or {}).get("html"), "exports": (art or {}).get("exports"),
        "nodes": (art or {}).get("nodes"), "edges": (art or {}).get("edges"),
    }


def handle_hypothesis_graph(query, goal, lang_filter, starts=None):
    """«збудуй граф гіпотез [про X]» / «граф гіпотез від X до Y».

    Два РІЗНІ алгоритми, а не одна команда з опцією:
      • без стартів — односторонній обхід від цілі назад (`hypotheses`);
      • зі стартами — двофронтовий пошук назустріч (`hypotheses_from`), тобто
        те саме «двостороннє формування», що заявлене в методиці.
    Який режим відпрацював — повертаємо назовні, щоб аналітик це БАЧИВ, а не
    здогадувався."""
    rows = fetch_events(lang_filter, limit=5000)
    if not rows:
        return {"role": "assistant", "kind": INTENT_HYPOTHESIS_GRAPH, "query": query,
                "goal": goal, "found": False, "hyps": [],
                "note": "Корпус порожній — немає з чого будувати ланцюги."}

    reasoner = _get_reasoner_for_rows(rows)
    anchors = sorted((n for n, d in reasoner.G.nodes(data=True) if d.get("is_anchor")),
                     key=lambda n: reasoner.G.nodes[n].get("doc_freq", 0), reverse=True)[:12]

    if not goal:
        return {"role": "assistant", "kind": INTENT_HYPOTHESIS_GRAPH, "query": query,
                "goal": None, "found": False, "hyps": [], "anchors": anchors,
                "note": "Не вказано ціль. Напишіть, наприклад: "
                        "«збудуй граф гіпотез про ransomware», "
                        "або двофронтово — «граф гіпотез від фішингу до ransomware»."}

    goal_node = reasoner.resolve_node(goal)
    if goal_node is None:
        return {"role": "assistant", "kind": INTENT_HYPOTHESIS_GRAPH, "query": query,
                "goal": goal, "found": False, "hyps": [], "anchors": anchors,
                "note": f"Концепт «{goal}» не знайдено у графі корпусу."}

    # Старти резолвимо ДО пошуку, щоб чесно сказати, які з них не знайшлись.
    requested = [s for s in (starts or []) if str(s).strip()]
    resolved_starts, unknown_starts = [], []
    for raw in requested:
        node = reasoner.resolve_node(raw)
        if node is None:
            unknown_starts.append(raw)
        elif node != goal_node and node not in resolved_starts:
            resolved_starts.append(node)

    if resolved_starts:
        mode = "bidirectional"
        with st.spinner("Двофронтовий обхід: від свідчень уперед і від цілі назад…"):
            hyps = reasoner.hypotheses_from(resolved_starts, goal_node,
                                            k=5, weight_mode="npmi")
    else:
        mode = "unidirectional"
        with st.spinner("Обхід мережі від цілі назад…"):
            hyps = reasoner.hypotheses(goal_node, k=5, weight_mode="npmi")

    label_map = _get_mitre_label_map()
    narrative = None
    if hyps:
        llm_block = format_hypotheses_for_llm(hyps, goal_node, label_map=label_map)
        # Режим кажемо моделі явно: у двофронтовому старти ЗАДАВ аналітик, і
        # трактувати їх як «знайдені системою» було б перебільшенням.
        if mode == "bidirectional":
            header = (f"Ціль: {goal_node}\n"
                      f"Стартові концепти (задані аналітиком): "
                      f"{', '.join(resolved_starts)}\n"
                      "Режим: двофронтовий обхід назустріч.\n")
        else:
            header = (f"Ціль: {goal_node}\n"
                      "Режим: односторонній обхід від цілі назад; "
                      "старти вибрані алгоритмом.\n")
        with st.spinner("ONYX Predict інтерпретує ланцюги…"):
            narrative = _llm_generate(HYPOTHESIS_PROMPT,
                                      f"{header}\n{llm_block}\n\nАналітичний висновок:",
                                      temperature=0.2, num_predict=650)

    # Ланцюги готуємо ЗАРАЗ (разом із джерелами з Mongo), щоб історія чату
    # перемальовувалась без походів у базу.
    all_src = sorted({s for h in hyps for s in h["source_ids"]})
    src_texts = fetch_original_texts(all_src) if all_src else {}
    chains = []
    for h in hyps:
        chains.append({
            "rank": h["rank"], "is_anchor": bool(h.get("is_anchor")),
            "nodes": annotate_chain_for_display(h["chain"], label_map),
            "hops": h["hops"], "total_weight": h["total_weight"], "score": h["score"],
            "sources": [(sid, (src_texts.get(sid) or "").strip()[:280])
                        for sid in h["source_ids"][:8]],
        })

    return {
        "role": "assistant", "kind": INTENT_HYPOTHESIS_GRAPH, "query": query,
        "goal": goal_node, "found": True, "hyps": hyps, "chains": chains,
        "narrative": narrative, "anchors": anchors,
        "mode": mode, "starts": resolved_starts, "unknown_starts": unknown_starts,
        # Одна «колбаса» замість N карток із власними випадайками джерел.
        "chains_doc": build_hypothesis_document(chains, goal_node,
                                                start_label=VIRTUAL_START_NODE),
        "stats": hypothesis_stats(chains),
        "html": build_hypothesis_graph_html(hyps, goal_node, label_map),
    }


def handle_cti_graph(query, docs, meta, lang_filter):
    """«покажи граф звʼязків» — пости ↔ верифіковані CVE / MITRE."""
    records = docs if len(docs) >= 3 else list(fetch_events(lang_filter, limit=60))
    return {
        "role": "assistant", "kind": INTENT_CTI_GRAPH, "query": query, "meta": meta,
        "docs": docs, "count": len(records),
        "rag_doc": build_rag_document(docs), "stats": rag_stats(docs),
        "html": build_cti_graph_html(records[:40]),
    }


def process_chat_message(text, lang_filter):
    """Один прохід: намір -> пошук контексту -> обробник -> повідомлення в історію."""
    intent = detect_intent(text)
    kind = intent["kind"]

    # Файли користувача завжди попереду документів корпусу: якщо людина щойно
    # докинула звіт, питання майже напевно про нього.
    pinned = list(st.session_state.uploaded_docs)

    if kind == INTENT_HYPOTHESIS_GRAPH:
        return handle_hypothesis_graph(text, intent["goal"], lang_filter,
                                       starts=intent.get("starts"))

    # Для графових команд шукаємо по ТЕМІ, а не по всьому тексту: інакше
    # «згенуй семантичну мережу понять» шукало б документи зі словами «мережа»
    # й «поняття» і будувало мережу по них замість усього корпусу.
    search_text = intent["goal"] if kind != INTENT_ANALYSIS else text
    found, meta = [], {}
    if search_text:
        with st.spinner("🔎 ONYX шукає релевантні джерела…"):
            found, meta = retrieve_context(search_text, lang_filter)
    docs = pinned + found

    if kind == INTENT_SEMANTIC_NETWORK:
        return handle_semantic_network(text, docs, meta, lang_filter)
    if kind == INTENT_CTI_GRAPH:
        return handle_cti_graph(text, docs, meta, lang_filter)
    return handle_analysis(text, docs, meta)


# --- рендер повідомлень ------------------------------------------------------
def render_rag_sausage(msg, key):
    """«Колбаса»: усі документи, що пішли в модель, одним читабельним текстом."""
    rag_doc = msg.get("rag_doc")
    if not rag_doc:
        return
    s = msg.get("stats") or {}
    label = (f"📦 Контекст RAG — {s.get('documents', 0)} док. · "
             f"{s.get('characters', 0):,} симв. · {s.get('sources', 0)} джерел"
             .replace(",", " "))
    if s.get("cve") or s.get("ttp"):
        label += f" · 🔴 {s.get('cve', 0)} CVE · 🟠 {s.get('ttp', 0)} MITRE"
    with st.expander(label):
        st.caption(
            "Це рівно ті документи, що пішли в модель як RAG — зібрані в один "
            "документ для читання. У саму модель вони йдуть ізольованими "
            "делімітерами (захист від prompt injection у тексті допису)."
        )
        meta = msg.get("meta") or {}
        if meta.get("strategies"):
            st.caption("🔎 Стратегії пошуку: " + ", ".join(meta["strategies"]) +
                       (f" · англ. варіант запиту: `{meta['translated']}`"
                        if meta.get("translated") else ""))
        st.markdown(rag_doc)
        st.download_button("⬇️ Зберегти контекст (.md)", rag_doc,
                           file_name="onyx_rag_context.md", mime="text/markdown",
                           key=f"dl_rag_{key}")


def render_analysis_message(msg, key):
    if msg.get("no_results"):
        st.info("ℹ️ Не знайшлося релевантних джерел під цей запит. Спробуйте "
                "інші слова або зніміть фільтр мов у лівій панелі.")
        return
    render_rag_sausage(msg, key)

    if msg.get("no_agents"):
        st.warning("⚠️ Немає активних агентів — увімкніть хоча б одного у «🤖 Агенти».")

    for a in msg.get("agents", []):
        with st.container(border=True):
            st.markdown(
                f'<div class="onyx-result-bar" style="--c:{a["color"]}"></div>'
                f'<div class="onyx-result-head">{a["icon"]} {html.escape(a["name"])}'
                f'<span class="onyx-result-tag">AGENT</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(a["answer"])

    if msg.get("summary") and msg.get("synth"):
        synth = msg["synth"]
        with st.container(border=True):
            st.markdown(f'<div class="onyx-summary-head">{synth["icon"]} '
                        f'{html.escape(synth["name"])} — зведений висновок</div>',
                        unsafe_allow_html=True)
            st.markdown(msg["summary"])

    if msg.get("cti_html"):
        with st.expander("🕸️ Граф звʼязків за джерелами відповіді"):
            components.html(msg["cti_html"], height=470, scrolling=False)


def render_semantic_message(msg, key):
    render_rag_sausage(msg, key)
    if not msg.get("html"):
        st.info("ℹ️ Недостатньо термінів для побудови семантичної мережі "
                "(замало документів або вони надто короткі).")
        return
    st.markdown(f"**🧠 Семантична мережа понять** — {msg.get('scope', '')} · "
                f"{msg.get('nodes')} вузлів, {msg.get('edges')} звʼязків")
    st.caption("Вузли — терміни-поняття та верифіковані сутності (🔴 CVE, 🟠 MITRE); "
               "ребра — співзустрічальність у межах документа (методологія Д.В. Ланде).")
    components.html(msg["html"], height=620, scrolling=False)
    render_graph_export_buttons(msg.get("exports"), key)


def render_hypothesis_message(msg, key):
    if not msg.get("found"):
        st.warning("⚠️ " + msg.get("note", "Не вдалося побудувати гіпотези."))
        if msg.get("anchors"):
            st.caption("🔎 Верифіковані якірні концепти у корпусі: " +
                       ", ".join(f"`{a}`" for a in msg["anchors"]))
        return

    chains = msg.get("chains") or []
    if not chains:
        st.info(f"ℹ️ До цілі `{msg['goal']}` не знайдено підтримувальних ланцюгів у корпусі.")
        return

    # Ланцюги — однією «колбасою», як і контекст RAG. Раніше тут було N карток,
    # і в кожній ще один expander «📚 Джерела»: щоб прочитати одну думку,
    # доводилось клікати крізь десяток випадайок.
    s = msg.get("stats") or {}
    label = (f"🧩 Ланцюги міркування — {s.get('hypotheses', len(chains))} гіпотез · "
             f"{s.get('sources', 0)} джерел")
    if s.get("anchors"):
        label += f" · 🔗 {s['anchors']} на верифікованих якорях"
    with st.expander(label, expanded=True):
        # Ціль тут НЕ дублюємо — вона у шапці самого документа нижче.
        # Режим обходу показуємо явно: односторонній і двофронтовий — це різні
        # алгоритми з різною якістю, і аналітик має знати, що саме відпрацювало.
        if msg.get("mode") == "bidirectional":
            st.caption("↔️ **Двофронтовий обхід**: уперед від "
                       + ", ".join(f"`{n}`" for n in msg.get("starts", []))
                       + f" і назад від `{msg['goal']}`, зустріч посередині.")
        else:
            st.caption(f"⬅️ **Односторонній обхід** від `{msg['goal']}` назад; "
                       "старти вибрав алгоритм. Щоб задати їх самому — "
                       "«граф гіпотез **від** X **до** " + str(msg["goal"]) + "».")
        if msg.get("unknown_starts"):
            st.caption("⚠️ Не знайдено у графі й пропущено: "
                       + ", ".join(f"`{n}`" for n in msg["unknown_starts"]))
        st.caption("Вага звʼязків: NPMI (нормалізована поточкова взаємна "
                   "інформація — гасить вузли-хаби).")
        st.markdown(msg.get("chains_doc") or "")
        if msg.get("chains_doc"):
            st.download_button("⬇️ Зберегти ланцюги (.md)", msg["chains_doc"],
                               file_name="onyx_hypotheses.md", mime="text/markdown",
                               key=f"dl_hyp_{key}")

    if msg.get("html"):
        st.markdown("**🕸️ Граф гіпотез** — єдиний початок «Теперішній стан» → ланцюги → ціль")
        st.caption("Сірий трикутник — фіктивна точка відліку (у корпусі її немає, "
                   "вона лише зводить усі ланцюги до одного початку); фіолетова зірка — "
                   "ціль; золоті вузли — верифіковані якорі MITRE/NVD.")
        components.html(msg["html"], height=540, scrolling=False)

    if msg.get("narrative"):
        with st.container(border=True):
            st.markdown("#### 🧠 ONYX Predict — інтерпретація ланцюгів")
            st.markdown(msg["narrative"])


def render_cti_graph_message(msg, key):
    render_rag_sausage(msg, key)
    if not msg.get("html"):
        st.info("ℹ️ У цих джерелах немає верифікованих CVE/MITRE — граф звʼязків порожній.")
        return
    st.markdown(f"**🕸️ Граф звʼязків** — пости ↔ верифіковані сутності "
                f"({msg.get('count', 0)} подій)")
    components.html(msg["html"], height=470, scrolling=False)


_MESSAGE_RENDERERS = {
    INTENT_ANALYSIS: render_analysis_message,
    INTENT_SEMANTIC_NETWORK: render_semantic_message,
    INTENT_HYPOTHESIS_GRAPH: render_hypothesis_message,
    INTENT_CTI_GRAPH: render_cti_graph_message,
}


def render_chat_message(msg, index):
    """Один бабл діалогу. Ключі віджетів прив'язані до індексу повідомлення —
    інакше Streamlit сваритиметься на дублі при перемальовуванні історії."""
    key = f"m{index}"
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg.get("content", ""))
            if msg.get("files"):
                st.caption("📎 " + ", ".join(msg["files"]))
        return

    with st.chat_message("assistant", avatar="🔷"):
        if msg.get("error"):
            st.error(f"❌ {msg['error']}")
            return
        _MESSAGE_RENDERERS.get(msg.get("kind"), render_analysis_message)(msg, key)


def render_chat_mode(lang_filter):
    """Режим «чат»: один вхід, історія діалогу, файли, графи прямо у стрічці."""
    if not st.session_state.chat_messages:
        st.markdown(
            '<div class="onyx-chat-hint">'
            '<b>Напишіть запит звичайною мовою.</b> ONYX сам зрозуміє, що робити:'
            '<ul>'
            '<li><code>розкажи про останні вразливості</code> — відповідь ради агентів '
            'за знайденими джерелами;</li>'
            '<li><code>згенуй мені семантичну мережу понять</code> — мережа + експорт '
            'GraphML/GEXF для Gephi;</li>'
            '<li><code>збудуй граф гіпотез про ransomware</code> — ланцюги міркування '
            'від «Теперішнього стану» до цілі;</li>'
            '<li><code>покажи граф звʼязків</code> — пости ↔ верифіковані CVE/MITRE.</li>'
            '</ul>'
            'Файли можна докинути скріпкою — вони стають частиною контексту.'
            '</div>', unsafe_allow_html=True,
        )

    for i, msg in enumerate(st.session_state.chat_messages):
        render_chat_message(msg, i)

    user_input = st.chat_input(
        "Запит до ONYX… (можна докинути файл скріпкою)",
        accept_file="multiple",
        file_type=["txt", "md", "log", "json", "csv", "yaml", "yml", "html"],
    )
    if not user_input:
        return

    # st.chat_input з accept_file повертає об'єкт із .text і .files (Streamlit
    # ≥1.43); підстраховуємось на випадок звичайного рядка.
    text = getattr(user_input, "text", user_input) or ""
    files = list(getattr(user_input, "files", []) or [])
    added = add_uploaded_files(files)
    if not text.strip() and not added:
        return

    st.session_state.chat_messages.append(
        {"role": "user", "content": text.strip(), "files": added})
    render_chat_message(st.session_state.chat_messages[-1],
                        len(st.session_state.chat_messages) - 1)

    if not text.strip():
        st.session_state.chat_messages.append({
            "role": "assistant", "kind": INTENT_ANALYSIS,
            "agents": [{"name": "ONYX", "icon": "📎", "color": "#7c3aed",
                        "answer": "Файл(и) додано до контексту: **" + "**, **".join(added) +
                                  "**. Тепер можна ставити запитання по них."}],
        })
        st.rerun()

    with st.chat_message("assistant", avatar="🔷"):
        try:
            reply = process_chat_message(text.strip(), lang_filter)
        except Exception as e:
            reply = {"role": "assistant", "kind": INTENT_ANALYSIS, "error": str(e)}
    st.session_state.chat_messages.append(reply)
    st.rerun()


# --- ВІЗУАЛЬНИЙ ІНТЕРФЕЙС STREAMLIT ---

# Ініціалізація стану ШІ-агентів ONYX
init_agent_state()
init_chat_state()

# --- БОКОВА ПАНЕЛЬ: перемикач режимів + налаштування ---
# Два режими замість одного перевантаженого екрана:
#   💬 Чат     — весь аналітичний контур (агенти, графи, гіпотези) через один вхід;
#   📰 Стрічка — сира стрічка подій і звичайний повнотекстовий пошук.
st.sidebar.markdown('<div class="onyx-side-brand">&#9670; ONYX</div>', unsafe_allow_html=True)
app_mode = st.sidebar.radio(
    "Режим роботи",
    options=["💬 Чат з ONYX", "📰 Стрічка подій", "🎯 Сценарії"],
    index=0,
    help="Чат — аналітика й графи одним запитом. Стрічка — перегляд корпусу й "
         "пошук. Сценарії — орієнтований граф переходів за описом інфраструктури.",
)
IS_CHAT = app_mode.startswith("💬")
IS_SCENARIOS = app_mode.startswith("🎯")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Налаштування")
onyx_theme = st.sidebar.radio(
    "🎨 Тема інтерфейсу",
    options=["🌑 ONYX Amethyst", "☀️ Класична світла"],
    index=0,
    help="Фірмова темна аметистова тема або класична світла (для друку/проєктора).",
)
theme_mode = "amethyst" if onyx_theme.startswith("🌑") else "light"
# «unknown» — повноцінна опція, а не виняток: детектор мови проставив її 952 з
# 1495 документів. Поки її не було в переліку, дві третини корпусу були
# невидимі і в стрічці, і в контексті агентів. Порожній вибір = усі мови.
LANGUAGE_OPTIONS = ["uk", "ru", "en", "unknown"]
lang_filter = st.sidebar.multiselect(
    "Мова оригіналу", LANGUAGE_OPTIONS, default=LANGUAGE_OPTIONS,
    help="Порожній вибір = без обмежень. «unknown» — документи, яким детектор "
         "не визначив мову (їх у корпусі більшість).",
)

show_semantic = True
if IS_CHAT:
    st.sidebar.markdown("---")
    docs_pool = st.session_state.uploaded_docs
    st.sidebar.markdown(f"**📎 Файли в контексті: {len(docs_pool)}**")
    if docs_pool:
        for d in docs_pool:
            st.sidebar.caption(f"• {d['title']} ({len(d['content'])} симв.)")
        if st.sidebar.button("🗑 Прибрати файли", use_container_width=True):
            st.session_state.uploaded_docs = []
            st.rerun()
    else:
        st.sidebar.caption("Докиньте файл скріпкою у полі вводу — він стане частиною контексту.")
    if st.session_state.chat_messages and st.sidebar.button("🧹 Очистити діалог",
                                                            use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
else:
    show_semantic = st.sidebar.checkbox(
        "🧠 Семантична мережа понять",
        value=False,
        help="Горизонтальна семантична мережа співзустрічальності термінів "
             "(методологія Д.В. Ланде). У режимі чату — запитом «згенуй семантичну мережу понять».",
    )
    if st.sidebar.button("🔄 Оновити дані", use_container_width=True):
        st.rerun()

# Інʼєкція обраної теми ONYX (після вибору у сайдбарі)
inject_onyx_theme(theme_mode)

# =========================================================================
#  РЕЖИМ «СТРІЧКА» — сирий корпус подій і звичайний повнотекстовий пошук
# =========================================================================
# Свідомо БЕЗ аналітики: агенти, гіпотези й мережі понять переїхали в чат.
# Тут людина дивиться, що взагалі є в корпусі, і шукає по ньому словами.
def render_feed_mode(lang_filter, theme_mode, show_semantic):
    st.markdown('<div class="onyx-section-title">📰 Стрічка верифікованих подій</div>',
                unsafe_allow_html=True)
    st.caption("Повний корпус OSINT-подій після збагачення. Пошук морфологічний: "
               "«вразливість» знайде «вразливості», «vulnerability» — «vulnerabilities».")

    search_query = st.text_input(
        "🔍 Повнотекстовий пошук (наприклад: Ubuntu, Cisco, вразливість)", "",
        help="Спецсимволи (@ ( ) | - ! ~ \" / ^ $ = * ? та лапки) екрануються й "
             "трактуються буквально — напр. «-» більше НЕ оператор NOT.",
    )

    try:
        rows = fetch_events(lang_filter, search_query, limit=5000)
    except Exception as e:
        st.error(f"❌ Помилка підключення до Manticore Search: {e}")
        rows = []

    if not rows:
        st.info("ℹ️ Поки немає даних для відображення або змініть налаштування "
                "фільтрів чи пошукового запиту.")
        return

    df = pd.DataFrame(rows)

    col1, col2, col3 = st.columns(3)
    col1.metric("Всього знайдено алертів", len(rows))
    col2.metric("Унікальних джерел", df['channel_name'].nunique())
    col3.metric("Останнє оновлення", datetime.now().strftime("%H:%M:%S"))

    st.markdown("---")
    st.markdown("### 📊 Аналітичний огляд загрози")
    # Колір тексту графіків та прозорий фон — щоб діаграми зливалися з обраною темою
    chart_font = "#1b1b2b" if theme_mode == "light" else "#e7e7f2"
    chart_grid = "rgba(124,58,237,0.18)"

    def _style_fig(fig):
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=chart_font, legend_font_color=chart_font,
            title_font_color=chart_font,
        )
        fig.update_xaxes(gridcolor=chart_grid, zerolinecolor=chart_grid)
        fig.update_yaxes(gridcolor=chart_grid, zerolinecolor=chart_grid)
        return fig

    graph_col1, graph_col2 = st.columns(2)
    with graph_col1:
        lang_counts = df['language'].value_counts().reset_index()
        lang_counts.columns = ['Мова', 'Кількість']
        fig_lang = px.pie(lang_counts, values='Кількість', names='Мова',
                          title='🌐 Розподіл джерел за мовами', hole=0.4,
                          color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(_style_fig(fig_lang), use_container_width=True, theme=None)

    with graph_col2:
        src_counts = df['source_type'].value_counts().reset_index()
        src_counts.columns = ['Джерело', 'Кількість']
        fig_src = px.bar(src_counts, x='Джерело', y='Кількість',
                         title='🛰️ Розподіл подій за типом джерела', color='Джерело',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(_style_fig(fig_src), use_container_width=True, theme=None)

    if search_query:
        st.markdown("---")
        st.markdown(f"### 🕸️ Складений граф кореляції для запиту: *\"{search_query}\"*")
        render_filtered_cti_graph(df)

    if show_semantic:
        st.markdown("---")
        st.markdown("### 🧠 Семантична мережа понять (співзустрічальність термінів)")
        st.caption(
            "Вузли — значущі терміни-поняття та верифіковані сутності (🔴 CVE, 🟠 MITRE ATT&CK); "
            "ребра — семантичні асоціації за спільною появою в межах документа, товщина ребра — "
            "частота співзустрічальності. Реалізація горизонтальних семантичних мереж (Д.В. Ланде)."
        )
        if not _build_semantic_network(df.to_dict("records"), "feed"):
            st.caption("ℹ️ Недостатньо даних для побудови семантичної мережі.")

    st.markdown("---")
    st.markdown("### 📰 Події")

    source_labels = {
        "telegram": "📡 TELEGRAM",
        "youtube": "▶️ YOUTUBE",
        "website": "🌐 WEBSITE",
        "rss": "📰 RSS",
    }

    # Оригінальні тексти для видимих карток тягнемо з Mongo одним запитом.
    feed_texts = fetch_original_texts([r.get('mongo_id') for r in rows[:50]])

    for row in rows[:50]:
        src_type = (row.get('source_type') or "").lower()
        src_badge = source_labels.get(src_type, f"📁 {src_type.upper() or 'UNKNOWN'}")
        channel = row.get('channel_name') or "—"

        with st.container():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.subheader(row['title'])
                st.caption(f"📅 Отримано: {datetime.fromtimestamp(row['collected_at'])} | "
                           f"🌐 Мова оригіналу: **{row['language'].upper()}** | "
                           f"📥 Джерело: **{channel}**")
            with c2:
                st.markdown(f"### `{src_badge}`")

            st.markdown(f"**🤖 AI Summary (EN):** {row['summary']}")

            with st.expander("📄 Переглянути оригінальний текст повідомлення (Джерело: MongoDB)"):
                original_text = (feed_texts.get(row.get('mongo_id'))
                                 or "Текст відсутній або не знайдений у MongoDB.")
                st.text_area(
                    label="Сирий текст OSINT-події",
                    value=original_text,
                    height=180,
                    disabled=True,
                    key=f"txt_{row['id']}",
                )

            # Ідентифікатори приходять з індексу (тобто з ланцюга «допис →
            # LLM → верифікація»), а йдуть у сирий HTML. Фільтр за формою +
            # екранування — у ui_html.py, разом з тестами.
            tags_html = build_verified_tags_html(
                row['verified_cve_ids'], row['verified_ttp_ids'])
            if tags_html:
                st.markdown(tags_html, unsafe_allow_html=True)

            st.markdown("---")



def render_scenarios_mode(lang_filter):
    """Сценарії на ОРІЄНТОВАНОМУ типізованому графі.

    Чим це відрізняється від «графа гіпотез». Там мережа неорієнтована, ребро
    означає співзустрічальність понять в одному документі, а стрілка на схемі —
    лише порядок читання. Тут кожен перехід має НАЗВУ, узяту з поля опису
    інфраструктури, тож ланцюг читається як сценарій, а не як список сусідів.

    Поділ джерел свідомий: СТРУКТУРУ дає аналітик (опис інфраструктури),
    ДОКАЗОВІСТЬ — зібраний корпус OSINT. Витягати переходи з новин мовною
    моделлю ми не будемо: це той самий клас помилки, що й вигаданий T1190,
    тільки замість мітки вигадувався б причинний зв'язок.
    """
    st.markdown("### 🎯 Сценарії на орієнтованому графі")
    st.caption(
        "Опис інфраструктури задає структуру переходів, корпус OSINT — їх "
        "доказовість. Формат опису — той самий, що в наборі експерименту: "
        "об'єкт із `components`, за потреби `identities`, `permission_edges`, "
        "`artifacts`, `vulnerabilities`."
    )

    left, right = st.columns([3, 2])
    with left:
        uploaded = st.file_uploader("Опис інфраструктури (JSON)", type=["json"],
                                    key="twin_model_file")
        raw = st.text_area(
            "…або вставте опис сюди",
            value=(uploaded.read().decode("utf-8", "replace") if uploaded else ""),
            height=180, key="twin_model_text",
            placeholder='{"machine": "...", "components": [...]}')
    with right:
        goal = st.text_input("Цільовий стан", value="root", key="twin_goal",
                             help="Те, чим закінчується сценарій: root, SYSTEM, "
                                  "administrator — як воно назване в описі.")
        k = st.slider("Скільки альтернатив показати", 1, 10, 5, key="twin_k")
        run = st.button("Побудувати сценарії", type="primary",
                        use_container_width=True)

    if not run:
        return
    if not (raw or "").strip():
        st.info("Додайте опис інфраструктури — з чого будувати переходи.")
        return

    try:
        machine = parse_infrastructure_model(raw)
    except ValueError as exc:
        st.error(f"Опис не прийнято: {exc}")
        return

    # Корпус для доказовості: title+summary з індексу. Повні тексти живуть у
    # Mongo (сліпий індекс), тягнути їх на весь корпус заради перевірки ребер
    # надто дорого — цього зрізу достатньо, щоб показати згадки.
    with st.spinner("Будую орієнтований граф і перебираю допустимі шляхи…"):
        docs = fetch_events(lang_filter, limit=2000)
        result = build_scenarios(machine, goal, docs, k=int(k))

    cols = st.columns(4)
    cols[0].metric("Вузлів", result["nodes"])
    cols[1].metric("Ребер", result["edges"])
    cols[2].metric("З них переходів", result["transitions"])
    cols[3].metric("Кандидатів", result.get("candidates", 0))

    if not result["found"]:
        st.warning(result["note"])
        return

    evidence_by_edge = {}
    for item in result["evidence"]:
        evidence_by_edge.setdefault(
            (item["source"], item["relation"], item["target"]), item)

    st.success(f"Знайдено сценаріїв: {len(result['scenarios'])} "
               f"(ціль — «{result['goal']}»)")

    for scenario in result["scenarios"]:
        header = (f"№{scenario['rank']} · J = {scenario['j']:.3f} · "
                  f"переходів: {scenario['hops']}")
        with st.expander(header, expanded=scenario["rank"] == 1):
            crit = " · ".join(
                f"{c['criterion']} {c['score']:.2f}" for c in scenario["criteria"])
            st.caption(f"Критерії: {crit}")
            for step in scenario["steps"]:
                mark = evidence_by_edge.get(
                    (step.get("from_id"), step["relation"], step.get("to_id")))
                st.markdown(
                    f"**{step['order']}.** `{step['from']}` "
                    f"— *{step['relation']}* → `{step['to']}`"
                    + ("  ✅ є згадка в корпусі" if mark else ""))
            if not scenario["steps"]:
                st.caption("Шлях без переходів — показувати нічого.")

    # ---------------------------------------------------------------
    #  Захисне плече: набори заходів, що розривають знайдені сценарії.
    #
    #  НАВІЩО САМЕ ТУТ. `twin_defense.rank_defenses` рахує набори control-
    #  вузлів САМЕ за `edge_path` шляхів вище — це не окремий режим, а
    #  продовження того самого екрана, бо без знайдених сценаріїв рахувати
    #  нічого. Показуємо явно й те, що зазвичай ховають: набори, які
    #  розривають усі шляхи, але не проходять бюджет/строки/персонал/
    #  regulatory (`excluded_by_constraints`); шляхи, які не розриває
    #  жоден наявний засіб (`unbreakable_path_indices`); ознаку обірваного
    #  перебору (`truncated`).
    # ---------------------------------------------------------------
    st.markdown("### 🛡️ Що цьому протиставити")
    model = result["model"]
    has_control_nodes = any(node.kind == "control" for node in model.nodes)
    if not has_control_nodes:
        st.info(
            "У цьому описі інфраструктури немає жодного вузла `control` — "
            "нема із чого будувати захист. Додайте control-вузли з ребрами "
            "`mitigates` до відповідних вразливостей, щоб побачити набори заходів."
        )
    else:
        attack_paths = [scenario["edge_path"] for scenario in result["scenarios"]]
        defense = rank_defenses(model, attack_paths)
        total_paths = len(attack_paths)

        if defense["ranked"]:
            st.success(f"Знайдено наборів заходів: {len(defense['ranked'])}")
            for idx, entry in enumerate(defense["ranked"], 1):
                labels = [model.node_by_id[cid].label or cid for cid in entry["controls"]]
                broken = total_paths - len(entry["uncovered_path_indices"])
                header = (f"№{idx} · J = {entry['j']:.3f} · заходів: {entry['size']} · "
                          f"розриває {broken} з {total_paths} шляхів")
                with st.expander(header, expanded=idx == 1):
                    st.markdown("**Заходи:** " + ", ".join(f"`{lbl}`" for lbl in labels))
                    crit = " · ".join(
                        f"{c['criterion']} {c['score']:.2f}" for c in entry["criteria"])
                    st.caption(f"Критерії: {crit}")
                    st.caption(
                        "Обмеження моделі (бюджет/строки/персонал/regulatory): "
                        + ("✅ пройдено" if entry["within_constraints"] else "❌ не пройдено"))
                    if not entry["exact"]:
                        st.warning(
                            "Точного повного розриву в межах перебору не знайдено — "
                            "це найкращий знайдений частковий (жадібний) набір."
                        )
                    if entry["uncovered_path_indices"]:
                        nums = ", ".join(str(i + 1) for i in entry["uncovered_path_indices"])
                        st.caption(f"Не розриває шлях(и) №{nums}.")
        elif defense["unevaluable_requirements"]:
            # `ranked` порожній, але є набори, що проходять УСЕ оцінюване -
            # це НЕ "нічого не проходить" (стара двійкова відповідь), а
            # "чисто прохідних немає, є прийнятні з застереженням" — див.
            # розділ нижче.
            st.info(
                "Набору, що проходить усі перевірки БЕЗ застережень, немає — "
                "але є набори, що проходять усе ОЦІНЮВАНЕ: див. розділ "
                "«Відповідність деяким вимогам підтвердити нічим» нижче."
            )
        else:
            st.warning(
                "Жоден набір заходів не проходить обмеження моделі "
                "(бюджет/строки/персонал/regulatory) — див. розділ нижче."
            )

        if defense["excluded_by_constraints"]:
            with st.expander(
                "⚠️ Розривають усі шляхи, але НЕ проходять обмеження: "
                f"{len(defense['excluded_by_constraints'])}"
            ):
                st.caption(
                    "Ці набори повністю закривають перелічені сценарії, але "
                    "порушують бюджет, строки, персонал чи regulatory-вимогу "
                    "моделі — система свідомо показує їх окремо, а не ховає."
                )
                for entry in defense["excluded_by_constraints"]:
                    labels = [model.node_by_id[cid].label or cid for cid in entry["controls"]]
                    st.markdown(
                        f"`{'`, `'.join(labels)}` — J = {entry['j']:.3f}, "
                        "обмеження не пройдено")

        # ------------------------------------------------------------
        #  Третій стан: НЕ порушник і НЕ "просто прохідний". Модель має
        #  вимогу (наприклад regulatory), якої в ній НЕМАЄ жодного вузла,
        #  здатного підтвердити (`twin_defense.unevaluable_constraints`) -
        #  видавати брак даних за доведену невідповідність системі
        #  заборонено, тож цей стан показується окремо від
        #  `excluded_by_constraints`, формулюванням "потрібне рішення
        #  аналітика", а не як помилка чи порушення.
        # ------------------------------------------------------------
        if defense["unevaluable_requirements"]:
            with st.expander(
                "❓ Відповідність деяким вимогам підтвердити нічим: "
                f"{len(defense['unevaluable_requirements'])}"
            ):
                st.caption(
                    "Ці набори повністю закривають перелічені сценарії й НЕ "
                    "порушують жодного обмеження, для якого є дані — але "
                    "модель не містить жодного вузла, здатного підтвердити "
                    "ще одну активну вимогу. Це не помилка й не порушення: "
                    "рішення про прийнятність лишається за аналітиком."
                )
                for entry in defense["unevaluable_requirements"]:
                    labels = [model.node_by_id[cid].label or cid for cid in entry["controls"]]
                    unmet = ", ".join(sorted(entry["unevaluable_constraints"]))
                    st.markdown(
                        f"`{'`, `'.join(labels)}` — J = {entry['j']:.3f}, "
                        f"відповідність вимозі «{unmet}» підтвердити нічим — "
                        "потрібне рішення аналітика")

        if defense["unbreakable_path_indices"]:
            nums = ", ".join(str(i + 1) for i in defense["unbreakable_path_indices"])
            st.error(
                f"Шлях(и) №{nums} не розриває жоден наявний засіб захисту в "
                "моделі — це реальний залишковий ризик, а не артефакт підбору набору."
            )

        if defense["truncated"]:
            st.caption(
                "⏱️ Перебір наборів заходів обірвано межею кількості кандидатів — "
                "знайдені набори лишаються коректними, але могли лишитись "
                "неперевірені комбінації більшого розміру."
            )

    if result["evidence"]:
        with st.expander(f"📎 Переходи, згадувані в корпусі: {len(result['evidence'])}"):
            for item in result["evidence"]:
                st.markdown(
                    f"`{item['source']}` — *{item['relation']}* → `{item['target']}` "
                    f"— {item['mongo_id']} (довіра {item['confidence']})")
                st.caption(item["fragment"][:280])
    else:
        st.caption("Жоден перехід не згадується в зібраному корпусі — "
                   "структура тримається лише на описі інфраструктури.")



# =========================================================================
#  ДИСПЕТЧЕР РЕЖИМІВ
# =========================================================================
render_onyx_hero(compact=IS_CHAT)

if IS_CHAT:
    # Агенти — під згорткою: вони налаштовуються раз, а екран мають займати
    # діалог і результати, а не сітка карток (претензія наукового керівника).
    n_active = sum(1 for a in st.session_state.agents if a["enabled"])
    with st.expander(f"🤖 Аналітичні агенти ONYX — активних: {n_active} / {len(st.session_state.agents)}"):
        st.caption(
            "Кожен агент — окрема грань ONYX зі своїм системним промптом. Наведіть на ⓘ, "
            "щоб побачити роль; «✏️ Редагувати» — щоб змінити промпт. "
            "До 5 агентів + автоматичний синтезатор (вмикається від двох агентів)."
        )
        render_agent_grid()
        render_synth_card()
    render_chat_mode(lang_filter)
elif IS_SCENARIOS:
    render_scenarios_mode(lang_filter)
else:
    render_feed_mode(lang_filter, theme_mode, show_semantic)
