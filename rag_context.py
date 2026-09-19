"""
«Колбаса» — N документів RAG, зібраних в ОДИН читабельний документ для чату.

НАВІЩО. Агент відповідає не з голови, а по знайдених джерелах. Користувач має
бачити, що саме полетіло в модель — але не як купу JSON-об'єктів, а як суцільний
текст, який можна прочитати очима. Це вітрина контексту: той самий набір
документів, тільки у людському вигляді.

МЕЖА ВІДПОВІДАЛЬНОСТІ (важливо). Цей модуль готує текст ДЛЯ ЕКРАНА. Контекст
для LLM будується окремо — `prompt_guard.format_untrusted_context`, з
делімітерами ізоляції проти indirect prompt injection. Не змішувати: тут ріжемо
довгі тексти заради браузера, там текст іде повністю.

Єдина безпекова турбота тут — щоб markdown із недовіреного допису не вдавав із
себе структуру нашого документа (підроблені заголовки секцій, обірвані ```
огорожі, горизонтальні лінійки). HTML знешкоджує сам Streamlit: st.markdown без
unsafe_allow_html його екранує.

Модуль чистий — без streamlit/мережі, тестується ізольовано.
"""

from __future__ import annotations

import re
from datetime import datetime

# Скільки символів ОДНОГО документа показуємо у «колбасі». Вітрина не має
# вивалювати мегабайти в браузер; у LLM при цьому йде повний текст.
MAX_DOC_CHARS = 2500

# Скільки символів беремо із ЗАВАНТАЖЕНОГО користувачем файлу. Захист від
# «перетягнув дамп на 50 МБ»: це і браузер, і вікно контексту моделі.
MAX_UPLOAD_CHARS = 60_000

# Мітки типів джерел — ті самі, що у стрічці (feed), щоб UI був однорідний.
SOURCE_LABELS = {
    "telegram": "📡 telegram",
    "youtube": "▶️ youtube",
    "website": "🌐 website",
    "rss": "📰 rss",
    "upload": "📎 файл користувача",
}

# Рядки, якими markdown задає структуру. На початку рядка їх знешкоджуємо
# нерозривним пробілом — символи лишаються видимими, а розмітка не спрацьовує.
_STRUCTURAL_PREFIX_RE = re.compile(r"^(\s*)([#>]|```|~~~|---|===|\*\*\*|___)", re.MULTILINE)


def _neutralize_markdown(text: str) -> str:
    """Знешкоджує структурний markdown на початку рядків недовіреного тексту.

    Не «чистить» і не видаляє нічого зі змісту — лише зсуває структурні
    послідовності нерозривним пробілом, щоб допис не вдавав із себе заголовок
    секції чи не відкривав кодову огорожу, яку ми не закриємо."""
    if not text:
        return ""
    return _STRUCTURAL_PREFIX_RE.sub(lambda m: m.group(1) + " " + m.group(2), str(text))


def _split_ids(raw) -> list[str]:
    """«CVE-1, CVE-2» -> ["CVE-1", "CVE-2"]; None/порожнє -> []."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _doc_text(doc: dict) -> str:
    """Оригінальний текст документа; якщо його не дотягнули з Mongo — summary."""
    return (doc.get("content") or doc.get("summary") or "").strip()


def _format_date(value) -> str:
    """collected_at приходить UNIX-таймстампом з Manticore; буває й порожнім."""
    if not value:
        return "—"
    try:
        return datetime.fromtimestamp(int(value)).strftime("%Y-%m-%d %H:%M")
    except (ValueError, OSError, TypeError, OverflowError):
        return str(value)


def decode_upload(raw, max_chars: int = MAX_UPLOAD_CHARS) -> str:
    """Байти завантаженого файлу -> текст (з обрізанням).

    errors="replace", а не "strict": файл у чужому кодуванні чи взагалі
    бінарник не повинен валити чат винятком. Бінарні формати (PDF/DOCX) ми
    свідомо не парсимо — окрема залежність заради одного сценарію не варта
    того, а підсунути в контекст кашу з байтів гірше, ніж чесно нічого не
    зрозуміти."""
    if not raw:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", errors="replace")
    return str(raw)[:max_chars]


def make_upload_doc(filename: str, text: str) -> dict:
    """Файл, який користувач докинув у чат, -> документ контексту.

    Форма навмисно та сама, що й у документа з корпусу: далі по конвеєру
    («колбаса», контекст для LLM, семантична мережа) обробка спільна — друга
    гілка для завантажень лише плодила б розбіжності."""
    return {
        "mongo_id": None,
        "title": filename or "файл",
        "channel_name": "користувач",
        "source_type": "upload",
        "language": "",
        "content": text or "",
        "summary": "",
        "verified_cve_ids": "",
        "verified_ttp_ids": "",
        "collected_at": None,
    }


def format_doc(doc: dict, index: int | None = None,
               max_doc_chars: int = MAX_DOC_CHARS) -> str:
    """Одна секція «колбаси»: шапка з метаданими + текст документа."""
    title = str(doc.get("title") or "Без заголовка").strip()
    num = f"{index}. " if index is not None else ""

    src_type = str(doc.get("source_type") or "").lower()
    src = SOURCE_LABELS.get(src_type, src_type or "—")
    channel = str(doc.get("channel_name") or "—")

    meta = [f"джерело: {src}", f"канал: {channel}", f"дата: {_format_date(doc.get('collected_at'))}"]
    lang = str(doc.get("language") or "").strip()
    if lang:
        meta.append(f"мова: {lang.upper()}")
    dist = doc.get("distance")
    if dist is not None:
        meta.append("збіг: текстовий" if not dist else f"вектор. відстань: {float(dist):.4f}")

    cves = _split_ids(doc.get("verified_cve_ids"))
    ttps = _split_ids(doc.get("verified_ttp_ids"))

    text = _doc_text(doc)
    if max_doc_chars and len(text) > max_doc_chars:
        text = text[:max_doc_chars].rstrip() + "\n\n*…текст обрізано для показу; у модель пішов повний*"
    text = _neutralize_markdown(text) or "*(текст недоступний)*"

    parts = [f"### {num}{_neutralize_markdown(title)}", "`" + " · ".join(meta) + "`"]
    if cves or ttps:
        badges = []
        if cves:
            badges.append("🔴 CVE: " + ", ".join(cves))
        if ttps:
            badges.append("🟠 MITRE: " + ", ".join(ttps))
        parts.append(" | ".join(badges))
    parts.append(text)
    return "\n\n".join(parts)


def build_rag_document(docs, max_doc_chars: int = MAX_DOC_CHARS) -> str:
    """Складає всі документи контексту в один markdown-документ.

    Порожній список -> "" (UI сам вирішує, що показати замість вітрини)."""
    if not docs:
        return ""
    sections = [format_doc(doc, index=i, max_doc_chars=max_doc_chars)
                for i, doc in enumerate(docs, start=1)]
    return "\n\n".join(sections)


def build_hypothesis_document(chains, goal: str, start_label: str | None = None) -> str:
    """Ланцюги гіпотез -> ОДИН читабельний документ (та сама «колбаса»).

    НАВІЩО. Гіпотези виводились як N окремих карток, і в кожній ще один
    expander «📚 Джерела» — на 5 гіпотез це до десятка випадайок, крізь які
    треба клікати, щоб прочитати одну думку. Тут усе одразу: ланцюг, метрики,
    джерела з прев'ю.

    `start_label` — імʼя фіктивного початку («Теперішній стан»). Передається
    ззовні, а не імпортується з hypothesis_graph: цей модуль свідомо не тягне
    networkx, щоб лишатись тестованим без важких залежностей.

    Прев'ю джерел — недовірений OSINT, тому проходять ту саму нейтралізацію
    markdown, що й тіла документів у build_rag_document."""
    chains = list(chains or [])
    if not chains:
        return ""

    parts = []
    if start_label:
        # Заголовок навмисно НЕ повторює саму мітку: вона й так стоїть на
        # початку кожного ланцюга нижче, а зайвий дубль у шапці читається як
        # ще один вузол.
        parts.append(
            f"**Ціль:** `{_neutralize_markdown(str(goal))}`. Кожен ланцюг "
            f"починається з фіктивної точки відліку — у корпусі такого вузла "
            f"немає, він лише зводить усі ланцюги до одного початку."
        )

    for chain in chains:
        rank = chain.get("rank", len(parts) + 1)
        nodes = [str(n) for n in (chain.get("nodes") or [])]
        if start_label:
            nodes = [start_label] + nodes
        arrow_chain = "  →  ".join(f"`{_neutralize_markdown(n)}`" for n in nodes) or "—"

        anchor = " · 🔗 якір (код сутності верифіковано в MITRE/NVD)" \
            if chain.get("is_anchor") else ""
        metrics = (f"кроків: {chain.get('hops', len(nodes) - 1)} · "
                   f"сумарна вага: {float(chain.get('total_weight', 0.0)):.3f} · "
                   f"сер. вага ребра: {float(chain.get('score', 0.0)):.3f}")

        section = [f"### {rank}. {arrow_chain}", "`" + metrics + "`" + anchor]

        sources = list(chain.get("sources") or [])
        if sources:
            section.append(f"**Джерела ({len(sources)}):**")
            for sid, preview in sources:
                text = _neutralize_markdown((preview or "").strip()) or "—"
                section.append(f"- `{sid}` — {text}")
        else:
            section.append("*Прямих джерел для цього ланцюга не знайдено.*")

        parts.append("\n\n".join(section))

    return "\n\n".join(parts)


def hypothesis_stats(chains) -> dict:
    """Числа для шапки «колбаси» гіпотез: скільки ланцюгів, унікальних джерел
    і скільки з ланцюгів спираються на верифікований якір."""
    chains = list(chains or [])
    sources = set()
    anchors = 0
    for chain in chains:
        if chain.get("is_anchor"):
            anchors += 1
        for sid, _ in (chain.get("sources") or []):
            sources.add(sid)
    return {"hypotheses": len(chains), "sources": len(sources), "anchors": anchors}


def rag_stats(docs) -> dict:
    """Числа для шапки «колбаси»: скільки документів, символів, джерел і
    верифікованих сутностей реально пішло в контекст."""
    docs = list(docs or [])
    cves, ttps, sources = set(), set(), set()
    chars = 0
    for doc in docs:
        chars += len(_doc_text(doc))
        cves.update(_split_ids(doc.get("verified_cve_ids")))
        ttps.update(_split_ids(doc.get("verified_ttp_ids")))
        channel = (doc.get("channel_name") or "").strip()
        if channel:
            sources.add(channel)
    return {
        "documents": len(docs),
        "characters": chars,
        "sources": len(sources),
        "cve": len(cves),
        "ttp": len(ttps),
    }
