"""
ONYX Semantic Reasoner — обхідна (traversable) підкладка для міркування.

Перетворює семантичну мережу співзустрічальності термінів із суто
візуалізаційної (див. `_build_semantic_network` в ui.py) на граф, яким МОЖНА
ходити й будувати гіпотези. Логіка Ланде: рішення будуємо «з кінця» — від
цільового результату розкриваємо назад підтримувальні ланцюги на реальному
корпусі (двонапрямний обхід, зустріч посередині).

Модуль — чистий Python (лише networkx), БЕЗ Streamlit: це навмисно, щоб його
можна було:
  * юніт-тестувати без підняття UI;
  * інкрементально смикати з shipper/ (див. `upsert_document`).

Словник вузлів навмисно збігається з наявною візуалізацією ui.py:
  * термін            → нормалізований токен у нижньому регістрі;
  * верифікований CVE → "CVE-2024-1234" (upper);
  * верифікований TTP → "MITRE T1059"  (upper, префікс "MITRE ").
Це дає єдину мову для картинки й для інференсу.

Конфіг — лише з env (хостів/ключів тут немає, лише параметри графа):
  REASONER_GRAPH_PATH   — куди зберігати/звідки вантажити граф (pickle);
  REASONER_ANCHOR_BOOST — множник ваги якірних (верифікованих) вузлів.

Контроль густоти/росту графа (проти квадратичного вибуху ребер і розпухання
pickle на always-on корпусі — граф росте вічно, cap і prune тримають його в
розумних межах):
  REASONER_MAX_TERMS_PER_DOC        — cap не-якірних термінів на один документ
                                       (0/від'ємне = без ліміту);
  REASONER_MAX_DOCS_PER_EDGE        — cap розміру docs-set на ребрі (семпл
                                       провенансу; count лишається точним);
  REASONER_PRUNE_MIN_DOC_FREQ       — periodic prune(): поріг doc_freq вузла;
  REASONER_PRUNE_MIN_EDGE_COUNT     — periodic prune(): поріг count ребра;
  REASONER_PRUNE_DROP_NEGATIVE_NPMI — periodic prune(): чи різати ребра з
                                       npmi<0 (1/0).

FOLLOW-UP (поза scope MVP, свідомо не реалізовано):
  * directed temporal-шар (ребро A→B за принципом «зʼявляється після»);
  * плагінний verifier під інші домени (зараз якорі — лише CVE/TTP);
  * нативний function-calling агент поверх цих функцій.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from collections import Counter
from itertools import combinations

import networkx as nx

# --- Конфіг (env-driven) -------------------------------------------------
REASONER_GRAPH_PATH = os.getenv("REASONER_GRAPH_PATH", "onyx_reasoner_graph.pkl")
# Верифіковані CVE/TTP — «сильні» якірні вузли: їхня ефективна вага в обході
# множиться на цей коефіцієнт (щоб ланцюги охочіше проходили через факти, а не
# через випадкові часті слова).
ANCHOR_BOOST = float(os.getenv("REASONER_ANCHOR_BOOST", "2.0"))

_EPS = 1e-9


def _int_env(name: str, default: int) -> int:
    """Читає ціле з env СВІЖИМ на кожен виклик (а не раз при імпорті) — щоб
    caps/prune-пороги можна було міняти "на льоту" (шippер, тести) без
    перезавантаження модуля."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")

# --- Витяг термінів (сумісний з ui.py `_extract_terms`) ------------------
# Копія стоп-слів і логіки навмисна: reasoner має бути автономним (тягнеться з
# shipper без Streamlit), а ui.py імпортує streamlit на рівні модуля, тож
# імпортувати звідти не можна. Ціна — дубль словника; вигода — розвʼязаність.
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
    "within", "without", "would", "your",
    # RU / UK
    "после", "через", "которые", "более", "может", "можно", "если", "чтобы",
    "этого", "этом", "были", "было", "есть", "тому", "після", "щодо",
    "який", "яка", "яке", "які", "також", "більше", "менше",
    "коли", "цього", "цьому", "проти", "своїх", "своєї", "дуже", "лише",
    # технічний шум
    "https", "http", "www", "com", "net", "org", "amp", "quot", "mitre", "cve",
    "року", "users", "user", "channel", "telegram", "post", "report",
    # загальні дієслова/зв'язки
    "various", "discusses", "discuss", "including", "include", "includes",
    "provided", "provides", "provide", "appears", "appear", "article",
    "online", "gain", "gains", "related", "regarding", "involving", "mentions",
    "mentioned", "claims", "claimed", "reportedly", "allegedly", "potential",
    "potentially", "several", "recent", "recently", "highlights", "describes",
    "according", "based",
    # generic-шум (спостережені порушники: found/files рвуть ланцюги гіпотез
    # через побутову лексику, не CTI-концепти — див. приклад "MITRE T1494 →
    # found → ransomware" з поста, ніяк не повʼязаного з ransomware) + весь
    # клас: медіа/файли, час, побутові іменники/дієслова-звʼязки, оцінки.
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

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁёЇїІіЄєҐґ][A-Za-zА-Яа-яЁёЇїІіЄєҐґ0-9\-]{3,}")


def extract_terms(text: str) -> list[str]:
    """Витягує значущі терміни-кандидати з тексту (та сама логіка, що в ui.py).

    Токени довжиною >= 4, без службових слів, чисел і CVE-ідентифікаторів
    (верифіковані CVE/TTP додаються окремо як якірні вузли)."""
    if not text:
        return []
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip("-")
        if len(tok) < 4:
            continue
        if tok in SEMANTIC_STOPWORDS:
            continue
        if tok.startswith("cve-"):
            continue
        if tok.replace("-", "").isdigit():
            continue
        out.append(tok)
    return out


def _split_ids(raw) -> list[str]:
    """Розбиває 'CVE-1, CVE-2' у список (сумісно з ui.py `_split_ids`)."""
    if not isinstance(raw, str):
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


_TECHNIQUE_RE = re.compile(r"t\d{4}(\.\d{3})?$")


def concept_candidates(raw) -> list[str]:
    """Ввід користувача → кандидати на ідентифікатор вузла, у порядку спадання
    пріоритету.

    Вузли CVE/MITRE живуть у графі в канонічному вигляді (`CVE-2024-1234`,
    `MITRE T1566`), а звичайні терміни — у нижньому регістрі. Користувач у чаті
    пише як завгодно, тож перед обходом ввід треба привести до словника графа.
    Функція чиста — без графа, тому тестується окремо."""
    text = str(raw or "").strip()
    if not text:
        return []
    low = text.lower()
    out: list[str] = []
    if low.startswith("cve-"):
        out.append(text.upper())
    elif low.startswith("mitre "):
        out.append("MITRE " + text[len("mitre "):].strip().upper())
    elif _TECHNIQUE_RE.fullmatch(low):
        out.append("MITRE " + text.upper())
    out.append(text)
    out.append(low)
    seen: set[str] = set()
    return [c for c in out if c and not (c in seen or seen.add(c))]


def _node_kind(node: str) -> str:
    """Класифікує вузол за його ідентифікатором."""
    if node.startswith("CVE-"):
        return "cve"
    if node.startswith("MITRE "):
        return "ttp"
    return "term"


def _doc_concepts_and_freq(
    record: dict, text_fields: tuple[str, ...] = ("title", "summary")
) -> tuple[set[str], dict[str, int]]:
    """Як `doc_concepts`, але додатково повертає частоту появи кожного терміна
    в ТЕКСТІ дока (для ранжування при cap не-якірних термінів — див.
    `_cap_terms`). Верифіковані CVE/TTP у freq не потрапляють (вони не
    з тексту, а з окремих полів — і завжди якірні, cap їх не стосується)."""
    text = " ".join(str(record.get(f, "") or "") for f in text_fields)
    term_freq = Counter(extract_terms(text))
    concepts: set[str] = set(term_freq)
    for cve in _split_ids(record.get("verified_cve_ids")):
        concepts.add(cve.upper())
    for ttp in _split_ids(record.get("verified_ttp_ids")):
        concepts.add(f"MITRE {ttp.upper()}")
    return concepts, dict(term_freq)


def doc_concepts(record: dict, text_fields: tuple[str, ...] = ("title", "summary")) -> set[str]:
    """Перетворює один запис корпусу на множину концептів-вузлів.

    record — словник у форматі Manticore `events_enriched` (title, summary,
    verified_cve_ids, verified_ttp_ids). Верифіковані сутності нормалізуються в
    той самий вигляд, що й у візуалізації."""
    concepts, _freq = _doc_concepts_and_freq(record, text_fields)
    return concepts


def _cap_terms(
    concepts: set[str], anchors: set[str], term_freq: dict[str, int] | None, cap: int
) -> set[str]:
    """Обрізає НЕ-якірні концепти дока до `cap` штук — top-N за частотою
    появи в тексті дока (якщо `term_freq` відоме), інакше детерміновано
    (сортування за назвою). Якірні концепти (CVE/TTP) НІКОЛИ не ріжуться —
    завжди лишаються понад cap.

    cap<=0 трактується як «без ліміту» (стара поведінка)."""
    if cap <= 0:
        return concepts
    anchor_terms = {c for c in concepts if c in anchors or _node_kind(c) in ("cve", "ttp")}
    non_anchor = concepts - anchor_terms
    if len(non_anchor) <= cap:
        return concepts
    if term_freq:
        ranked = sorted(non_anchor, key=lambda t: (-term_freq.get(t, 0), t))
    else:
        ranked = sorted(non_anchor)
    return anchor_terms | set(ranked[:cap])


# =========================================================================
#  SemanticReasoner
# =========================================================================
class SemanticReasoner:
    """Персистентний зважений граф співзустрічальності + API обходу/інференсу.

    Внутрішньо — неорієнтований `nx.Graph`:
      node attrs: kind, doc_freq, is_anchor, boost
      edge attrs: count (сирий лічильник співпоявин), docs (set mongo_id)
      graph attrs: n_docs (усього документів), seen_docs (set — захист від
                   подвійного інкременту при upsert).
    """

    def __init__(self, graph: nx.Graph | None = None):
        if graph is None:
            graph = nx.Graph()
        self.G = graph
        self.G.graph.setdefault("n_docs", 0)
        self.G.graph.setdefault("seen_docs", set())

    # --- Побудова / інкремент ------------------------------------------
    def upsert_document(
        self,
        doc_id,
        record: dict | None = None,
        concepts: set[str] | None = None,
        anchors: set[str] | None = None,
        text_fields: tuple[str, ...] = ("title", "summary"),
    ) -> bool:
        """Інкрементально додає ОДИН документ у граф.

        Можна передати або готовий `record` (буде розібраний `doc_concepts`),
        або вже готову множину `concepts` (+ окремо `anchors` — які з них
        якірні верифіковані сутності).

        Ідемпотентний за `doc_id`: повторний виклик з тим самим id ігнорується
        (щоб лічильники n_docs/count не дублювались). Це і є гачок під shipper —
        після запису в Manticore можна смикнути upsert одним документом.

        Контроль густоти (проти квадратичного вибуху ребер на довгих доках):
        не-якірні концепти обрізаються до `REASONER_MAX_TERMS_PER_DOC`
        (top-N за частотою в тексті — доступно лише коли переданий `record`;
        для готового `concepts` без тексту ранжувати нічим — cap застосовується
        детерміновано за сортуванням). Якірні (CVE/TTP) — завжди понад cap.

        Повертає True, якщо документ реально додано."""
        doc_key = str(doc_id) if doc_id is not None else None
        seen = self.G.graph["seen_docs"]
        if doc_key is not None and doc_key in seen:
            return False

        term_freq: dict[str, int] | None = None
        if concepts is None:
            if record is None:
                raise ValueError("upsert_document: потрібен record або concepts")
            concepts, term_freq = _doc_concepts_and_freq(record, text_fields)
            anchors = {c for c in concepts if _node_kind(c) in ("cve", "ttp")}
        if anchors is None:
            anchors = {c for c in concepts if _node_kind(c) in ("cve", "ttp")}

        concepts = {c for c in concepts if c}
        cap = _int_env("REASONER_MAX_TERMS_PER_DOC", 40)
        concepts = _cap_terms(concepts, anchors, term_freq, cap)
        if not concepts:
            # Порожній документ теж рахуємо як бачений, але без ребер —
            # щоб n_docs відповідав реальному розміру корпусу.
            if doc_key is not None:
                seen.add(doc_key)
            self.G.graph["n_docs"] += 1
            return False

        # 1. Вузли: оновлюємо doc_freq та якірність.
        for c in concepts:
            if self.G.has_node(c):
                self.G.nodes[c]["doc_freq"] += 1
            else:
                is_anchor = c in anchors or _node_kind(c) in ("cve", "ttp")
                self.G.add_node(
                    c,
                    kind=_node_kind(c),
                    doc_freq=1,
                    is_anchor=is_anchor,
                    boost=ANCHOR_BOOST if is_anchor else 1.0,
                )

        # 2. Ребра: усі пари в межах документа (+ атрибуція джерела).
        # docs-set — лише СЕМПЛ провенансу, обмежений REASONER_MAX_DOCS_PER_EDGE
        # (0/від'ємне = без ліміту): без цього він росте безмежно на always-on
        # корпусі. count — джерело правди для npmi/ваг, інкрементується ЗАВЖДИ,
        # незалежно від того, чи потрапив doc_key у (обрізаний) docs-set.
        docs_cap = _int_env("REASONER_MAX_DOCS_PER_EDGE", 200)
        for a, b in combinations(sorted(concepts), 2):
            if self.G.has_edge(a, b):
                data = self.G[a][b]
                data["count"] += 1
                if doc_key is not None and (docs_cap <= 0 or len(data["docs"]) < docs_cap):
                    data["docs"].add(doc_key)
            else:
                self.G.add_edge(a, b, count=1, docs=({doc_key} if doc_key else set()))

        if doc_key is not None:
            seen.add(doc_key)
        self.G.graph["n_docs"] += 1
        return True

    def build_from_records(self, records) -> "SemanticReasoner":
        """Будує граф з корпусу (ітерований набір записів Manticore-формату)."""
        for i, rec in enumerate(records):
            # Якщо в записі немає id/mongo_id — синтезуємо стабільний ключ за
            # позицією, щоб дедуп upsert не схлопнув різні документи в один.
            doc_id = rec.get("mongo_id") or rec.get("id") or f"_pos_{i}"
            self.upsert_document(doc_id, record=rec)
        return self

    # --- Контроль густоти (періодичний sweep) ---------------------------
    def prune(
        self,
        min_doc_freq: int | None = None,
        min_edge_count: int | None = None,
        drop_negative_npmi: bool | None = None,
    ) -> dict:
        """Періодична чистка графа від шуму (ВИКЛИКАТИ ВРУЧНУ/за розкладом з
        shipper — НЕ на кожен upsert, бо це O(E+V) прохід).

        Видаляє:
          * ребра з `count < min_edge_count` та/або (якщо `drop_negative_npmi`)
            з `npmi < 0` — КРІМ ребер, інцидентних якірному (CVE/TTP) вузлу
            (верифіковані факти не ріжемо);
          * вузли з `doc_freq < min_doc_freq` — КРІМ якірних І КРІМ не-якірних
            вузлів, що мають якірного сусіда (інакше `remove_nodes_from`
            каскадно знесе щойно захищене на кроці 1 якірне ребро — вузол
            без ребра не буває);
          * не-якірні вузли, що лишились ІЗОЛЬОВАНИМИ (без жодного ребра)
            після чистки ребер вище — незалежно від їхнього doc_freq (це
            завжди безпечно: немає ребра — немає що каскадно зносити).

        n_docs/seen_docs (сам факт розміру корпусу) НЕ чіпаються — prune
        чистить лише структуру звʼязків, не переписує історію.

        KNOWN LIMITATION (свідомо поза scope): degree якірного вузла нічим не
        обмежений — якір може накопичити необмежену кількість сусідів
        (кожен захищений як «інцидентний факту»). Це майбутній temporal/
        global cap, не в цьому MVP.

        Пороги — з аргументів або (за замовчуванням) з env
        REASONER_PRUNE_MIN_DOC_FREQ / REASONER_PRUNE_MIN_EDGE_COUNT /
        REASONER_PRUNE_DROP_NEGATIVE_NPMI.

        Повертає {"nodes_removed": int, "edges_removed": int} — РЕАЛЬНА
        кількість видалених вузлів/ребер (edges_removed рахує підсумкову
        різницю `number_of_edges()` до/після, тобто враховує і явно зрізані
        на кроці 1, і каскадно зняті разом з вузлами на кроці 2)."""
        if min_doc_freq is None:
            min_doc_freq = _int_env("REASONER_PRUNE_MIN_DOC_FREQ", 2)
        if min_edge_count is None:
            min_edge_count = _int_env("REASONER_PRUNE_MIN_EDGE_COUNT", 2)
        if drop_negative_npmi is None:
            drop_negative_npmi = _bool_env("REASONER_PRUNE_DROP_NEGATIVE_NPMI", True)

        edges_before = self.G.number_of_edges()

        # 1. Ребра.
        edges_to_drop = []
        for a, b, data in self.G.edges(data=True):
            if self.G.nodes[a].get("is_anchor") or self.G.nodes[b].get("is_anchor"):
                continue  # факт — не ріжемо
            weak_count = data.get("count", 0) < min_edge_count
            weak_npmi = drop_negative_npmi and self.npmi(a, b) < 0
            if weak_count or weak_npmi:
                edges_to_drop.append((a, b))
        self.G.remove_edges_from(edges_to_drop)

        # 2. Вузли: ізольовані не-якірні (завжди) + низька doc_freq (окрім
        #    якірних І окрім не-якірних з якірним сусідом — щоб не знести
        #    захищене на кроці 1 якірне ребро каскадом remove_nodes_from).
        def _has_anchor_neighbor(n: str) -> bool:
            return any(self.G.nodes[nb].get("is_anchor") for nb in self.G.neighbors(n))

        nodes_to_drop = [
            n for n, d in self.G.nodes(data=True)
            if not d.get("is_anchor")
            and (
                self.G.degree(n) == 0
                or (d.get("doc_freq", 0) < min_doc_freq and not _has_anchor_neighbor(n))
            )
        ]
        self.G.remove_nodes_from(nodes_to_drop)

        edges_after = self.G.number_of_edges()
        return {"nodes_removed": len(nodes_to_drop), "edges_removed": edges_before - edges_after}

    # --- Персистентність ------------------------------------------------
    def save(self, path: str | None = None) -> str:
        """Зберігає граф на диск (pickle — зберігає set-атрибути без втрат).

        SECURITY: pickle при завантаженні виконує код, тож файл графа — це,
        по суті, виконуваний артефакт. На спільному боксі права 0600 (лише
        власник) не дають сусідньому користувачу підмінити його й отримати
        виконання коду в нашому процесі. На Windows chmod майже no-op —
        помилку глушимо, це defense-in-depth, а не єдиний бар'єр."""
        path = path or REASONER_GRAPH_PATH
        with open(path, "wb") as f:
            pickle.dump(self.G, f, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "SemanticReasoner":
        """Завантажує граф з диска."""
        path = path or REASONER_GRAPH_PATH
        with open(path, "rb") as f:
            graph = pickle.load(f)
        return cls(graph)

    def to_graphml(self) -> str:
        """Експорт у GraphML (для Gephi). set-атрибути (docs) серіалізуються у
        рядок, бо GraphML не тримає складних типів."""
        H = nx.Graph()
        H.graph["n_docs"] = self.G.graph.get("n_docs", 0)
        for n, d in self.G.nodes(data=True):
            H.add_node(n, kind=d.get("kind", "term"),
                       doc_freq=int(d.get("doc_freq", 1)),
                       is_anchor=bool(d.get("is_anchor", False)))
        for a, b, d in self.G.edges(data=True):
            H.add_edge(a, b, count=int(d.get("count", 1)),
                       docs=",".join(sorted(str(x) for x in d.get("docs", set()))))
        return "\n".join(nx.generate_graphml(H))

    # --- Ваги / вартість ------------------------------------------------
    def _node_boost(self, node: str) -> float:
        return float(self.G.nodes[node].get("boost", 1.0)) if self.G.has_node(node) else 1.0

    def edge_weight(self, a: str, b: str, weight_mode: str = "raw") -> float:
        """Ефективна вага ребра (більше = сильніший звʼязок).

        raw  — сирий лічильник співзустрічальності;
        npmi — normalized pointwise mutual information ∈ [-1, 1], зсунута в
               (0, ~2] через (1 + npmi), щоб приборкати хаб-ефект (часті вузли
               не домінують просто за рахунок частоти).
        Обидва режими домножуються на геосереднє boost кінців (якірні
        верифіковані вузли «важать» більше)."""
        if not self.G.has_edge(a, b):
            return 0.0
        count = self.G[a][b]["count"]
        boost = math.sqrt(self._node_boost(a) * self._node_boost(b))
        if weight_mode == "npmi":
            base = 1.0 + self.npmi(a, b)          # (0, ~2], монотонно за npmi
            return max(base, _EPS) * boost
        return count * boost

    def npmi(self, a: str, b: str) -> float:
        """Normalized PMI пари вузлів на основі документних частот.

        npmi = pmi / (-log p(a,b)),  pmi = log( p(a,b) / (p(a)·p(b)) ).
        Діапазон [-1, 1]: 1 — завжди разом, 0 — незалежні, <0 — уникають.
        При n_docs=0 або відсутньому ребрі повертає -1 (немає свідчень)."""
        if not self.G.has_edge(a, b):
            return -1.0
        n = self.G.graph.get("n_docs", 0)
        if n <= 0:
            return -1.0
        c_ab = self.G[a][b]["count"]
        f_a = self.G.nodes[a].get("doc_freq", 0)
        f_b = self.G.nodes[b].get("doc_freq", 0)
        if c_ab <= 0 or f_a <= 0 or f_b <= 0:
            return -1.0
        p_ab = c_ab / n
        p_a = f_a / n
        p_b = f_b / n
        pmi = math.log(p_ab / (p_a * p_b))
        denom = -math.log(p_ab)
        if denom <= _EPS:                          # p_ab == 1 → повний збіг
            return 1.0
        return max(-1.0, min(1.0, pmi / denom))

    def _cost(self, a: str, b: str, weight_mode: str) -> float:
        """Вартість ребра для Dijkstra: обернена до ваги (сильне ребро = дешеве),
        тож найсильніший ланцюг = найкоротший шлях."""
        w = self.edge_weight(a, b, weight_mode)
        return 1.0 / max(w, _EPS)

    def _cost_fn(self, weight_mode: str):
        """networkx-сумісна функція вартості weight(u, v, data)."""
        return lambda u, v, _d: self._cost(u, v, weight_mode)

    # --- Traversal API --------------------------------------------------
    def neighbors(self, node: str, min_weight: float = 1.0,
                  weight_mode: str = "raw") -> list[tuple[str, float]]:
        """Сусіди вузла з ефективною вагою >= min_weight, за спаданням ваги."""
        if not self.G.has_node(node):
            return []
        out = []
        for nb in self.G.neighbors(node):
            w = self.edge_weight(node, nb, weight_mode)
            if w >= min_weight:
                out.append((nb, w))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def top_related(self, node: str, k: int = 10,
                    weight_mode: str = "raw") -> list[tuple[str, float]]:
        """Топ-k найсильніше повʼязаних сусідів вузла."""
        return self.neighbors(node, min_weight=0.0, weight_mode=weight_mode)[:k]

    def shortest_path(self, a: str, b: str,
                      weight_mode: str = "raw") -> dict | None:
        """Найсильніший (= найдешевший за вартістю) шлях між a та b.

        Повертає dict з ланцюгом, ребрами, сумарною вагою та джерелами, або
        None, якщо шляху немає / вузлів немає в графі."""
        if not (self.G.has_node(a) and self.G.has_node(b)):
            return None
        try:
            path = nx.dijkstra_path(self.G, a, b, weight=self._cost_fn(weight_mode))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return self._describe_path(path, weight_mode)

    def bidirectional_path(self, start_concepts, goal_concept: str,
                           weight_mode: str = "raw") -> dict | None:
        """Двонапрямний зважений обхід: мультистарт із `start_concepts` вперед +
        від `goal_concept` назад, зустріч посередині.

        Це операціоналізація логіки Ланде «будувати рішення з кінця»: пошук іде
        і від наявних свідчень (start), і назад від цільового результату (goal).
        Повертає найсильніший ланцюг start→goal (dict як у `_describe_path`,
        додатково: 'start', 'meet') або None."""
        if isinstance(start_concepts, str):
            start_concepts = [start_concepts]
        starts = [s for s in start_concepts if self.G.has_node(s)]
        if not starts or not self.G.has_node(goal_concept):
            return None
        if goal_concept in starts:
            return self._describe_path([goal_concept], weight_mode) | {
                "start": goal_concept, "meet": goal_concept}

        best = self._bidir_dijkstra(set(starts), goal_concept, weight_mode)
        if best is None:
            return None
        path, meet = best
        desc = self._describe_path(path, weight_mode)
        desc["start"] = path[0]
        desc["meet"] = meet
        return desc

    def hypotheses(self, goal_concept: str, k: int = 5, weight_mode: str = "npmi",
                   anchors_only: bool = False, max_hops: int = 6,
                   min_total_weight: float = 0.0) -> list[dict]:
        """Predict-режим: від цільового результату розкриває НАЗАД підтримувальні
        ланцюги на реальному корпусі й повертає ранжований список гіпотез.

        Реалізація: одне Dijkstra-розкриття від goal до всіх досяжних вузлів
        (це і є «обхід з кінця»). Кандидати-старти — досяжні вузли (за
        замовчуванням якірні верифіковані сутності мають перевагу, але не лише
        вони, якщо anchors_only=False). Кожна гіпотеза — це ланцюг start→goal.

        Кожен елемент:
          {rank, start, goal, chain, edges, total_weight, cost, hops,
           source_ids, score, narrative_stub}
        Формат готовий до передачі в LLM (див. `format_hypotheses_for_llm`).

        Скоринг: `score` = СЕРЕДНЯ вага ребра ланцюга (total_weight / hops),
        НЕ обернена сума dijkstra-costs. Причина: cost = сума (1/w) по ребрах
        штрафує багатохопові ланцюги за саму довжину (кожен додатковий хоп —
        ще один доданок 1/w), тож сильний 3-хоповий ланцюг тонув під слабшим
        1-хоповим лише через довжину. Середня вага ребра — чесніша міра
        «наскільки сильний ланцюг у середньому», незалежна від довжини; для
        1-хопових ланцюгів вона рівна total_weight (hops=1), тобто зберігає
        стару поведінку в цьому випадку. Поле `cost` (сирий dijkstra-cost)
        лишається для телеметрії/сумісності, у ранжуванні більше НЕ бере
        участі напряму (крім бенефіту від самого dijkstra-розкриття, яке й
        так гарантує НАЙДЕШЕВШИЙ шлях до кожного вузла, тобто найкращий
        кандидат-ланцюг саме до нього)."""
        if not self.G.has_node(goal_concept):
            return []
        cost_fn = self._cost_fn(weight_mode)
        # single_source_dijkstra: найдешевші шляхи від goal до всіх вузлів.
        dist, paths = nx.single_source_dijkstra(self.G, goal_concept, weight=cost_fn)

        candidates = []
        for node, cost in dist.items():
            if node == goal_concept:
                continue
            path_from_goal = paths[node]
            hops = len(path_from_goal) - 1
            if hops < 1 or hops > max_hops:
                continue
            is_anchor = self.G.nodes[node].get("is_anchor", False)
            if anchors_only and not is_anchor:
                continue
            chain = list(reversed(path_from_goal))    # start → … → goal
            desc = self._describe_path(chain, weight_mode)
            if desc["total_weight"] < min_total_weight:
                continue
            mean_weight = desc["total_weight"] / hops  # hops>=1 гарантовано вище
            candidates.append({
                "chain": chain,
                "desc": desc,
                "cost": cost,
                "hops": hops,
                "is_anchor": is_anchor,
                "mean_weight": mean_weight,
            })

        # Ранжування: (1) якірні (факти) раніше неякірних; (2) далі за
        # спаданням середньої ваги ребра (score); (3) тай-брейк — менше хопів
        # раніше; (4) фінальний тай-брейк — імʼя start-вузла (детермінізм).
        candidates.sort(key=lambda c: (
            0 if c["is_anchor"] else 1,
            -c["mean_weight"],
            c["hops"],
            c["chain"][0],
        ))

        hyps = []
        for c in candidates[:k]:
            desc = c["desc"]
            hyp = {
                "start": c["chain"][0],
                "goal": goal_concept,
                "chain": desc["path"],
                "edges": desc["edges"],
                "total_weight": desc["total_weight"],
                "cost": c["cost"],
                "hops": c["hops"],
                "is_anchor": c["is_anchor"],
                "source_ids": desc["source_ids"],
                # score: більше = краще. Середня вага ребра ланцюга.
                "score": c["mean_weight"],
            }
            hyp["narrative_stub"] = " → ".join(hyp["chain"])
            hyps.append(hyp)

        for i, h in enumerate(hyps, 1):
            h["rank"] = i
        return hyps

    def resolve_node(self, raw) -> str | None:
        """Ввід користувача → реальний вузол графа, або None якщо такого немає."""
        for candidate in concept_candidates(raw):
            if self.G.has_node(candidate):
                return candidate
        return None

    def hypotheses_from(self, start_concepts, goal_concept: str, k: int = 5,
                        weight_mode: str = "npmi", max_hops: int = 6,
                        bidirectional: bool = True) -> list[dict]:
        """ДВОФРОНТОВИЙ режим: ланцюги від ЗАДАНИХ стартів до цілі.

        Відмінність від `hypotheses()` принципова, а не косметична. Там —
        одне розкриття Дейкстри від цілі назад, а старти вибирає сам алгоритм
        (беруться найсильніші досяжні вузли). Тут аналітик НАЗИВАЄ, від чого
        йти, і для кожного старту запускається `bidirectional_path` — два
        фронти назустріч, вперед від свідчення і назад від цілі.

        Це і є «двостороннє формування» з методики: рух з двох боків замість
        одного. На контрольному наборі (`experiments/htb_reconstruction`) цей
        режим дає помітно кращі покриття й точність, ніж односторонній.

        Формат елемента ТОЙ САМИЙ, що в `hypotheses()` — UI малює обидва
        результати одним кодом. Один старт → щонайбільше один ланцюг.

        `bidirectional=False` — КОНТРОЛЬНА гілка для експерименту, не режим для
        UI. Вона лишає все як є (задані старти, ранжування, k, стеля хопів) і
        міняє ЛИШЕ примітив обходу: замість двох фронтів — одне розкриття
        Дейкстри від цілі. Потрібна тому, що порівняння `hypotheses_from` із
        `hypotheses()` змішує два фактори — обхід і те, ХТО обирає старти, — і
        без такого контролю не можна сказати, який із них дав ефект."""
        goal = self.resolve_node(goal_concept)
        if goal is None:
            return []
        if isinstance(start_concepts, str):
            start_concepts = [start_concepts]

        # Резолвимо ДО пошуку: різні написання того самого концепту («cve-…»
        # та «CVE-…») мають дати один старт, а не дубльований ланцюг.
        resolved: list[str] = []
        for raw in start_concepts:
            node = self.resolve_node(raw)
            if node is not None and node != goal and node not in resolved:
                resolved.append(node)
        if not resolved:
            return []

        # Односторонній контроль: одне розкриття від цілі на всі старти —
        # рівно те, що робить `hypotheses()`, але без вибору стартів алгоритмом.
        uni_paths = None
        if not bidirectional:
            if not self.G.has_node(goal):
                return []
            _dist, uni_paths = nx.single_source_dijkstra(
                self.G, goal, weight=self._cost_fn(weight_mode))

        candidates = []
        for start in resolved:
            if bidirectional:
                found = self.bidirectional_path([start], goal, weight_mode=weight_mode)
            else:
                path_from_goal = uni_paths.get(start)
                # start → … → goal: розкриття йшло від цілі, тож розвертаємо.
                found = (self._describe_path(list(reversed(path_from_goal)), weight_mode)
                         if path_from_goal and len(path_from_goal) > 1 else None)
            if not found:
                continue
            chain = found["path"]
            hops = len(chain) - 1
            if hops < 1 or hops > max_hops:
                continue
            candidates.append({
                "desc": found,
                "hops": hops,
                "is_anchor": bool(self.G.nodes[start].get("is_anchor", False)),
                # Скоринг той самий, що в `hypotheses()`: середня вага ребра,
                # щоб довгі ланцюги не топились за саму лише довжину.
                "mean_weight": found["total_weight"] / hops,
                "cost": sum(self._cost(a, b, weight_mode)
                            for a, b in zip(chain, chain[1:])),
            })

        candidates.sort(key=lambda c: (
            0 if c["is_anchor"] else 1,
            -c["mean_weight"],
            c["hops"],
            c["desc"]["path"][0],
        ))

        hyps = []
        for rank, c in enumerate(candidates[:k], 1):
            desc = c["desc"]
            hyp = {
                "start": desc["path"][0],
                "goal": goal,
                "chain": desc["path"],
                "edges": desc["edges"],
                "total_weight": desc["total_weight"],
                "cost": c["cost"],
                "hops": c["hops"],
                "is_anchor": c["is_anchor"],
                "source_ids": desc["source_ids"],
                "score": c["mean_weight"],
            }
            hyp["narrative_stub"] = " → ".join(hyp["chain"])
            hyp["rank"] = rank
            hyps.append(hyp)
        return hyps

    # --- Внутрішні ------------------------------------------------------
    def _describe_path(self, path: list[str], weight_mode: str) -> dict:
        """Збирає опис шляху: ребра з вагами, сумарна вага, джерела (mongo_id)."""
        edges = []
        source_ids: set = set()
        total = 0.0
        for a, b in zip(path, path[1:]):
            w = self.edge_weight(a, b, weight_mode)
            edges.append((a, b, w))
            total += w
            source_ids |= set(self.G[a][b].get("docs", set()))
        return {
            "path": list(path),
            "edges": edges,
            "total_weight": total,
            "source_ids": sorted(s for s in source_ids if s),
        }

    def _bidir_dijkstra(self, sources: set, target: str, weight_mode: str):
        """Двонапрямний Dijkstra на неорієнтованому графі (мультистарт → target).

        Класична схема з двома фронтами: вперед від sources, назад від target.
        Оскільки граф неорієнтований, суміжність однакова в обидва боки.
        Зупинка — коли сума мінімумів обох черг >= найкраще знайдене mu.
        Повертає (path, meet_node) або None."""
        import heapq

        dist_f = {s: 0.0 for s in sources}
        dist_b = {target: 0.0}
        prev_f: dict = {s: None for s in sources}
        prev_b: dict = {target: None}
        seen_f: set = set()
        seen_b: set = set()
        pq_f = [(0.0, s) for s in sources]
        heapq.heapify(pq_f)
        pq_b = [(0.0, target)]

        mu = math.inf
        meet = None

        def relax(u, du, dist, prev, pq, other_dist, other_seen):
            nonlocal mu, meet
            for v in self.G.neighbors(u):
                w = self._cost(u, v, weight_mode)
                nd = du + w
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
                # Оновлюємо mu на КОЖНОМУ ребрі до вузла, вже досягнутого іншим
                # фронтом (а не лише при покращенні dist[v]) — інакше можна
                # проґавити оптимальну точку зустрічі.
                if v in other_dist:
                    total = nd + other_dist[v]
                    if total < mu:
                        mu = total
                        meet = v

        while pq_f and pq_b:
            top_f = pq_f[0][0]
            top_b = pq_b[0][0]
            if top_f + top_b >= mu:
                break
            # Розвиваємо той фронт, що «дешевший» — балансує роботу.
            if top_f <= top_b:
                du, u = heapq.heappop(pq_f)
                if u in seen_f:
                    continue
                seen_f.add(u)
                relax(u, du, dist_f, prev_f, pq_f, dist_b, seen_b)
            else:
                du, u = heapq.heappop(pq_b)
                if u in seen_b:
                    continue
                seen_b.add(u)
                relax(u, du, dist_b, prev_b, pq_b, dist_f, seen_f)

        if meet is None:
            return None

        # Реконструкція: sources → meet (вперед) + meet → target (назад).
        left = []
        node = meet
        while node is not None:
            left.append(node)
            node = prev_f[node]
        left.reverse()

        right = []
        node = prev_b[meet]
        while node is not None:
            right.append(node)
            node = prev_b[node]

        return left + right, meet


# =========================================================================
#  Форматування під LLM
# =========================================================================
def format_hypotheses_for_llm(hyps: list[dict], goal_concept: str,
                              max_sources: int = 5,
                              label_map: dict | None = None) -> str:
    """Готує компактний текстовий блок гіпотез для передачі в hermes3.

    Свідомо детермінований, без вигадок: лише ланцюги, ваги й ідентифікатори
    джерел, витягнуті з реального корпусу.

    `label_map` (опційно) — dict[node_id -> читабельна мітка]; вузли, для яких
    є запис, рендеряться як "node_id (Мітка)". Без label_map (None/{}) вивід
    ідентичний попередній поведінці (регрес-гард)."""
    def _annotate(node: str) -> str:
        if label_map:
            label = label_map.get(node)
            if label:
                return f"{node} ({label})"
        return node

    if not hyps:
        return (f"Ціль: {_annotate(goal_concept)}\n"
                "У наявному корпусі не знайдено підтримувальних ланцюгів до цієї цілі.")
    lines = [f"ЦІЛЬОВИЙ РЕЗУЛЬТАТ (goal): {_annotate(goal_concept)}",
             f"Знайдено гіпотез: {len(hyps)} (ранжовано за силою звʼязку).", ""]
    for h in hyps:
        chain = " → ".join(_annotate(node) for node in h["chain"])
        srcs = h["source_ids"][:max_sources]
        src_txt = ", ".join(srcs) if srcs else "—"
        # Пом'якшений ярлик: якір верифікує лише ІДЕНТИФІКАТОР сутності
        # (CVE/MITRE ID валідний) — а не факт, що документ демонструє техніку.
        anchor = " [ЯКІР: код сутності верифіковано в MITRE/NVD]" if h.get("is_anchor") else ""
        n_sources = len(h["source_ids"])
        if n_sources == 0:
            weak = " (немає джерел — не підтверджено)"
        elif n_sources == 1:
            weak = " (одне джерело — слабке підтвердження)"
        else:
            weak = ""
        lines.append(
            f"Гіпотеза #{h['rank']}{anchor}\n"
            f"  Ланцюг (start→goal): {chain}\n"
            f"  Кроків: {h['hops']} | сумарна вага: {h['total_weight']:.3f} "
            f"| середня вага ребра: {h['score']:.3f}\n"
            f"  Джерел: {n_sources}{weak}\n"
            f"  Джерела (mongo_id, до {max_sources}): {src_txt}"
        )
        lines.append("")
    return "\n".join(lines).strip()


# --- Зручний фасад для UI/скриптів ---------------------------------------
def build_reasoner_from_records(records) -> SemanticReasoner:
    """Одноразова побудова резонера з корпусу (без персистентності)."""
    return SemanticReasoner().build_from_records(records)
