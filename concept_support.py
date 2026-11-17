"""Опора розкриття понять на корпус: які фрагменти джерел давати моделі.

НАВІЩО. Метод «двосторонньої семантичної мережі» (`semantic_bridge.py`) перед
розкриттям кожного поняття може дати моделі фрагменти корпусу (RAG). Запит —
англійська форма поняття. Перша реалізація в ONYX була голим KNN по збережених
векторах `embedding` із порогом `KNN_MAX_DISTANCE = 0.30`, і виміри на живому
індексі (1009 документів, 43 реальні поняття) показали, що вона мовчки нічого
не давала.

ЩО ВИМІРЯНО.
  * Шиппер ембедить СИРИЙ ОРИГІНАЛ посту (`doc["text"]`): 832 документи
    українською, 174 російською, 3 англійською. `nomic-embed-text` —
    англоцентричний ембедер.
  * Англійський запит проти цих векторів: навіть релевантні документи лежать на
    косинусній відстані 0.53-0.58, а найближчі сусіди — «хаби» (сміттєві
    документи). Поріг 0.30 не пройшло жодне з 43 понять (0/43): RAG не додавав
    нічого й не казав про це.
  * Український запит проти тих самих векторів: УСЕ потрапляє в 0.10-0.27, а
    нагорі знову хаби. Це зламало б поріг і підсунуло б моделі сміття, тому
    українську форму НЕ використовуємо.
  * Повнотекстовий пошук по title/summary сам по собі не годиться: OR тягне
    документ за одним спільним словом (кожне поняття зі словом «segment»
    діставало «Audio Segment Removal and Browser Image Editor»); кворум по
    інформативних термах — ≈40% точності; IDF-покриття чіткої межі не дало.

ЩО ОБРАНО (без переіндексації) — два етапи:
  1. BM25 (OR-збіг англійського поняття) по ЗБЕРЕЖЕНИХ англійських полях
     `title`, `summary` (їх пише модель) -> до `SUPPORT_CANDIDATES` кандидатів.
     Поняття методу постійно називають вузли (`web-01`, `net-001`), а терм із
     дефісом у OR-виразі мовчки перетворював OR на AND — поняття з `net-001`
     давало 0 кандидатів замість 63. Тепер такий терм — одна фраза
     (`search_utils._as_or_operand`);
  2. ембедимо англійський запит і «title. summary» кожного кандидата тим самим
     nomic і лишаємо тих, чия косинусна відстань < `SUPPORT_MAX_DISTANCE`,
     найближчі `SUPPORT_FRAGMENTS`.
Результат: опору отримали 16 із 43 понять (було 0); із 25 виданих фрагментів
явно мимо — 1, по дотичній — близько 5, решта по темі (ручна розмітка, один
розмітник). Покриття однакове для K=10/20/40; усі 43 поняття — ≈5-7 с завдяки
кешу векторів документів між запитами. Корені мережі (сам текст проблеми й
розв'язку) опори не отримують: їхня «англійська форма» — це український текст
користувача, і BM25 по англійських полях кандидатів не дає. Сміття це не додає.

ПІСЛЯ ПЕРЕРАХУНКУ ІНДЕКСУ. Тепер шиппер рахує вектори від англійського «title.
summary» (`embedding_text.py`), і голий KNN по них уже не марний: 20 із 43 понять.
Але приблизно третина його фрагментів не по темі, тож двоетапний відбір лишився
— він точніший. Текст документа для ембедера — той самий `embedding_text`, що й у
шиппера, тому вектори кандидатів і індексу лежать в одному просторі.

Модуль чистий — без streamlit/pymysql/httpx: пошук і ембедер передаються
функціями, тож усе тестується без мережі й бази.
"""

from __future__ import annotations

from typing import Callable

from embedding_text import EMBED_TEXT_CHARS, embedding_text
from semantic_bridge import cosine

#: Косинусна відстань nomic: англійське поняття проти англійського «title.
#: summary». Окрема від `ui.KNN_MAX_DISTANCE` свідомо: це інше подання тексту
#: (модельні title/summary, а не сирий оригінал), і підлаштування чату не має
#: непомітно змінювати метод.
SUPPORT_MAX_DISTANCE = 0.30
#: Скільки BM25-кандидатів переглядати на другому етапі.
SUPPORT_CANDIDATES = 20
#: Скільки фрагментів віддавати моделі на одне розкриття.
SUPPORT_FRAGMENTS = 2
#: Довжина одного фрагмента, символів (як max_len у wrap_untrusted).
FRAGMENT_CHARS = 500
#: Стеля тексту документа, що йде в ембедер. Без неї довгий документ
#: завантаженого в стенд корпусу впирається в контекст моделі. У корпусі ONYX
#: «title. summary» не довші за ~1000 символів, тож виміряний режим вона не чіпає.
#: Та сама стеля, що в шиппері (`embedding_text.EMBED_TEXT_CHARS`): подання
#: документа в UI й у індексі має збігатися.
DOCUMENT_CHARS = EMBED_TEXT_CHARS
#: Стеля кешу векторів документів; перевищення — кеш скидається цілком.
MAX_CACHED_VECTORS = 5000
#: Ідентифікатор способу відбору. Іде в ключ кешу побудованих мереж, щоб мережі,
#: зібрані зі старим (зламаним) пошуком, не підхоплювались як актуальні.
SUPPORT_SCHEME = "bm25:title,summary -> nomic:title.summary < 0.30"

#: (запит, k) -> кандидати з полями title/summary, за можливості id і content.
Search = Callable[[str, int], list[dict]]
Embed = Callable[[list[str]], list[list[float]]]


def document_text(doc: dict) -> str:
    """Англійське подання документа: «title. summary». Якщо реального summary
    нема — запасний варіант `content` (напр. завантажений у стенд корпус без
    summary). Правило спільне з шиппером — `embedding_text.embedding_text`."""
    return embedding_text(doc.get("title"), doc.get("summary"), doc.get("content"))


class SupportRetriever:
    """`(запит) -> [фрагменти]`: BM25-кандидати, відфільтровані за ембединг-відстанню.

    Один екземпляр живе довше за один запит (у ONYX — на процес), тож кеш
    векторів документів переживає виклики: ті самі кандидати для сусідніх понять
    ембедяться раз."""

    def __init__(self, search: Search, embed: Embed, *,
                 max_distance: float = SUPPORT_MAX_DISTANCE,
                 candidates: int = SUPPORT_CANDIDATES,
                 limit: int = SUPPORT_FRAGMENTS,
                 max_cached: int = MAX_CACHED_VECTORS):
        self.search = search
        self.embed = embed
        self.max_distance = max_distance
        self.candidates = candidates
        self.limit = limit
        self.max_cached = max_cached
        self._cache: dict[str, list[float]] = {}

    def __call__(self, query: str) -> list[str]:
        query = (query or "").strip()
        if not query:
            return []
        texts = [text for text in (document_text(doc) for doc in
                                   self.search(query, self.candidates)) if text]
        if not texts:
            return []

        # Ключ кешу — сам текст, а не id: вектор залежить лише від тексту. Ключ за
        # id дав би однаковим id у завантаженому JSON один чужий вектор на всіх, а
        # оновленому summary — старий вектор.
        #
        # Ретривер спільний для сесій Streamlit, і інша сесія може скинути кеш
        # будь-якої миті. Тому читаємо одним атомарним get (а не `in` + `[]`) у
        # локальну мапу й далі оцінюємо лише з неї — KeyError неможливий.
        vectors = {}
        for text in texts:
            vec = self._cache.get(text)
            if vec is not None:
                vectors[text] = vec
        missing = list(dict.fromkeys(t for t in texts if t not in vectors))
        embedded = self.embed([query] + missing)
        if len(embedded) != len(missing) + 1:
            raise ValueError(f"ембедер повернув {len(embedded)} векторів замість "
                             f"{len(missing) + 1}")
        query_vec = embedded[0]
        fresh = dict(zip(missing, embedded[1:]))
        vectors.update(fresh)
        self._remember(fresh)

        scored = []
        for index, text in enumerate(texts):
            distance = 1.0 - cosine(query_vec, vectors[text])
            if distance < self.max_distance:
                scored.append((distance, index, text))
        scored.sort(key=lambda item: (item[0], item[1]))
        # Однаковий текст під різними id (передрук іншим каналом) — один фрагмент:
        # повтор у промпті нічого не дає, лише витісняє наступний інший.
        fragments: list[str] = []
        for _d, _i, text in scored:
            if len(fragments) >= self.limit:
                break
            fragment = text[:FRAGMENT_CHARS]
            if fragment not in fragments:
                fragments.append(fragment)
        return fragments

    def _remember(self, fresh: dict[str, list[float]]) -> None:
        """Покласти нові вектори в кеш; стеля — простий скид усього кешу."""
        if len(self._cache) + len(fresh) > self.max_cached:
            self._cache.clear()
        for key, vec in list(fresh.items())[:self.max_cached]:
            self._cache[key] = vec


def corpus_retriever(documents: list[dict], embed: Embed) -> "SupportRetriever | None":
    """Опора на корпус у пам'яті (стенд: завантажений користувачем JSON).

    Правило й поріг ті самі, що в ONYX, але етапу BM25 немає: кандидати — увесь
    корпус. Кеш вміщує його цілком, інакше корпус, більший за MAX_CACHED_VECTORS,
    переембеджувався б на кожному понятті. Чесне застереження: поріг 0.30 виміряно
    на ONYX (Ollama nomic, «title. summary»); для чужого корпусу й іншого
    ембедера його ніхто не перевіряв."""
    if not documents:
        return None
    return SupportRetriever(lambda _query, _k: documents, embed,
                            max_cached=max(MAX_CACHED_VECTORS, len(documents)))


def support_message(stats: dict | None) -> tuple[str, str] | None:
    """Що сказати користувачеві про опору на корпус: (рівень, текст) або None.

    stats — `SemanticNet.support`: asked (скільки розкриттів питали корпус),
    supported (скільком дісталися фрагменти), errors (скільки запитів упало).
    None — опору не вмикали, нічого казати не треба."""
    if not stats or not stats.get("asked"):
        return None
    asked = int(stats["asked"])
    supported = int(stats.get("supported") or 0)
    errors = int(stats.get("errors") or 0)
    if supported == 0 and errors >= asked:
        return ("error", f"Опора на корпус недоступна: жоден із {asked} запитів не "
                         "вдався, фронти росли без неї.")
    if supported == 0:
        return ("warning", f"Корпус не дав жодного релевантного фрагмента для жодного з "
                           f"{asked} розкриттів: фронти росли без опори на джерела.")
    text = f"Опора на корпус: {supported} з {asked} розкриттів отримали фрагменти джерел."
    if errors:
        text += f" Запитів до корпусу, що не вдались: {errors}."
    return ("info", text)
