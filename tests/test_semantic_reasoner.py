"""
Юніт-тести для semantic_reasoner.py.

Запуск:
    pytest tests/test_semantic_reasoner.py -v
або як звичайний скрипт (без pytest):
    python tests/test_semantic_reasoner.py
"""

import math
import os
import sys

# Дозволяємо запуск і як `pytest`, і як `python tests/...` з кореня репо.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_reasoner import SemanticReasoner  # noqa: E402


def _reasoner_with_edges(edges, n_docs=None):
    """Синтетичний резонер: edges = [(a, b, count), ...].

    doc_freq вузла = сума count інцидентних ребер (грубе наближення, достатнє
    для перевірки обходу); n_docs задаємо явно для NPMI-тестів."""
    r = SemanticReasoner()
    G = r.G
    from collections import Counter
    freq = Counter()
    for a, b, c in edges:
        for node in (a, b):
            if not G.has_node(node):
                kind = ("cve" if node.startswith("CVE-")
                        else "ttp" if node.startswith("MITRE ") else "term")
                G.add_node(node, kind=kind, doc_freq=0,
                           is_anchor=kind in ("cve", "ttp"),
                           boost=1.0)  # boost=1 → перевіряємо чисту топологію
        G.add_edge(a, b, count=c, docs={f"doc_{a}_{b}"})
        freq[a] += c
        freq[b] += c
    for node, f in freq.items():
        G.nodes[node]["doc_freq"] = f
    G.graph["n_docs"] = n_docs if n_docs is not None else sum(c for _, _, c in edges)
    return r


# --- bidirectional_path: правильний найсильніший шлях ---------------------
def test_bidirectional_picks_strongest_path():
    """Два маршрути start→goal: сильний (великі count) та слабкий обхідний.
    Найсильніший шлях = найдешевший за вартістю (cost=1/вага)."""
    # Сильний прямий місток: A-M-Z (ваги 10, 10).
    # Слабкий довгий обхід:  A-P-Q-Z (ваги 1, 1, 1).
    r = _reasoner_with_edges([
        ("A", "M", 10), ("M", "Z", 10),
        ("A", "P", 1), ("P", "Q", 1), ("Q", "Z", 1),
    ])
    res = r.bidirectional_path(["A"], "Z", weight_mode="raw")
    assert res is not None
    assert res["path"] == ["A", "M", "Z"], res["path"]
    assert res["start"] == "A"
    # Точка зустрічі має лежати на знайденому шляху.
    assert res["meet"] in res["path"]


def test_bidirectional_matches_dijkstra():
    """Двонапрямний обхід має давати ту саму вартість, що й прямий Dijkstra."""
    r = _reasoner_with_edges([
        ("A", "B", 5), ("B", "C", 5), ("C", "D", 5),
        ("A", "X", 2), ("X", "D", 2),
    ])
    bi = r.bidirectional_path(["A"], "D", weight_mode="raw")
    sp = r.shortest_path("A", "D", weight_mode="raw")
    assert bi is not None and sp is not None
    # Обидва мають знайти той самий (найсильніший) ланцюг.
    assert bi["path"] == sp["path"], (bi["path"], sp["path"])


def test_bidirectional_multi_start_meets_in_middle():
    """Мультистарт: із двох стартів обирається той, що дає сильніший ланцюг,
    і зустріч реально відбувається в проміжному вузлі."""
    # Старт S1 підключений слабко, S2 — сильно; goal один.
    r = _reasoner_with_edges([
        ("S1", "MID", 1), ("MID", "GOAL", 1),
        ("S2", "MID", 9), ("MID", "GOAL", 1),  # друге ребро MID-GOAL злиється
    ])
    res = r.bidirectional_path(["S1", "S2"], "GOAL", weight_mode="raw")
    assert res is not None
    assert res["start"] == "S2", res["path"]        # сильніший старт
    assert res["meet"] == "MID"                       # зустріч посередині
    assert res["path"] == ["S2", "MID", "GOAL"]


def test_bidirectional_no_path():
    """Роз'єднані компоненти → None."""
    r = _reasoner_with_edges([("A", "B", 3), ("X", "Y", 3)])
    assert r.bidirectional_path(["A"], "Y", weight_mode="raw") is None


# --- save / load ----------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    """Граф (з count, docs-множинами, n_docs) переживає save→load без втрат."""
    records = [
        {"mongo_id": "m1", "title": "ransomware attack cisco",
         "summary": "phishing dropped ransomware", "verified_cve_ids": "CVE-2024-0001",
         "verified_ttp_ids": "T1059"},
        {"mongo_id": "m2", "title": "ransomware phishing campaign",
         "summary": "cisco router exploited", "verified_cve_ids": "CVE-2024-0001",
         "verified_ttp_ids": ""},
    ]
    r = SemanticReasoner().build_from_records(records)
    path = str(tmp_path / "g.pkl")
    r.save(path)

    r2 = SemanticReasoner.load(path)
    assert r2.G.graph["n_docs"] == r.G.graph["n_docs"] == 2
    assert set(r2.G.nodes) == set(r.G.nodes)
    assert set(r2.G.edges) == set(r.G.edges)
    # Атрибути ребра (лічильник + множина джерел) збережені.
    for a, b, d in r.G.edges(data=True):
        d2 = r2.G[a][b]
        assert d2["count"] == d["count"]
        assert d2["docs"] == d["docs"]
    # Якірні верифіковані сутності присутні як вузли.
    assert "CVE-2024-0001" in r2.G
    assert "MITRE T1059" in r2.G
    assert r2.G.nodes["CVE-2024-0001"]["is_anchor"] is True


# --- NPMI -----------------------------------------------------------------
def test_npmi_perfect_cooccurrence():
    """Вузли, що зустрічаються ЛИШЕ разом і в кожному документі, дають npmi≈1."""
    # a,b у кожному з 4 документів разом → p_ab=1, p_a=p_b=1 → npmi=1.
    r = _reasoner_with_edges([("a", "b", 4)], n_docs=4)
    assert abs(r.npmi("a", "b") - 1.0) < 1e-9


def test_npmi_independent_pair():
    """Незалежні вузли дають npmi≈0."""
    # Побудуємо явно: N=100, f_a=10, f_b=10, c_ab=1 → p_ab=0.01=p_a*p_b → pmi=0.
    r = SemanticReasoner()
    G = r.G
    G.add_node("a", kind="term", doc_freq=10, is_anchor=False, boost=1.0)
    G.add_node("b", kind="term", doc_freq=10, is_anchor=False, boost=1.0)
    G.add_edge("a", "b", count=1, docs={"d"})
    G.graph["n_docs"] = 100
    assert abs(r.npmi("a", "b")) < 1e-9


def test_npmi_value_matches_formula():
    """Звіряємо npmi з формулою вручну на нетривіальних числах."""
    N, f_a, f_b, c_ab = 50, 10, 8, 5
    r = SemanticReasoner()
    G = r.G
    G.add_node("a", kind="term", doc_freq=f_a, is_anchor=False, boost=1.0)
    G.add_node("b", kind="term", doc_freq=f_b, is_anchor=False, boost=1.0)
    G.add_edge("a", "b", count=c_ab, docs={"d"})
    G.graph["n_docs"] = N

    p_ab = c_ab / N
    pmi = math.log(p_ab / ((f_a / N) * (f_b / N)))
    expected = pmi / (-math.log(p_ab))
    assert abs(r.npmi("a", "b") - expected) < 1e-9


def test_npmi_missing_edge():
    """Немає ребра → npmi = -1 (немає свідчень)."""
    r = _reasoner_with_edges([("a", "b", 2)], n_docs=2)
    assert r.npmi("a", "c") == -1.0


# --- anchor boost впливає на обхід ---------------------------------------
def test_anchor_boost_favours_verified_path():
    """За однакових сирих count шлях через якірний верифікований вузол має бути
    не слабшим за шлях через звичайний термін (boost>=1 не шкодить)."""
    r = SemanticReasoner()
    recs = [
        {"mongo_id": "m1", "title": "alpha beta", "summary": "",
         "verified_cve_ids": "", "verified_ttp_ids": ""},
        {"mongo_id": "m2", "title": "alpha", "summary": "gamma",
         "verified_cve_ids": "CVE-2020-9999", "verified_ttp_ids": ""},
    ]
    r.build_from_records(recs)
    # CVE-вузол має boost > 1, звичайний термін — рівно 1.
    assert r.G.nodes["CVE-2020-9999"]["boost"] > 1.0
    assert r.G.nodes["alpha"]["boost"] == 1.0


# --- hypotheses -----------------------------------------------------------
def test_hypotheses_returns_ranked_chains_with_sources():
    """hypotheses від цілі повертає ранжовані ланцюги з джерелами (mongo_id)."""
    records = [
        {"mongo_id": "s1", "title": "phishing email lockbit",
         "summary": "attacker sent phishing", "verified_cve_ids": "",
         "verified_ttp_ids": "T1566"},
        {"mongo_id": "s2", "title": "lockbit ransomware encryption",
         "summary": "lockbit deployed ransomware", "verified_cve_ids": "",
         "verified_ttp_ids": ""},
        {"mongo_id": "s3", "title": "phishing lockbit ransomware",
         "summary": "phishing led to ransomware lockbit", "verified_cve_ids": "",
         "verified_ttp_ids": "T1566"},
    ]
    r = SemanticReasoner().build_from_records(records)
    hyps = r.hypotheses("ransomware", k=5, weight_mode="raw")
    assert hyps, "мають бути гіпотези"
    # Ранги послідовні, у кожної є ланцюг, що завершується ціллю.
    assert [h["rank"] for h in hyps] == list(range(1, len(hyps) + 1))
    for h in hyps:
        assert h["chain"][-1] == "ransomware"
        assert h["chain"][0] == h["start"]
        assert h["total_weight"] > 0
    # Джерела складаються з реальних mongo_id корпусу.
    all_src = {s for h in hyps for s in h["source_ids"]}
    assert all_src <= {"s1", "s2", "s3"}
    assert all_src, "має бути хоча б одне джерело"


def test_hypotheses_unknown_goal():
    """Невідома ціль → порожній список (без винятку)."""
    r = _reasoner_with_edges([("a", "b", 2)], n_docs=2)
    assert r.hypotheses("nonexistent", k=3) == []


# --- per-doc term cap (контроль густоти графа) -----------------------------
def test_upsert_document_caps_non_anchor_terms_by_frequency():
    """Довгий документ (>40 унікальних не-якірних термінів) обрізається до
    REASONER_MAX_TERMS_PER_DOC (дефолт 40) за частотою появи в ТЕКСТІ дока:
    топ-N за частотою лишаються, рідкісні відкидаються. Якір (CVE) ніколи не
    ріжеться і завжди присутній понад cap."""
    top_words = [f"topterm{i:02d}" for i in range(40)]
    rare_words = [f"rareterm{i:02d}" for i in range(10)]
    # top-слова зустрічаються двічі в тексті, рідкісні — по разу.
    text = " ".join(w + " " + w for w in top_words) + " " + " ".join(rare_words)
    record = {
        "title": text, "summary": "",
        "verified_cve_ids": "CVE-2024-9999", "verified_ttp_ids": "",
    }
    r = SemanticReasoner()
    r.upsert_document("d1", record=record)

    term_nodes = {n for n, d in r.G.nodes(data=True) if d["kind"] == "term"}
    assert "CVE-2024-9999" in r.G.nodes
    assert r.G.nodes["CVE-2024-9999"]["is_anchor"] is True
    assert len(term_nodes) == 40, len(term_nodes)
    assert term_nodes == set(top_words), term_nodes - set(top_words)
    assert not (set(rare_words) & term_nodes)


def test_upsert_document_term_cap_zero_means_unlimited():
    """REASONER_MAX_TERMS_PER_DOC=0 (чи від'ємне) — трактується як «без ліміту»
    (старе поведінка, зворотна сумісність)."""
    words = [f"uncappedterm{i:02d}" for i in range(55)]
    text = " ".join(words)
    record = {"title": text, "summary": "", "verified_cve_ids": "", "verified_ttp_ids": ""}
    old = os.environ.get("REASONER_MAX_TERMS_PER_DOC")
    os.environ["REASONER_MAX_TERMS_PER_DOC"] = "0"
    try:
        r = SemanticReasoner()
        r.upsert_document("d1", record=record)
    finally:
        if old is None:
            os.environ.pop("REASONER_MAX_TERMS_PER_DOC", None)
        else:
            os.environ["REASONER_MAX_TERMS_PER_DOC"] = old
    term_nodes = {n for n, d in r.G.nodes(data=True) if d["kind"] == "term"}
    assert len(term_nodes) == 55, len(term_nodes)


# --- docs-set cap на ребрі (обмеження провенансу) ---------------------------
def test_edge_docs_set_capped_but_count_reflects_all_matches():
    """>200 доків з однією й тією ж парою концептів: len(docs) обмежено
    REASONER_MAX_DOCS_PER_EDGE (дефолт 200), але count лишається реальною
    кількістю співпоявин (джерело правди для npmi/ваг ребра)."""
    r = SemanticReasoner()
    n = 250
    for i in range(n):
        r.upsert_document(f"doc{i}", concepts={"alpha", "beta"}, anchors=set())
    assert r.G.has_edge("alpha", "beta")
    data = r.G["alpha"]["beta"]
    assert data["count"] == n
    assert len(data["docs"]) == 200, len(data["docs"])


# --- prune (періодичний sweep густоти) --------------------------------------
def test_prune_removes_weak_edges_keeps_anchor_edges_and_cleans_isolated_nodes():
    """prune(): ребра з count<min_edge_count АБО npmi<0 видаляються (крім
    інцидентних якорю); вузли з doc_freq<min_doc_freq видаляються (крім
    якірних); ізольовані після чистки не-якірні вузли теж прибираються;
    n_docs/seen_docs (корпус) не чіпаються."""
    r = SemanticReasoner()
    G = r.G
    G.graph["n_docs"] = 100

    # Якірне ребро з малим count має вижити (інцидентне якорю — факти не ріжемо).
    G.add_node("CVE-2024-1111", kind="cve", doc_freq=1, is_anchor=True, boost=2.0)
    G.add_node("rareterm", kind="term", doc_freq=2, is_anchor=False, boost=1.0)
    G.add_edge("CVE-2024-1111", "rareterm", count=1, docs={"d1"})

    # Слабке неякірне ребро (count<2) з низькою doc_freq -> ребро й обидва
    # вузли видаляються.
    G.add_node("weakA", kind="term", doc_freq=1, is_anchor=False, boost=1.0)
    G.add_node("weakB", kind="term", doc_freq=1, is_anchor=False, boost=1.0)
    G.add_edge("weakA", "weakB", count=1, docs={"d2"})

    # count>=2, але npmi<0 (антикорельовані частотні терміни) -> ребро
    # видаляється; обидва вузли лишаються без ребер -> прибираються як
    # ізольовані (навіть попри doc_freq>=2).
    G.add_node("commonA", kind="term", doc_freq=40, is_anchor=False, boost=1.0)
    G.add_node("commonB", kind="term", doc_freq=40, is_anchor=False, boost=1.0)
    G.add_edge("commonA", "commonB", count=2, docs={"d3", "d4"})
    assert r.npmi("commonA", "commonB") < 0   # передумова тесту

    # Сильне ребро -> лишається як є.
    G.add_node("goodA", kind="term", doc_freq=5, is_anchor=False, boost=1.0)
    G.add_node("goodB", kind="term", doc_freq=5, is_anchor=False, boost=1.0)
    G.add_edge("goodA", "goodB", count=5, docs={"d5"})
    assert r.npmi("goodA", "goodB") >= 0      # передумова тесту

    stats = r.prune()

    assert stats == {"nodes_removed": 4, "edges_removed": 2}, stats
    assert set(r.G.nodes) == {"CVE-2024-1111", "rareterm", "goodA", "goodB"}
    assert r.G.has_edge("CVE-2024-1111", "rareterm")
    assert r.G.has_edge("goodA", "goodB")
    assert not r.G.has_edge("weakA", "weakB")
    assert not r.G.has_edge("commonA", "commonB")
    assert r.G.graph["n_docs"] == 100   # корпус (n_docs) не чіпаємо


def test_prune_keeps_anchor_neighbor_and_edge_despite_low_doc_freq():
    """РЕГРЕС (HIGH, знайдено code-review): крок 1 prune щадить ребра, що
    інцидентні якірному вузлу («факти не ріжемо»), але крок 2 раніше видаляв
    НЕ-якірний вузол лише за doc_freq<min_doc_freq, не перевіряючи, чи є в
    нього якірний сусід — `remove_nodes_from` каскадно зносив ЗАХИЩЕНЕ на
    кроці 1 якірне ребро разом із вузлом. Інваріант «факти не ріжемо» мусить
    зберігатись і для вузла, і для його якірного ребра.

    Також перевіряємо, що `edges_removed` відповідає РЕАЛЬНІЙ кількості
    видалених ребер (включно з каскадними при видаленні вузлів), а не лише
    тим, що явно зрізані на кроці 1."""
    r = SemanticReasoner()
    G = r.G
    G.graph["n_docs"] = 10

    # rareterm: не-якір, doc_freq=1 (< поріг 2), але має ЯКІРНОГО сусіда через
    # ребро з count=1 (< поріг 2) — ребро захищене на кроці 1 (інцидентне якорю).
    G.add_node("CVE-2024-2222", kind="cve", doc_freq=1, is_anchor=True, boost=2.0)
    G.add_node("rareterm", kind="term", doc_freq=1, is_anchor=False, boost=1.0)
    G.add_edge("CVE-2024-2222", "rareterm", count=1, docs={"d1"})

    edges_before = r.G.number_of_edges()
    stats = r.prune(min_doc_freq=2, min_edge_count=2, drop_negative_npmi=False)
    edges_after = r.G.number_of_edges()

    assert "rareterm" in r.G.nodes, "не-якірний вузол з якірним сусідом має вижити"
    assert r.G.has_edge("CVE-2024-2222", "rareterm"), \
        "якірне ребро не можна знести каскадно через видалення сусіднього вузла"
    assert stats["edges_removed"] == edges_before - edges_after == 0, stats
    assert stats["nodes_removed"] == 0, stats


# --- edge docs-set cap: симетрія з cap термінів ----------------------------
def test_edge_docs_set_uncapped_when_env_non_positive():
    """REASONER_MAX_DOCS_PER_EDGE<=0 — без ліміту (симетрично до cap=0 для
    термінів: стара поведінка, зворотна сумісність)."""
    old = os.environ.get("REASONER_MAX_DOCS_PER_EDGE")
    os.environ["REASONER_MAX_DOCS_PER_EDGE"] = "0"
    try:
        r = SemanticReasoner()
        n = 250
        for i in range(n):
            r.upsert_document(f"doc{i}", concepts={"alpha", "beta"}, anchors=set())
    finally:
        if old is None:
            os.environ.pop("REASONER_MAX_DOCS_PER_EDGE", None)
        else:
            os.environ["REASONER_MAX_DOCS_PER_EDGE"] = old
    data = r.G["alpha"]["beta"]
    assert data["count"] == n
    assert len(data["docs"]) == n, len(data["docs"])


# --- ідемпотентність порожнього документа ----------------------------------
def test_upsert_document_empty_then_repeat_is_idempotent():
    """Порожній документ (без концептів) рахується як «бачений» без ребер, але
    n_docs інкрементується РІВНО раз; повторний upsert з тим самим doc_id —
    no-op (ідемпотентність за doc_id), n_docs не подвоюється."""
    r = SemanticReasoner()
    record = {"title": "", "summary": "", "verified_cve_ids": "", "verified_ttp_ids": ""}
    added1 = r.upsert_document("empty1", record=record)
    assert added1 is False
    assert r.G.graph["n_docs"] == 1

    added2 = r.upsert_document("empty1", record=record)
    assert added2 is False
    assert r.G.graph["n_docs"] == 1, "повторний upsert того ж doc_id не має подвоювати n_docs"


# --- hypotheses: скоринг за СЕРЕДНЬОЮ вагою ребра (не за сумою costs) -----
def test_hypotheses_multi_hop_strong_chain_outranks_weak_single_hop():
    """Проблема старого скорингу (score=1/cost, cost=сума 1/w по ребрах):
    багатохопові ланцюги штрафуються за довжину, навіть якщо ребра сильні.

    Приклад: 2-хопий ланцюг A-M-Z із двох ребер ваги 10 (mean=10) має бути
    ранжований ВИЩЕ за 1-хопий ланцюг W-Z зі слабшого ребра ваги 6 (mean=6) —
    хоча сирий dijkstra-cost слабшого ребра (1/6≈0.167) МЕНШИЙ за сумарний
    cost сильного ланцюга (1/10+1/10=0.2), тобто стара сортировка «по cost
    ASC» помилково ставила б W-Z першим."""
    r = _reasoner_with_edges([
        ("A", "M", 10), ("M", "Z", 10),
        ("W", "Z", 6),
    ])
    hyps = r.hypotheses("Z", k=5, weight_mode="raw")
    starts = [h["start"] for h in hyps]
    assert starts.index("A") < starts.index("W"), starts


def test_hypotheses_score_equals_total_weight_for_single_hop():
    """Для 1-хопового ланцюга mean_weight == total_weight (hops=1) — нове поле
    score має зберігати цю властивість (сумісність зі старим сенсом «вага
    єдиного ребра»)."""
    r = _reasoner_with_edges([("A", "Z", 7)])
    hyps = r.hypotheses("Z", k=5, weight_mode="raw")
    assert len(hyps) == 1
    h = hyps[0]
    assert h["hops"] == 1
    assert h["score"] == h["total_weight"]


def test_hypotheses_anchor_ranks_before_stronger_non_anchor():
    """Якірні (верифіковані) гіпотези йдуть ПЕРШИМИ незалежно від score —
    навіть якщо в неякірної гіпотези середня вага ребра вища."""
    r = _reasoner_with_edges([
        ("CVE-2024-0001", "Z", 2),   # якір, слабке ребро
        ("term1", "Z", 50),          # неякір, набагато сильніше ребро
    ])
    hyps = r.hypotheses("Z", k=5, weight_mode="raw")
    assert hyps[0]["start"] == "CVE-2024-0001"
    assert hyps[0]["is_anchor"] is True
    assert hyps[1]["start"] == "term1"


def test_hypotheses_tiebreak_prefers_fewer_hops_on_equal_score():
    """При однаковому score (mean_weight) тай-брейк — менша кількість хопів
    раніше (P: 1 хоп, вага 5; A2-M2: 2 хопи, обидва ребра по 5 → той самий
    mean=5, але P має менше хопів)."""
    r = _reasoner_with_edges([
        ("P", "Z", 5),
        ("A2", "M2", 5), ("M2", "Z", 5),
    ])
    hyps = r.hypotheses("Z", k=5, weight_mode="raw")
    p = next(h for h in hyps if h["start"] == "P")
    a2 = next(h for h in hyps if h["start"] == "A2")
    assert p["score"] == a2["score"] == 5.0
    starts = [h["start"] for h in hyps]
    assert starts.index("P") < starts.index("A2")
    assert p["hops"] < a2["hops"]



# --- resolve_node: ввід користувача → ідентифікатор вузла -----------------
def test_resolve_node_uppercases_cve():
    r = _reasoner_with_edges([("CVE-2024-1234", "ransomware", 3)])
    assert r.resolve_node("cve-2024-1234") == "CVE-2024-1234"


def test_resolve_node_prefixes_bare_technique_id():
    r = _reasoner_with_edges([("MITRE T1566", "ransomware", 3)])
    assert r.resolve_node("t1566") == "MITRE T1566"
    assert r.resolve_node("mitre T1566") == "MITRE T1566"


def test_resolve_node_falls_back_to_lowercase_term():
    r = _reasoner_with_edges([("phishing", "ransomware", 3)])
    assert r.resolve_node("Phishing") == "phishing"


def test_resolve_node_returns_none_for_unknown():
    r = _reasoner_with_edges([("phishing", "ransomware", 3)])
    assert r.resolve_node("нічого-такого") is None
    assert r.resolve_node("") is None


# --- hypotheses_from: двофронтовий пошук у бойовій формі ------------------
def _two_route_graph():
    """Ціль Z. Сильний маршрут веде з B, слабкий — з A.

        A -1- P -1- Z          (слабкий, але КОРОТКИЙ від A)
        B -9- M -9- Z          (сильний)
        A -1- B                (щоб A і B були в одній компоненті)
    """
    return _reasoner_with_edges([
        ("A", "P", 1), ("P", "Z", 1),
        ("B", "M", 9), ("M", "Z", 9),
        ("A", "B", 1),
    ])


def test_hypotheses_from_starts_only_at_requested_nodes():
    r = _two_route_graph()
    hyps = r.hypotheses_from(["A"], "Z", weight_mode="raw")
    assert hyps, "мав знайти щонайменше один ланцюг"
    assert all(h["start"] == "A" for h in hyps), [h["chain"] for h in hyps]
    assert all(h["chain"][-1] == "Z" for h in hyps)


def test_hypotheses_from_returns_one_chain_per_start():
    r = _two_route_graph()
    hyps = r.hypotheses_from(["A", "B"], "Z", weight_mode="raw")
    assert {h["start"] for h in hyps} == {"A", "B"}


def test_hypotheses_from_ranks_stronger_chain_first():
    r = _two_route_graph()
    hyps = r.hypotheses_from(["A", "B"], "Z", weight_mode="raw")
    assert hyps[0]["start"] == "B", [(h["start"], h["score"]) for h in hyps]
    assert hyps[0]["rank"] == 1
    assert hyps[1]["rank"] == 2


def test_hypotheses_from_skips_unknown_starts_without_failing():
    r = _two_route_graph()
    hyps = r.hypotheses_from(["A", "неіснуючий-вузол"], "Z", weight_mode="raw")
    assert [h["start"] for h in hyps] == ["A"]


def test_hypotheses_from_empty_when_goal_missing():
    r = _two_route_graph()
    assert r.hypotheses_from(["A"], "нема-такої-цілі") == []


def test_hypotheses_from_empty_when_no_start_resolves():
    r = _two_route_graph()
    assert r.hypotheses_from(["зовсім-нема"], "Z") == []


# --- КОНТРОЛЬНА ГІЛКА: той самий режим, але односторонній обхід ----------
# Навіщо. В експерименті двофронтовий режим (`hypotheses_from`) порівнювався з
# `hypotheses()`, але ті відрізняються ДВОМА речами: примітивом обходу І тим,
# що старти задає аналітик, а не обирає алгоритм. Прапорець `bidirectional`
# лишає ОДНУ відмінність — сам обхід, — щоб можна було виміряти внесок кожного
# фактора окремо. Це ablation, а не режим для UI.

def test_unidirectional_mode_finds_the_same_optimal_chain():
    """Обидва примітиви шукають найдешевший шлях між тими самими вузлами.

    На неорієнтованому графі двофронтовий Дейкстра — це оптимізація обходу, а
    не інший критерій оптимальності, тож ланцюг має збігатися."""
    r = _two_route_graph()
    bi = r.hypotheses_from(["B"], "Z", weight_mode="raw")
    uni = r.hypotheses_from(["B"], "Z", weight_mode="raw", bidirectional=False)
    assert bi and uni
    assert uni[0]["chain"] == bi[0]["chain"]
    assert uni[0]["total_weight"] == bi[0]["total_weight"]


def test_unidirectional_mode_keeps_the_same_result_shape():
    r = _two_route_graph()
    bi = r.hypotheses_from(["A", "B"], "Z", weight_mode="raw")
    uni = r.hypotheses_from(["A", "B"], "Z", weight_mode="raw", bidirectional=False)
    assert [h["start"] for h in uni] == [h["start"] for h in bi]
    assert set(uni[0]) == set(bi[0]), "поля результату мають збігатися"
    assert [h["rank"] for h in uni] == [1, 2]


def test_unidirectional_mode_still_starts_only_at_requested_nodes():
    r = _two_route_graph()
    uni = r.hypotheses_from(["A"], "Z", weight_mode="raw", bidirectional=False)
    assert uni and all(h["start"] == "A" for h in uni)


def test_unidirectional_mode_respects_the_hop_limit():
    r = _reasoner_with_edges([("A", "B", 5), ("B", "C", 5), ("C", "Z", 5)])
    assert r.hypotheses_from(["A"], "Z", weight_mode="raw",
                             max_hops=2, bidirectional=False) == []
    assert r.hypotheses_from(["A"], "Z", weight_mode="raw",
                             max_hops=3, bidirectional=False)


def test_unidirectional_mode_respects_k():
    r = _two_route_graph()
    uni = r.hypotheses_from(["A", "B"], "Z", weight_mode="raw", k=1,
                            bidirectional=False)
    assert len(uni) == 1


def test_unidirectional_mode_empty_when_goal_unreachable():
    r = _reasoner_with_edges([("A", "B", 3), ("Y", "Z", 3)])
    assert r.hypotheses_from(["A"], "Z", weight_mode="raw",
                             bidirectional=False) == []


def test_hypotheses_from_accepts_raw_user_input():
    """Старт із чату приходить у довільному регістрі — resolve_node всередині."""
    r = _reasoner_with_edges([("CVE-2024-1234", "M", 5), ("M", "ransomware", 5)])
    hyps = r.hypotheses_from(["cve-2024-1234"], "ransomware", weight_mode="raw")
    assert hyps and hyps[0]["start"] == "CVE-2024-1234"


def test_hypotheses_from_shape_matches_hypotheses():
    """UI малює обидва результати тим самим кодом — ключі мають збігатися."""
    r = _two_route_graph()
    a = r.hypotheses("Z", k=1, weight_mode="raw")
    b = r.hypotheses_from(["B"], "Z", weight_mode="raw")
    assert a and b
    assert set(a[0]) == set(b[0]), set(a[0]) ^ set(b[0])


def test_hypotheses_from_deduplicates_same_start_written_differently():
    """Той самий концепт, набраний по-різному, — це ОДИН старт, не два.
    Інакше користувач отримав би дубльований ланцюг у видачі."""
    r = _reasoner_with_edges([("CVE-2024-1234", "M", 5), ("M", "ransomware", 5)])
    hyps = r.hypotheses_from(["cve-2024-1234", "CVE-2024-1234"], "ransomware",
                             weight_mode="raw")
    assert len(hyps) == 1, [h["chain"] for h in hyps]


def test_hypotheses_from_ignores_start_equal_to_goal():
    r = _reasoner_with_edges([("A", "Z", 5)])
    assert r.hypotheses_from(["Z"], "Z", weight_mode="raw") == []

# --- ручний запуск без pytest --------------------------------------------
if __name__ == "__main__":
    import tempfile
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                import pathlib
                with tempfile.TemporaryDirectory() as td:
                    fn(pathlib.Path(td))
            else:
                fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
