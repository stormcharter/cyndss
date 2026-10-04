"""Двостороння семантична мережа: від проблеми до розв'язку і від розв'язку до проблеми.

Метод (пункти 4-6):

    4  Мережа понять росте з ДВОХ фронтів. Прямий фронт іде від проблеми:
       «що з цього випливає, що можна зробити далі». Зворотний — від розв'язку:
       «що має бути зроблено безпосередньо перед цим». Кожне розкриття — виклик
       мовної моделі з контекстом фізичної мережі (GRAG) і, якщо є корпус,
       фрагментами джерел (RAG).
    5  Де фронти зустрічаються, з'являються МОСТИ: пара «поняття прямого
       фронту — поняття зворотного», яка веде одне до одного. Кандидатів
       відбирає смислова близькість (ембединги), а підтверджує модель одним
       пакетним питанням. Кожен міст дає ланцюг «проблема → … → міст → … →
       розв'язок».
    6  Кожен крок ланцюга модель оцінює за ціною, часом і впливом (1-5),
       ланцюг агрегується, ранжування — інтегральна J (`qa.twin.aggregate`).

Модуль не знає про HTTP: модель і ембедер передаються функціями
`generate(system, user, tag) -> str` і `embed(texts) -> list[vector]`,
тож усе тестується без мережі.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

import networkx as nx

from prompt_guard import ISOLATION_INSTRUCTION_UK, wrap_untrusted
from qa.twin import TwinCriterion, aggregate

Generate = Callable[[str, str, str], str]
Embed = Callable[[list[str]], list[list[float]]]
Retrieve = Callable[[str], list[str]]

PROBLEM_ID = "P"
SOLUTION_ID = "S"

#: Поняття одного фронту з такою близькістю вважаються тим самим поняттям.
#: Поріг високий: англійські ембединги nomic дають 0.93-0.94 навіть різним
#: діям над тими самими вузлами («інвентаризація вразливостей на web-01» і
#: «усунення вразливостей на web-01») — назви вузлів тягнуть близькість угору.
DUPLICATE_COS = 0.96
#: Міст із такою близькістю приймається без питання до моделі: це те саме
#: поняття, назване з двох боків. Усе нижче підтверджує модель.
AUTO_BRIDGE_COS = 0.96
#: Скільки найближчих пар віддавати моделі на підтвердження одним викликом.
BRIDGE_CANDIDATES = 20
#: Внесок кроку з впливом s (1..5) у зниження проблеми: 1 - Π(1 - IMPACT_UNIT·s).
IMPACT_UNIT = 0.15
MAX_CHAINS = 200


@dataclass
class Concept:
    id: str
    label: str            # українською — те, що бачить людина
    en: str               # англійською — для ембедингів (nomic не розрізняє український зміст)
    side: str             # problem | solution | forward | backward
    depth: int
    parent: str = ""


@dataclass
class SemanticNet:
    problem: str
    solution: str
    concepts: dict[str, Concept] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)   # (from, to, relation)
    bridges: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def side(self, name: str) -> list[Concept]:
        return [c for c in self.concepts.values() if c.side == name]

    def graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.concepts)
        for src, dst, relation in self.edges:
            g.add_edge(src, dst, relation=relation)
        return g

    def to_dict(self) -> dict:
        return {"problem": self.problem, "solution": self.solution,
                "concepts": [asdict(c) for c in self.concepts.values()],
                "edges": [list(e) for e in self.edges],
                "bridges": self.bridges, "log": self.log}

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticNet":
        net = cls(problem=data["problem"], solution=data["solution"])
        for raw in data.get("concepts", []):
            concept = Concept(**raw)
            net.concepts[concept.id] = concept
        net.edges = [tuple(e) for e in data.get("edges", [])]
        net.bridges = list(data.get("bridges", []))
        net.log = list(data.get("log", []))
        return net


# =========================================================================
#  Допоміжне
# =========================================================================

def extract_json(text: str) -> dict:
    """Перший збалансований JSON-об'єкт у відповіді моделі (моделі люблять
    обгортати його в ```json або дописувати пояснення)."""
    cleaned = re.sub(r"```(?:json)?", "", text or "")
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(cleaned[start:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = cleaned.find("{", start + 1)
    raise ValueError("у відповіді немає JSON-об'єкта")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _clean_label(text: object, limit: int = 120) -> str:
    return " ".join(str(text or "").split())[:limit].strip(" .;")


def _parse_concepts(raw: str) -> list[tuple[str, str]]:
    """{"concepts": [{"uk": ..., "en": ...}]} -> [(uk, en)]; порожні відкидаються."""
    data = extract_json(raw)
    items = data.get("concepts")
    if not isinstance(items, list):
        raise ValueError("немає списку concepts")
    out = []
    for item in items:
        if isinstance(item, dict):
            uk, en = _clean_label(item.get("uk")), _clean_label(item.get("en"))
        else:
            uk = en = _clean_label(item)
        if uk:
            out.append((uk, en or uk))
    return out


# =========================================================================
#  Пункт 4: розкриття двох фронтів
# =========================================================================

COMMON_RULES = (
    "Кожне поняття — конкретна дія або проміжний стан, 3-9 слів, про ЦЮ мережу "
    "(називай вузли, сегменти, засоби захисту, якщо доречно). Не повторюй уже "
    "наявні поняття і не переписуй їх іншими словами. Відповідай СТРОГО одним "
    "JSON-об'єктом: {\"concepts\": [{\"uk\": \"поняття українською\", "
    "\"en\": \"the same concept in English\"}]}"
)

FORWARD_SYSTEM = (
    "Ти будуєш семантичну мережу понять від ПРОБЛЕМИ до РОЗВ'ЯЗКУ в кібербезпеці. "
    "Тобі дають поточне поняття на шляху від проблеми. Запропонуй {k} різних "
    "наступних понять, які БЕЗПОСЕРЕДНЬО випливають із поточного і наближають до "
    "розв'язку. Різні поняття — різні напрями, а не варіації одного. " + COMMON_RULES
)

BACKWARD_SYSTEM = (
    "Ти будуєш семантичну мережу понять від РОЗВ'ЯЗКУ назад до ПРОБЛЕМИ в "
    "кібербезпеці. Тобі дають поняття на шляху до розв'язку. Назви {k} різних "
    "передумов — що має бути зроблено або досягнуто БЕЗПОСЕРЕДНЬО ПЕРЕД цим "
    "поняттям. Різні передумови — різні напрями, а не варіації однієї. " + COMMON_RULES
)


def _path_labels(net: SemanticNet, concept: Concept) -> list[str]:
    labels, node = [], concept
    while node:
        labels.append(node.label)
        node = net.concepts.get(node.parent) if node.parent else None
    return labels


def _expand(net: SemanticNet, concept: Concept, direction: str, k: int, context: str,
            generate: Generate, retrieve: Retrieve | None) -> list[tuple[str, str]]:
    forward = direction == "forward"
    system = (FORWARD_SYSTEM if forward else BACKWARD_SYSTEM).replace("{k}", str(k))
    chain = _path_labels(net, concept)
    route = " ← ".join(chain) if forward else " → ".join(chain)
    existing = [c.label for c in net.side(direction)] + [net.problem, net.solution]
    origin = "від проблеми" if forward else "від розв'язку"
    user = (f"Контекст мережі:\n{context}\n\nПРОБЛЕМА: {net.problem}\nРОЗВ'ЯЗОК: {net.solution}\n\n"
            f"Поточне поняття: {concept.label}\n"
            f"Шлях до нього ({origin}): {route}\n"
            f"Уже наявні поняття (не повторювати): {'; '.join(existing)}")
    if retrieve:
        snippets = retrieve(concept.en or concept.label)
        if snippets:
            system += "\n\n" + ISOLATION_INSTRUCTION_UK
            user += "\n\nФрагменти відкритих джерел (для опори):\n" + "\n".join(
                wrap_untrusted(s, max_len=500) for s in snippets)
    return _parse_concepts(generate(system, user, f"{direction}:{concept.id}"))[:k]


def _add_concepts(net: SemanticNet, parent: Concept, items: list[tuple[str, str]],
                  direction: str, embed: Embed, vectors: dict[str, list[float]]) -> list[Concept]:
    """Додати поняття фронту; дубль (за текстом або близькістю) не створює вузла,
    а лише нове ребро до вже наявного."""
    created = []
    if not items:
        return created
    new_vectors = embed([f"clustering: {en}" for _uk, en in items])
    same_side = [c for c in net.side(direction) if c.id in vectors]
    for (uk, en), vec in zip(items, new_vectors):
        twin = next((c for c in same_side if c.label.lower() == uk.lower()
                     or cosine(vectors[c.id], vec) >= DUPLICATE_COS), None)
        target = twin
        if twin is None:
            prefix = "f" if direction == "forward" else "b"
            cid = f"{prefix}{sum(1 for c in net.concepts.values() if c.side == direction) + 1}"
            target = Concept(cid, uk, en, direction, parent.depth + 1, parent.id)
            net.concepts[cid] = target
            vectors[cid] = vec
            same_side.append(target)
            created.append(target)
        edge = (parent.id, target.id, "веде до") if direction == "forward" \
            else (target.id, parent.id, "передує")
        if edge not in net.edges and edge[0] != edge[1]:
            net.edges.append(edge)
    return created


def build_semantic_net(problem: str, solution: str, context: str, generate: Generate,
                       embed: Embed, k: int = 3, depth: int = 2,
                       retrieve: Retrieve | None = None,
                       progress: Callable[[str], None] | None = None) -> SemanticNet:
    """Розкрити обидва фронти на `depth` кроків по `k` понять і знайти мости."""
    net = SemanticNet(problem=problem.strip(), solution=solution.strip())
    p = Concept(PROBLEM_ID, net.problem, net.problem, "problem", 0)
    s = Concept(SOLUTION_ID, net.solution, net.solution, "solution", 0)
    net.concepts = {p.id: p, s.id: s}
    vectors: dict[str, list[float]] = {}
    frontiers = {"forward": [p], "backward": [s]}
    for level in range(1, depth + 1):
        for direction in ("forward", "backward"):
            fresh = []
            for concept in frontiers[direction]:
                if progress:
                    progress(f"{'Прямий' if direction == 'forward' else 'Зворотний'} фронт, "
                             f"крок {level}: «{concept.label[:60]}»")
                try:
                    items = _expand(net, concept, direction, k, context, generate, retrieve)
                except ValueError as exc:
                    net.log.append(f"{direction} {concept.id}: {exc}")
                    continue
                fresh += _add_concepts(net, concept, items, direction, embed, vectors)
            frontiers[direction] = fresh
    if progress:
        progress("Шукаю мости між фронтами…")
    find_bridges(net, generate, embed, vectors)
    return net


# =========================================================================
#  Пункт 5: мости між фронтами й ланцюги
# =========================================================================

BRIDGE_SYSTEM = (
    "Перевір зв'язки в семантичній мережі розв'язання проблеми кібербезпеки. Ліве "
    "поняття — крок від проблеми, праве — крок, що веде до розв'язку. Для кожної "
    "пари «A → B» скажи, чи виконання A веде до B, є його частиною або необхідною "
    "передумовою (або це те саме поняття іншими словами). Часткове покриття "
    "(A робить B для частини вузлів) — ТАК. Просто спільна тема без такого зв'язку "
    "— НІ. Відповідай СТРОГО одним JSON-об'єктом: "
    "{\"links\": [{\"i\": 1, \"ok\": true, \"why\": \"коротко\"}]}"
)

#: Скільки пакетів кандидатів перевіряти, поки не знайдеться хоч один міст.
BRIDGE_BATCHES = 2


def _confirm_links(net: SemanticNet, ask: list, generate: Generate) -> list[dict]:
    """Один пакет пар -> підтверджені моделлю мости."""
    listing = "\n".join(f"{i}. {a.label} → {b.label}" for i, (_c, a, b) in enumerate(ask, 1))
    try:
        data = extract_json(generate(BRIDGE_SYSTEM, f"ПРОБЛЕМА: {net.problem}\n"
                                     f"РОЗВ'ЯЗОК: {net.solution}\n\nПари:\n{listing}",
                                     "bridges"))
    except ValueError as exc:
        net.log.append(f"bridges: {exc}")
        return []
    found = []
    for item in data.get("links") or []:
        if not isinstance(item, dict) or item.get("ok") is not True:
            continue
        try:
            cos, a, b = ask[int(item.get("i")) - 1]
        except (TypeError, ValueError, IndexError):
            continue
        found.append({"from": a.id, "to": b.id, "cos": round(cos, 3),
                      "kind": "link", "why": _clean_label(item.get("why"), 200)})
    return found


def find_bridges(net: SemanticNet, generate: Generate, embed: Embed,
                 vectors: dict[str, list[float]] | None = None,
                 candidates: int = BRIDGE_CANDIDATES) -> list[dict]:
    """Пари «прямий фронт -> зворотний фронт», що з'єднують проблему з розв'язком."""
    vectors = dict(vectors or {})
    left = [c for c in net.concepts.values() if c.side == "forward"]
    right = [c for c in net.concepts.values() if c.side == "backward"]
    missing = [c for c in left + right if c.id not in vectors]
    if missing:
        for concept, vec in zip(missing, embed([f"clustering: {c.en}" for c in missing])):
            vectors[concept.id] = vec
    ranked = sorted(((cosine(vectors[a.id], vectors[b.id]), a, b) for a in left for b in right),
                    key=lambda t: (-t[0], t[1].id, t[2].id))
    bridges = [{"from": a.id, "to": b.id, "cos": round(cos, 3), "kind": "same",
                "why": "те саме поняття з двох боків"}
               for cos, a, b in ranked if cos >= AUTO_BRIDGE_COS]
    rest = [t for t in ranked if t[0] < AUTO_BRIDGE_COS]
    for batch in range(BRIDGE_BATCHES):
        ask = rest[batch * candidates:(batch + 1) * candidates]
        if not ask:
            break
        found = _confirm_links(net, ask, generate)
        bridges += found
        if found or bridges:
            break
    seen = set()
    net.bridges = []
    for bridge in bridges:
        key = (bridge["from"], bridge["to"])
        if key not in seen:
            seen.add(key)
            net.bridges.append(bridge)
    net.edges = [e for e in net.edges if e[2] not in {"міст", "те саме"}]
    for bridge in net.bridges:
        net.edges.append((bridge["from"], bridge["to"],
                          "те саме" if bridge["kind"] == "same" else "міст"))
    return net.bridges


def chains(net: SemanticNet, limit: int = MAX_CHAINS) -> list[list[str]]:
    """Усі прості шляхи проблема -> розв'язок, що проходять через міст."""
    g = net.graph()
    if PROBLEM_ID not in g or SOLUTION_ID not in g:
        return []
    found = []
    for path in nx.all_simple_paths(g, PROBLEM_ID, SOLUTION_ID):
        found.append(path)
        if len(found) >= limit:
            break
    return sorted(found, key=lambda p: (len(p), p))


def chain_steps(net: SemanticNet, path: list[str]) -> list[str]:
    """Кроки ланцюга без проблеми й розв'язку; другий кінець мосту «те саме»
    не дублюється як окремий крок."""
    same = {(b["from"], b["to"]) for b in net.bridges if b["kind"] == "same"}
    steps = []
    for prev, node in zip(path, path[1:]):
        if node in {PROBLEM_ID, SOLUTION_ID} or (prev, node) in same:
            continue
        steps.append(node)
    return steps


# =========================================================================
#  Пункт 6: оцінка кроків моделлю й ранжування
# =========================================================================

SCORE_SYSTEM = (
    "Оціни кожен крок плану з кібербезпеки за трьома шкалами від 1 до 5: "
    "cost — ціна (1 дуже дешево, 5 дуже дорого), time — час (1 дні, 5 багато "
    "місяців), impact — вплив на розв'язання проблеми (1 майже ніякого, 5 "
    "вирішальний). Враховуй контекст мережі. Відповідай СТРОГО одним JSON-об'єктом: "
    "{\"scores\": [{\"i\": 1, \"cost\": 3, \"time\": 2, \"impact\": 4}]}"
)


def _scale(value) -> int | None:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return min(max(v, 1), 5)


def score_steps(net: SemanticNet, node_ids: list[str], context: str,
                generate: Generate) -> dict[str, dict]:
    """{concept_id: {"cost", "time", "impact", "assumed"}} — одним викликом моделі.
    Крок, який модель не оцінила, отримує середні 3 з позначкою assumed."""
    ids = list(dict.fromkeys(node_ids))
    scores = {cid: {"cost": 3, "time": 3, "impact": 3, "assumed": True} for cid in ids}
    if not ids:
        return scores
    listing = "\n".join(f"{i}. {net.concepts[cid].label}" for i, cid in enumerate(ids, 1))
    try:
        data = extract_json(generate(SCORE_SYSTEM, f"Контекст мережі:\n{context}\n\n"
                                     f"ПРОБЛЕМА: {net.problem}\n\nКроки:\n{listing}", "score"))
    except ValueError as exc:
        net.log.append(f"score: {exc}")
        return scores
    for item in data.get("scores") or []:
        if not isinstance(item, dict):
            continue
        try:
            cid = ids[int(item.get("i")) - 1]
        except (TypeError, ValueError, IndexError):
            continue
        values = {k: _scale(item.get(k)) for k in ("cost", "time", "impact")}
        if all(v is not None for v in values.values()):
            scores[cid] = {**values, "assumed": False}
    return scores


def _criteria(weights: dict | None) -> list[TwinCriterion]:
    weights = {**{"cost": 1.0, "time": 1.0, "impact": 1.0}, **(weights or {})}
    return [
        TwinCriterion("cost", "ціна", max(float(weights["cost"]), 1e-6), "cost"),
        TwinCriterion("time", "час", max(float(weights["time"]), 1e-6), "cost"),
        TwinCriterion("impact", "вплив", max(float(weights["impact"]), 1e-6), "benefit"),
    ]


def rank_chains(net: SemanticNet, scores: dict[str, dict], weights: dict | None = None,
                k: int = 5, min_impact: float = 0.0) -> dict:
    """Ланцюги -> ранжований список із сумарною ціною, часом і впливом.

    Ціна й час — сума балів кроків (довший план дорожчий і довший), вплив —
    1 - Π(1 - IMPACT_UNIT·s): кожен крок додає, але зі спадною віддачею."""
    candidates, rejected, seen = [], 0, set()
    for path in chains(net):
        steps = chain_steps(net, path)
        if not steps:
            continue
        # Різні шляхи через міст «те саме» дають ті самі кроки — це один варіант,
        # а не два. Залишаємо найкоротший шлях (chains() відсортовані за довжиною).
        key = tuple(net.concepts[cid].label.strip().lower() for cid in steps)
        if key in seen:
            continue
        seen.add(key)
        s = [scores.get(cid, {"cost": 3, "time": 3, "impact": 3, "assumed": True}) for cid in steps]
        impact = 1.0
        for item in s:
            impact *= 1.0 - IMPACT_UNIT * item["impact"]
        impact = round(1.0 - impact, 4)
        if impact < min_impact:
            rejected += 1
            continue
        candidates.append({
            "path": path,
            "steps": [{"id": cid, "label": net.concepts[cid].label,
                       "side": net.concepts[cid].side, **scores_of}
                      for cid, scores_of in zip(steps, s)],
            "cost": sum(item["cost"] for item in s),
            "time": sum(item["time"] for item in s),
            "impact": impact,
            "assumed": any(item["assumed"] for item in s),
        })
    criteria = _criteria(weights)
    max_cost = max((c["cost"] for c in candidates), default=0) or 1
    max_time = max((c["time"] for c in candidates), default=0) or 1
    for chain in candidates:
        chain["scores"] = [
            {"criterion": "cost", "score": round(chain["cost"] / max_cost, 4)},
            {"criterion": "time", "score": round(chain["time"] / max_time, 4)},
            {"criterion": "impact", "score": chain["impact"]},
        ]
        chain["j"], _spread = aggregate(chain["scores"], criteria)
    candidates.sort(key=lambda c: (-c["j"], len(c["steps"]), tuple(c["path"])))
    ranked = []
    for rank, chain in enumerate(candidates[:k], 1):
        ranked.append({**chain, "rank": rank,
                       "labels": [net.concepts[n].label for n in chain["path"]]})
    return {"chains": ranked, "total": len(candidates) + rejected,
            "admissible": len(candidates), "rejected_impact": rejected}


def concept_overlap(a: SemanticNet, b: SemanticNet, embed: Embed,
                    threshold: float = DUPLICATE_COS) -> float:
    """Частка понять мережі `a`, що мають близького двійника в `b` (без кінців).
    Міра того, наскільки мережі для різних фізичних мереж повторюють одна одну."""
    left = [c for c in a.concepts.values() if c.side in {"forward", "backward"}]
    right = [c for c in b.concepts.values() if c.side in {"forward", "backward"}]
    if not left or not right:
        return 0.0
    va = embed([f"clustering: {c.en}" for c in left])
    vb = embed([f"clustering: {c.en}" for c in right])
    hits = sum(1 for x in va if max(cosine(x, y) for y in vb) >= threshold)
    return round(hits / len(left), 4)
