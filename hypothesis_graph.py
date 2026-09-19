"""
Чистий модуль: будує networkx.DiGraph із ланцюгів гіпотез
`semantic_reasoner.SemanticReasoner.hypotheses()` — для подальшої
візуалізації (pyvis/streamlit у ui.py).

За образом `hypothesis_labels.py`: жодних залежностей на streamlit/pyvis,
лише networkx (уже тягнеться semantic_reasoner.py) — щоб можна було
юніт-тестувати без підйому UI.
"""

from __future__ import annotations

import networkx as nx

# Фіктивний вузол-початок (у корпусі його НЕМА).
VIRTUAL_START_NODE = "Теперішній стан"


def _annotate_label(node: str, label_map: dict | None) -> str:
    """"id (Назва)", якщо запис є в `label_map`, інакше — сам id незмінним
    (той самий стиль анотації, що й `hypothesis_labels.annotate_chain_for_display`
    та `semantic_reasoner.format_hypotheses_for_llm._annotate`)."""
    if label_map:
        label = label_map.get(node)
        if label:
            return f"{node} ({label})"
    return node


def _node_kind(node: str, goal_concept: str) -> str:
    """Класифікує вузол: "goal" для цільового вузла (перекриває все інше),
    інакше за префіксом ідентифікатора ("cve" / "mitre" / "term")."""
    if node == goal_concept:
        return "goal"
    if node.startswith("CVE-"):
        return "cve"
    if node.startswith("MITRE "):
        return "mitre"
    return "term"


def build_hypothesis_graph(hyps: list[dict], goal_concept: str,
                           label_map: dict | None = None,
                           start_node: str | None = VIRTUAL_START_NODE) -> nx.DiGraph:
    """Будує орієнтований граф ланцюгів гіпотез (start → … → goal).

    `hyps` — список гіпотез у форматі `SemanticReasoner.hypotheses()`
    (потрібні лише ключі "chain", "edges", "is_anchor"). Ребра йдуть УЗДОВЖ
    ланцюга, тобто в напрямку до цілі. Спільні вузли/ребра між різними
    ланцюгами не дублюються (властивість DiGraph) — при конфлікті ваги
    того самого ребра з різних ланцюгів лишається МАКСИМАЛЬНА (найсильніший
    із засвідчених ланцюгом звʼязків).

    `start_node` — імʼя ФІКТИВНОГО вузла-початку (деф. «Теперішній стан»), з
    якого проводяться ребра до початку КОЖНОГО ланцюга. Без нього обхід від
    цілі назад дає N незалежних стартів — на екрані це віяло «5 початків → 1
    кінець», а не ланцюг міркування. Вузол службовий: kind="start",
    virtual=True, ребра з нього virtual=True і weight=0.0, щоб їх не сплутати
    з реальними звʼязками корпусу. `start_node=None` вимикає надбудову."""
    G = nx.DiGraph()
    G.add_node(goal_concept, kind="goal",
               label=_annotate_label(goal_concept, label_map), is_anchor=False)

    chain_starts = []
    for h in hyps:
        chain = h.get("chain") or []
        edges = h.get("edges") or []
        is_anchor = bool(h.get("is_anchor"))
        start = chain[0] if chain else None
        if start is not None and start not in chain_starts:
            chain_starts.append(start)

        for node in chain:
            if node not in G:
                G.add_node(node, kind=_node_kind(node, goal_concept),
                           label=_annotate_label(node, label_map), is_anchor=False)
            if is_anchor and node == start:
                G.nodes[node]["is_anchor"] = True

        for a, b, w in edges:
            if G.has_edge(a, b):
                if w > G[a][b]["weight"]:
                    G[a][b]["weight"] = w
            else:
                G.add_edge(a, b, weight=w)

    # Фіктивний початок додаємо ОСТАННІМ і лише за наявності ланцюгів: без них
    # починати нема чого, а порожній граф із самим службовим вузлом ввів би в
    # оману («є початок, але доказів нема»).
    if start_node and chain_starts:
        G.add_node(start_node, kind="start", label=start_node,
                   is_anchor=False, virtual=True)
        for start in chain_starts:
            if start != start_node:
                G.add_edge(start_node, start, weight=0.0, virtual=True)

    return G
