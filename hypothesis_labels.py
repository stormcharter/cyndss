"""
Чисті хелпери для анотації вузлів-технік MITRE у панелі гіпотез ONYX.

Винесені окремо від ui.py, щоб їх можна було юніт-тестувати БЕЗ підйому
streamlit (ui.py на верхньому рівні тягне `import streamlit as st` та інші
side-effects, тому напряму імпортувати його в тестах не можна).
"""


def build_mitre_label_map(mitre_names: dict) -> dict:
    """Мапить "голі" id технік MITRE (як у mitre_names.json, напр. "T1056")
    на ключі вузлів графа semantic_reasoner (з префіксом "MITRE ", напр.
    "MITRE T1056") → читабельна назва.

    Результат придатний як `label_map` для
    `semantic_reasoner.format_hypotheses_for_llm`."""
    return {f"MITRE {tech_id}": name for tech_id, name in mitre_names.items()}


def annotate_chain_for_display(chain: list, label_map: dict) -> list:
    """Повертає копію `chain`, де вузли з записом у `label_map` анотовано
    читабельною назвою у форматі "node_id (Назва)" — для екранного рендеру
    ланцюгів гіпотез. Вузли без запису лишаються без змін."""
    out = []
    for node in chain:
        label = label_map.get(node) if label_map else None
        out.append(f"{node} ({label})" if label else node)
    return out
