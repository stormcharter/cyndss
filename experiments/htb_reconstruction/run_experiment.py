"""Експеримент: чи відновлює обхід семантичної мережі ONYX відомий ланцюг атаки.

Постановка
----------
Кожна машина датасету — це модель інформаційної інфраструктури: компоненти,
ідентичності, права доступу, артефакти, вразливості. Плюс ПРИХОВАНИЙ
`attack_path` — фактичний ланцюг, яким її було скомпрометовано.

Модель розкладається на дрібні документи (див. `corpus.py`) і подається у той
самий `SemanticReasoner`, що працює у проді. Далі задається лише ЦІЛЬ (здобутий
на останньому кроці доступ, типово `root`/`system`) — і система має сама, з
нуля, дійти до неї назад. `attack_path` вмикається аж на етапі оцінювання.

Дві гілки порівнюються між собою:
  A. `hypotheses()`     — односторонній обхід від цілі назад. Це те, що зараз
                          викликає UI (`ui.py:1448`).
  B. `bidirectional_path()` — двофронтовий пошук: вперед від периметра і назад
                          від цілі. У коді є і покритий тестами, але в бойовому
                          шляху НЕ використовується. Саме цю гілку заявка
                          називає «двостороннім формуванням».

Порівняння A з B — і є перевірка того, чи заявлене «двостороннє» щось дає.

Відтворюваність
---------------
Жодного звернення до мережі, жодного виклику LLM, жодного стану поза цим
каталогом. Обхід графа детермінований, тож повторний запуск дає побайтово той
самий результат — це перевіряє `--check-determinism`.

Запуск:
    python experiments/htb_reconstruction/run_experiment.py
    python experiments/htb_reconstruction/run_experiment.py --split demo --check-determinism
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from semantic_reasoner import SemanticReasoner  # noqa: E402
from twin_bridge import (  # noqa: E402
    build_twin_model, path_concepts, rank_scenarios, transition_edge_ids,
)

# Модуль має працювати і як скрипт (`python experiments/.../run_experiment.py`),
# і як частина пакета (`from experiments.htb_reconstruction.run_experiment import …`
# у тестах). Звідси дві гілки імпорту.
try:  # запуск як пакет
    from experiments.htb_reconstruction.corpus import (  # noqa: E402
        build_records, entry_concepts, goal_concept, gold_chain, scoring_gold,
    )
    from experiments.htb_reconstruction.metric import (  # noqa: E402
        ordered_coverage, precision_against_gold, step_coverage,
    )
except ImportError:  # запуск як окремий скрипт
    sys.path.insert(0, str(HERE))
    from corpus import (  # noqa: E402
        build_records, entry_concepts, goal_concept, gold_chain, scoring_gold,
    )
    from metric import (  # noqa: E402
        ordered_coverage, precision_against_gold, step_coverage,
    )

# Детермінований шар методу: формули (8)-(11) — розширення підмереж, їх
# узгодження та перебір допустимих шляхів на ОРІЄНТОВАНОМУ типізованому графі.
from qa.twin import (  # noqa: E402
    enumerate_scenarios, expand_backward, expand_forward, reconcile,
)

def entity_precision(node_concepts, gold) -> float:
    """Частка СУТНОСТЕЙ шляху, дотичних до еталона.

    Токенна точність двох гілок непорівнянна: вузол неорієнтованого графа — це
    один токен, а вузол орієнтованого — сутність із кількох слів (у прогоні
    demo перший ланцюг гілки E має 8,65 токена проти 2,55 у гілки C). Ділити
    влучання на кількість токенів означає карати орієнтовану гілку за спосіб
    запису. Тут одиниця — сутність, як і має бути для графа сутностей."""
    if not node_concepts:
        return 0.0
    gold_tokens: set = set()
    for step in gold:
        gold_tokens |= set(step)
    if not gold_tokens:
        return 0.0
    hits = sum(1 for concepts in node_concepts if gold_tokens & set(concepts))
    return hits / len(node_concepts)


SPLITS = ("demo", "train", "validation")


def load_machines(split: str) -> list[dict]:
    """Читає моделі машин одного розділу. `manifest.json` — не машина."""
    folder = HERE / "dataset" / split
    machines = []
    for path in sorted(folder.glob("*.json")):
        if path.name == "manifest.json":
            continue
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        data["_split"] = split
        data["_file"] = path.name
        machines.append(data)
    return machines


def evaluate(machine: dict, k: int, weight_mode: str, max_hops: int) -> dict:
    records = build_records(machine)
    reasoner = SemanticReasoner().build_from_records(records)

    goal = goal_concept(machine)
    gold = scoring_gold(machine)
    row = {
        "split": machine.get("_split", ""),
        "machine": machine.get("machine", ""),
        "documents": len(records),
        "nodes": reasoner.G.number_of_nodes(),
        "edges": reasoner.G.number_of_edges(),
        "gold_steps": len(gold),
        "goal": goal or "",
        "goal_in_graph": bool(goal and reasoner.G.has_node(goal)),
    }

    empty = {
        "a_hops": 0, "a_coverage": 0.0, "a_ordered": 0.0, "a_precision": 0.0,
        "a_coverage_topk": 0.0, "a_chains": 0,
        "b_found": False, "b_hops": 0, "b_coverage": 0.0,
        "b_ordered": 0.0, "b_precision": 0.0,
        "c_chains": 0, "c_hops": 0, "c_coverage": 0.0, "c_ordered": 0.0,
        "c_precision": 0.0, "c_coverage_topk": 0.0,
        "d_chains": 0, "d_hops": 0, "d_coverage": 0.0, "d_ordered": 0.0,
        "d_precision": 0.0, "d_coverage_topk": 0.0,
        "e_chains": 0, "e_hops": 0, "e_coverage": 0.0, "e_ordered": 0.0,
        "e_precision": 0.0, "e_coverage_topk": 0.0, "e_goal_reachable": False,
        "e_nodes": 0, "e_precision_nodes": 0.0,
    }
    if not row["goal_in_graph"] or not gold:
        row.update(empty)
        return row

    # --- Гілка A: односторонній обхід від цілі назад (поточний прод) --------
    hyps = reasoner.hypotheses(goal, k=k, weight_mode=weight_mode, max_hops=max_hops)
    if hyps:
        top = hyps[0]["chain"]
        union: list[str] = []
        for h in hyps:
            for node in h["chain"]:
                if node not in union:
                    union.append(node)
        row.update({
            "a_chains": len(hyps),
            "a_hops": hyps[0]["hops"],
            "a_coverage": step_coverage(top, gold),
            "a_ordered": ordered_coverage(top, gold),
            "a_precision": precision_against_gold(top, gold),
            "a_coverage_topk": step_coverage(union, gold),
        })
    else:
        row.update({key: empty[key] for key in empty if key.startswith("a_")})

    # --- Гілка B: сирий двофронтовий примітив, ОДИН найкращий шлях ----------
    # Лишена для наступності з першою редакцією звіту: саме її числа там.
    starts = sorted(entry_concepts(machine) - {goal})
    bidir = reasoner.bidirectional_path(starts, goal, weight_mode=weight_mode) if starts else None
    if bidir:
        path = bidir["path"]
        row.update({
            "b_found": True,
            "b_hops": len(path) - 1,
            "b_coverage": step_coverage(path, gold),
            "b_ordered": ordered_coverage(path, gold),
            "b_precision": precision_against_gold(path, gold),
        })
    else:
        row.update({key: empty[key] for key in empty if key.startswith("b_")})

    # --- Гілка C: `hypotheses_from` — те, що ЗАРАЗ працює у проді -----------
    # Відмінність від B принципова: там один виклик із мультистартом і один
    # шлях на виході, тут — окремий двофронтовий пошук ВІД КОЖНОГО старту й
    # до k ранжованих ланцюгів, як їх бачить аналітик в UI.
    prod = reasoner.hypotheses_from(starts, goal, k=k, weight_mode=weight_mode,
                                    max_hops=max_hops) if starts else []
    if prod:
        top = prod[0]["chain"]
        union: list[str] = []
        for h in prod:
            for node in h["chain"]:
                if node not in union:
                    union.append(node)
        row.update({
            "c_chains": len(prod),
            "c_hops": prod[0]["hops"],
            "c_coverage": step_coverage(top, gold),
            "c_ordered": ordered_coverage(top, gold),
            "c_precision": precision_against_gold(top, gold),
            "c_coverage_topk": step_coverage(union, gold),
        })
    else:
        row.update({key: empty[key] for key in empty if key.startswith("c_")})
    # --- Гілка D: КОНТРОЛЬ — ті самі старти, але односторонній обхід --------
    # Без неї порівняння A vs C не інтерпретується: там міняються ОДРАЗУ два
    # фактори — примітив обходу і те, хто обирає старти. D отримує рівно ті
    # самі старти, ранжування, k і стелю хопів, що C, і відрізняється лише
    # обходом. Тоді C−D — внесок двофронтовості, D−A — внесок заданих стартів.
    ctrl = reasoner.hypotheses_from(starts, goal, k=k, weight_mode=weight_mode,
                                    max_hops=max_hops,
                                    bidirectional=False) if starts else []
    if ctrl:
        top = ctrl[0]["chain"]
        union = []
        for h in ctrl:
            for node in h["chain"]:
                if node not in union:
                    union.append(node)
        row.update({
            "d_chains": len(ctrl),
            "d_hops": ctrl[0]["hops"],
            "d_coverage": step_coverage(top, gold),
            "d_ordered": ordered_coverage(top, gold),
            "d_precision": precision_against_gold(top, gold),
            "d_coverage_topk": step_coverage(union, gold),
        })
    else:
        row.update({key: empty[key] for key in empty if key.startswith("d_")})

    # --- Гілка E: ОРІЄНТОВАНИЙ ТИПІЗОВАНИЙ граф ----------------------------
    # Гілки A-D ходять одним неорієнтованим графом співзустрічальності й
    # відрізняються лише обходом та стартами. Тут міняється САМА МОДЕЛЬ: ребро
    # має напрям і назву, узяті з полів опису. Це повний конвеєр статті —
    # формули (8)-(11): пряме розширення, зворотне, узгодження підмереж,
    # перебір допустимих шляхів. Метрика та сама й еталон той самий, тож
    # числа зіставні з рештою гілок.
    twin_chains: list[list[str]] = []
    twin_nodes: list[list[str]] = []      # поняття кожної СУТНОСТІ першого шляху
    # Ціль передаємо явно — міст `attack_path` не читає.
    twin = build_twin_model(machine, goal or "")
    if twin.goal_state_id:
        context = reconcile(twin, expand_forward(twin), expand_backward(twin))
        # Ідемо ЛИШЕ переходами: структурні відношення (вкладеність, розміщення)
        # лишаються в моделі для читання, але кроком сценарію не є.
        allowed = set(transition_edge_ids(twin))
        context["edge_ids"] = [i for i in context["edge_ids"] if i in allowed]
        # Беремо ШИРШИЙ набір кандидатів і ранжуємо його інтегральною оцінкою J.
        # Власне ранжування потрібне тому, що `enumerate_scenarios` сортує за
        # кількістю вузлів kind=`control` (критерій із іншого застосування
        # модуля), а їх у цьому графі немає — сортування вироджувалось в
        # алфавітне за id, і «першим» ставав випадковий шлях.
        found = enumerate_scenarios(twin, context, matched=set(), max_paths=20)
        ranked = rank_scenarios(twin, found["scenarios"])[:k]
        twin_chains = [path_concepts(twin, sc["path"]) for sc in ranked]
        if ranked:
            by_id = twin.node_by_id
            twin_nodes = [by_id[nid].attributes.get("concepts", [])
                          for nid in ranked[0]["path"]]

    row["e_goal_reachable"] = bool(twin_chains)
    if twin_chains:
        top = twin_chains[0]
        union = []
        for chain in twin_chains:
            for node in chain:
                if node not in union:
                    union.append(node)
        row.update({
            "e_chains": len(twin_chains),
            "e_hops": max(len(top) - 1, 0),
            "e_coverage": step_coverage(top, gold),
            "e_ordered": ordered_coverage(top, gold),
            "e_precision": precision_against_gold(top, gold),
            "e_coverage_topk": step_coverage(union, gold),
            "e_nodes": len(twin_nodes),
            "e_precision_nodes": entity_precision(twin_nodes, gold),
        })
    else:
        row.update({key: empty[key] for key in empty
                    if key.startswith("e_") and key != "e_goal_reachable"})

    # Сирі ланцюги кожної гілки — для дампу даних і атласу. У CSV не йдуть
    # (їх немає у FIELDS), тому підкреслення в імені.
    row["_arms"] = {
        "a_unidirectional": [h["chain"] for h in hyps],
        "b_bidirectional_single": [bidir["path"]] if bidir else [],
        "c_bidirectional_production": [h["chain"] for h in prod],
        "d_unidirectional_given_starts": [h["chain"] for h in ctrl],
        "e_directed_typed_graph": twin_chains,
    }
    row["_gold"] = [sorted(step) for step in gold]
    row["_starts"] = starts
    row["_records"] = records
    return row


def summarise(rows: list[dict]) -> dict:
    """Агрегати рахуємо ЛИШЕ по машинах, які взагалі МОЖНА оцінити: ціль є в
    графі І після відкидання цільового кроку лишився хоч один еталонний крок.
    Інакше середнє змішує «не впорався» з «не було чого шукати»."""
    scored = [r for r in rows if r["goal_in_graph"] and r["gold_steps"] > 0]

    def stats(key: str) -> dict:
        values = [r[key] for r in scored]
        if not values:
            return {"mean": 0.0, "median": 0.0, "sd": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(statistics.fmean(values), 4),
            "median": round(statistics.median(values), 4),
            "sd": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    return {
        "machines_total": len(rows),
        "machines_goal_reachable_in_directed_graph": sum(
            1 for r in scored if r.get("e_goal_reachable")),
        "machines_scored": len(scored),
        "machines_goal_missing_from_graph": sum(1 for r in rows if not r["goal_in_graph"]),
        "machines_without_scorable_gold": sum(
            1 for r in rows if r["goal_in_graph"] and r["gold_steps"] == 0),
        "machines_with_bidirectional_path": sum(1 for r in scored if r["b_found"]),
        "unidirectional_from_goal": {
            "coverage": stats("a_coverage"),
            "ordered_coverage": stats("a_ordered"),
            "precision": stats("a_precision"),
            "coverage_top_k": stats("a_coverage_topk"),
        },
        "bidirectional_single_path": {
            "coverage": stats("b_coverage"),
            "ordered_coverage": stats("b_ordered"),
            "precision": stats("b_precision"),
        },
        "bidirectional_production": {
            "coverage": stats("c_coverage"),
            "ordered_coverage": stats("c_ordered"),
            "precision": stats("c_precision"),
            "coverage_top_k": stats("c_coverage_topk"),
        },
        # Орієнтований типізований граф: інша МОДЕЛЬ, а не інший обхід.
        "directed_typed_graph": {
            "coverage": stats("e_coverage"),
            "ordered_coverage": stats("e_ordered"),
            "precision": stats("e_precision"),
            "precision_entities": stats("e_precision_nodes"),
            "path_entities": stats("e_nodes"),
            "coverage_top_k": stats("e_coverage_topk"),
        },
        # Контроль: старти ті самі, що в гілці вище, обхід — односторонній.
        "unidirectional_from_given_starts": {
            "coverage": stats("d_coverage"),
            "ordered_coverage": stats("d_ordered"),
            "precision": stats("d_precision"),
            "coverage_top_k": stats("d_coverage_topk"),
        },
    }


FIELDS = [
    "split", "machine", "documents", "nodes", "edges", "gold_steps", "goal",
    "goal_in_graph", "a_chains", "a_hops", "a_coverage", "a_ordered",
    "a_precision", "a_coverage_topk", "b_found", "b_hops", "b_coverage",
    "b_ordered", "b_precision", "c_chains", "c_hops", "c_coverage",
    "c_ordered", "c_precision", "c_coverage_topk",
    "d_chains", "d_hops", "d_coverage", "d_ordered", "d_precision",
    "d_coverage_topk",
    "e_chains", "e_hops", "e_coverage", "e_ordered", "e_precision",
    "e_coverage_topk", "e_goal_reachable", "e_nodes", "e_precision_nodes",
]


def run(splits, k: int, weight_mode: str, max_hops: int) -> tuple[list[dict], dict]:
    rows = []
    for split in splits:
        for machine in load_machines(split):
            rows.append(evaluate(machine, k, weight_mode, max_hops))
    return rows, summarise(rows)


def write_raw(rows: list[dict], out_dir: Path) -> int:
    """Сирі дані по КОЖНІЙ машині окремим JSON.

    Навіщо окремо від CSV: у таблиці лишаються лише числа, а перевірити
    результат неможливо, не бачачи, ЩО саме подавалось на вхід і ЯКІ ланцюги
    вийшли. Тут є і корпус документів, і еталон, і ланцюги всіх трьох гілок —
    достатньо, щоб перерахувати метрику вручну."""
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        name = (row.get("machine") or "unknown").lower().replace(" ", "_")
        payload = {
            "machine": row.get("machine"),
            "split": row.get("split"),
            "input": {
                "documents": row.get("documents"),
                "graph_nodes": row.get("nodes"),
                "graph_edges": row.get("edges"),
                "entry_concepts": row.get("_starts", []),
                "corpus": row.get("_records", []),
            },
            "reference": {
                "goal": row.get("goal"),
                "goal_in_graph": row.get("goal_in_graph"),
                "gold_steps_scored": row.get("_gold", []),
                "note": "Цільовий крок відкинуто: він поданий на вхід обходу.",
            },
            "chains": row.get("_arms", {}),
            "metrics": {key: row.get(key) for key in FIELDS
                        if key.startswith(("a_", "b_", "c_"))},
        }
        with (raw_dir / f"{name}.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write(chr(10))
        written += 1
    return written


def write_results(rows: list[dict], summary: dict, config: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_case.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"config": config, "summary": summary}, fh,
                  ensure_ascii=False, indent=2)
        fh.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="demo",
                        help="за замовчуванням demo — контрольні 22 машини")
    parser.add_argument("--k", type=int, default=5, help="скільки гіпотез брати (прод: 5)")
    parser.add_argument("--weight-mode", default="npmi", choices=("npmi", "raw"))
    parser.add_argument("--max-hops", type=int, default=6)
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--check-determinism", action="store_true",
                        help="прогнати двічі й звірити результат")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass

    splits = SPLITS if args.split == "all" else (args.split,)
    config = {
        "splits": list(splits), "k": args.k, "weight_mode": args.weight_mode,
        "max_hops": args.max_hops,
        "reasoner_max_terms_per_doc": os.getenv("REASONER_MAX_TERMS_PER_DOC", "40"),
        "uses_llm": False, "uses_network": False,
    }

    rows, summary = run(splits, args.k, args.weight_mode, args.max_hops)

    if args.check_determinism:
        rows2, summary2 = run(splits, args.k, args.weight_mode, args.max_hops)
        identical = rows == rows2 and summary == summary2
        config["deterministic_second_run"] = identical
        print(f"детермінізм повторного запуску: {'ТАК' if identical else 'НІ'}")
        if not identical:
            return 1

    write_results(rows, summary, config, Path(args.out))
    n_raw = write_raw(rows, Path(args.out))

    a = summary["unidirectional_from_goal"]
    b = summary["bidirectional_single_path"]
    c = summary["bidirectional_production"]
    print(f"\nМашин усього: {summary['machines_total']}; "
          f"оцінено: {summary['machines_scored']} "
          f"(цілі немає в графі: {summary['machines_goal_missing_from_graph']}, "
          f"нічим оцінювати: {summary['machines_without_scorable_gold']}); "
          f"двофронтовий шлях знайдено: {summary['machines_with_bidirectional_path']}")
    print(f"{'':34}{'середнє':>9}{'медіана':>9}{'SD':>7}")
    for label, block in (("A · від цілі назад (одностор.)", a),
                         ("B · двофронтовий, один шлях", b),
                         ("C · двофронтовий у проді (hypotheses_from)", c)):
        print(f"  {label}")
        for name, key in (("покриття кроків", "coverage"),
                          ("покриття з порядком", "ordered_coverage"),
                          ("точність ланцюга", "precision")):
            s = block[key]
            print(f"    {name:<30}{s['mean']:>9.3f}{s['median']:>9.3f}{s['sd']:>7.3f}")
    for lbl, block in (("A", a), ("C", c)):
        s = block["coverage_top_k"]
        print(f"  {lbl} · покриття по top-k разом    {s['mean']:>9.3f}{s['median']:>9.3f}{s['sd']:>7.3f}")
    print(f"\nЗаписано: {Path(args.out) / 'per_case.csv'}, {Path(args.out) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
