#!/usr/bin/env python3
"""Абляція узгодження підмереж (`qa.twin.reconcile`) на 100 корпоративних
мережах (`experiments/corporate_networks/dataset/`).

ЩО ЦЕ ВИМІРЮЄ. Бойовий шлях звужує простір пошуку S0->ST формулою
"пряме розширення ∩ зворотне розширення" (`expand_forward`, `expand_backward`,
`reconcile` -> G_C) ДО того, як `enumerate_scenarios` іде в DFS. Контекст
задачі вже виміряв РОЗМІР цього звуження (G_C ~31 вузол проти ~154 у повному
графі) і що БЕЗ звуження (увесь граф як простір) пошук не завершується за
розумний час — комбінаторний вибух. Питання, на яке відповідає цей скрипт:
чи платимо ми ЯКІСТЮ знайдених шляхів за це звуження, а не лише швидкістю.

ДВІ ГІЛКИ, ОДНАКОВЕ ВСЕ ІНШЕ:
  A (із узгодженням): gc = reconcile(model, expand_forward(model),
                       expand_backward(model))
  B (без узгодження):  gc = увесь граф моделі — {"node_ids": усі вузли,
                       "edge_ids": усі індекси ребер} (`full_graph_scope`)
Обидві йдуть у ТОЙ САМИЙ виклик `enumerate_scenarios(model, gc,
matched=set(), max_paths=5)` і ТЕ САМЕ ранжування `rank_scenarios_corp`
(`experiments/corporate_networks/run_experiment.py`, не чіпаємо — лише
імпортуємо). Єдина відмінність між гілками зосереджена в `select_search_
space` нижче.

ЧОМУ ОКРЕМІ ПРОЦЕСИ, А НЕ ПОТОКИ АБО ГОЛИЙ ВИКЛИК З ТАЙМАУТОМ. Python не
вміє перервати обчислювальний потік: CPython перемикає потоки лише між
байт-кодами, а DFS-перебір `enumerate_scenarios` — суцільний CPU-цикл без
природних точок зупинки. "Таймаут" на потоці залишає сам потік живим і далі
споживаючим CPU — це вже сталося: прогін без узгодження на частковому
проході з'їв 1230с CPU за 24 хвилини, поки процес не вбили вручну. Єдиний
надійний спосіб гарантовано зупинити обчислення — окремий ОС-процес, який
можна вбити (`Popen.kill()`) і ПЕРЕВІРИТИ, що він дійсно мертвий
(`run_subprocess_with_deadline` нижче).

Метрики покриття/точності (`coverage`/`ordered_coverage`/`precision`/
`coverage_top_k`) — ПЕРЕВИКОРИСТАНІ з `experiments.htb_reconstruction.metric`
через `score_branch_run`; жодної власної формули покриття тут немає.

Запуск:
    python tools/run_reconcile_ablation.py
        [--dataset DIR] [--out DIR] [--budget SECONDS]

Внутрішній режим воркера (використовує сам скрипт, викликаючи себе окремим
процесом на кожну (мережа, гілка) — див. `run_branch_for_network`):
    python tools/run_reconcile_ablation.py --run-branch NETWORK_ID BRANCH
        --dataset DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from qa.twin import enumerate_scenarios, expand_backward, expand_forward, reconcile  # noqa: E402
from experiments.corporate_networks.dataset import (  # noqa: E402
    build_twin_model, load_answers, parse_gold_attacks, parse_network,
)
from experiments.corporate_networks.run_experiment import (  # noqa: E402
    gold_attack_cve_sets, load_manifest, path_cves, rank_scenarios_corp,
)
from experiments.htb_reconstruction.metric import (  # noqa: E402
    ordered_coverage, precision_against_gold, step_coverage,
)

DEFAULT_DATASET_DIR = REPO_ROOT / "experiments" / "corporate_networks" / "dataset"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "corporate_networks" / "results"

#: Бюджет часу на ОДНУ мережу, ОДНУ гілку — винесено константою за вимогою
#: задачі (і щоб її не можна було непомітно підкрутити під бажаний
#: результат). НЕ підбирався заднім числом за фактичними числами прогону.
NETWORK_BUDGET_SECONDS = 10.0

#: `max_paths` дослівно з формули завдання: `enumerate_scenarios(model, gc,
#: matched=set(), max_paths=5)` — БЕЗ ширшого пулу кандидатів (`SCENARIO_
#: POOL=20`), що використовує `run_experiment.evaluate_network`: тут
#: важливо, щоб обидві гілки бачили РІВНО той самий виклик, який задача
#: прописала явно, а не наближену репліку бойового шляху.
MAX_PATHS = 5

_QUALITY_KEYS = ("coverage", "ordered_coverage", "precision", "coverage_top_k")


# =====================================================================
#  Простір пошуку: єдине місце, де гілки A і B відрізняються
# =====================================================================

def full_graph_scope(model) -> dict[str, list]:
    """Увесь граф моделі як простір пошуку — контрольна гілка B: без
    жодного звуження (ні forward/backward BFS, ні reconcile), рівно
    {усі вузли, усі індекси ребер} у ТОМУ САМОМУ форматі, що повертає
    `qa.twin.reconcile` — так `enumerate_scenarios` не бачить різниці,
    окрім самого розміру простору."""
    return {
        "node_ids": [node.id for node in model.nodes],
        "edge_ids": list(range(len(model.edges))),
    }


def select_search_space(model, branch: str) -> dict[str, list]:
    """Простір пошуку для гілки абляції.

    "A" — узгоджена G_C (пряме розширення ∩ зворотне, дослівно формула
    задачі: `reconcile(model, expand_forward(model), expand_backward(model))`).
    "B" — контроль: увесь граф без жодного звуження (`full_graph_scope`).

    Уся відмінність між двома гілками експерименту зосереджена в ЦІЙ одній
    функції — `compute_branch` нижче викликає `enumerate_scenarios`/
    ранжування ОДНАКОВО для обох, що і є гарантією "нічого іншого не
    міняй" з вимоги задачі."""
    if branch == "A":
        forward = expand_forward(model)
        backward = expand_backward(model)
        return reconcile(model, forward, backward)
    if branch == "B":
        return full_graph_scope(model)
    raise ValueError(f"unknown ablation branch {branch!r} (очікувалось 'A' або 'B')")


def compute_branch(model, branch: str, max_paths: int = MAX_PATHS) -> dict[str, object]:
    """Пошук + ранжування сценаріїв на просторі однієї гілки.

    Формула — дослівно з вимоги задачі: `enumerate_scenarios(model, gc,
    matched=set(), max_paths=5)`, потім те САМЕ ранжування
    (`rank_scenarios_corp`), що й у бойовому шляху `run_experiment.py`, —
    жодної окремої формули ранжування для абляції не заводимо."""
    gc = select_search_space(model, branch)
    found = enumerate_scenarios(model, gc, matched=set(), max_paths=max_paths)
    ranked = rank_scenarios_corp(model, found["scenarios"])
    return {"total_found": found["total_found"], "scenarios": ranked}


# =====================================================================
#  Якість: ЛИШЕ перевикористані метрики `experiments.htb_reconstruction.
#  metric` — жодної власної формули покриття/точності.
# =====================================================================

def score_branch_run(model, scenarios: list[dict], gold_cve_sets: list[set]) -> dict[str, float]:
    """coverage/ordered_coverage/precision/coverage_top_k для сценаріїв
    ОДНІЄЇ гілки — та сама конвенція, що й `run_experiment.evaluate_network`:
    top-1 ланцюг для coverage/ordered_coverage/precision, обʼєднання ланцюгів
    усіх знайдених шляхів (у порядку появи, без дублів) для coverage_top_k.
    Відсутність шляхів -> чесний нуль, а не пропущений ключ."""
    if not scenarios:
        return {key: 0.0 for key in _QUALITY_KEYS}
    top1_chain = path_cves(model, scenarios[0]["path"])
    union_chain: list[str] = []
    for scenario in scenarios:
        for cve in path_cves(model, scenario["path"]):
            if cve not in union_chain:
                union_chain.append(cve)
    return {
        "coverage": step_coverage(top1_chain, gold_cve_sets),
        "ordered_coverage": ordered_coverage(top1_chain, gold_cve_sets),
        "precision": precision_against_gold(top1_chain, gold_cve_sets),
        "coverage_top_k": step_coverage(union_chain, gold_cve_sets),
    }


# =====================================================================
#  Ізоляція процесом: жорсткий дедлайн + гарантоване вбивство
# =====================================================================

def run_subprocess_with_deadline(cmd: list[str], timeout_seconds: float) -> dict[str, object]:
    """Запускає `cmd` окремим ОС-процесом із жорстким дедлайном; повертає
    словник з ЯВНИМ статусом (`completed: bool`), а не 0 чи виключення на
    таймауті (вимога задачі: таймаут має дати явний маркер "не завершилось").

    ЧОМУ ОКРЕМИЙ ПРОЦЕС, А НЕ ПОТІК. Python не вміє перервати обчислювальний
    потік — DFS-перебір `enumerate_scenarios` суцільний CPU-цикл без точок
    зупинки; "таймаут" на потоці лишає сам потік живим і далі споживаючим
    CPU (так один прогін без узгодження зжер 1230с CPU за 24 хв., поки
    процес не вбили вручну). ОС-процес можна гарантовано вбити.

    ЯК ПЕРЕВІРЕНО, ЩО ПРОЦЕС ДІЙСНО МЕРТВИЙ, А НЕ ЛИШИВСЯ У ФОНІ.
    `Popen.communicate()` ПІСЛЯ `kill()` не лише дочитує залишки виводу, а й
    ЧЕКАЄ на фактичне завершення процесу (POSIX: збирає зомбі; Windows:
    звільняє хендл) — саме тому `returncode` гарантовано НЕ `None` одразу
    після повернення з цієї функції: це офіційна семантика `subprocess.
    Popen` ("returncode встановлюється лише після завершення дочірнього
    процесу"), а не припущення."""
    started = time.monotonic()
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        elapsed = time.monotonic() - started
        return {
            "completed": True, "elapsed_seconds": elapsed,
            "stdout": stdout, "stderr": stderr, "returncode": process.returncode,
        }
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        elapsed = time.monotonic() - started
        assert process.returncode is not None, (
            "процес лишився живим після kill()+communicate() — "
            "гарантія завершення процесу порушена"
        )
        return {
            "completed": False, "elapsed_seconds": elapsed,
            "stdout": stdout, "stderr": stderr, "returncode": process.returncode,
        }


def run_branch_for_network(
    python_exe: str, script_path: Path, dataset_dir: Path,
    network_id: str, branch: str, budget_seconds: float,
) -> dict[str, object]:
    """Один (мережа, гілка) -> результат воркера, запущеного окремим
    процесом із бюджетом `budget_seconds`. `status`:
      "ok"      — воркер завершився в бюджеті з кодом 0, JSON розібрано;
      "timeout" — вихід за бюджет часу (процес вбито й підтверджено мертвим);
      "error"   — процес завершився сам, але не кодом 0 (не мало б
                  траплятись на тому самому коді пошуку, що й гілка A, —
                  фіксується окремо від "timeout", щоб не змішати дві різні
                  причини відсутності результату)."""
    cmd = [
        python_exe, str(script_path),
        "--run-branch", network_id, branch,
        "--dataset", str(dataset_dir),
    ]
    outcome = run_subprocess_with_deadline(cmd, budget_seconds)
    if not outcome["completed"]:
        return {
            "status": "timeout", "wall_seconds": outcome["elapsed_seconds"],
            "search_seconds": None, "total_found": None, "scenarios": [], "quality": None,
        }
    if outcome["returncode"] != 0:
        return {
            "status": "error", "wall_seconds": outcome["elapsed_seconds"],
            "search_seconds": None, "total_found": None, "scenarios": [], "quality": None,
            "stderr_tail": outcome["stderr"][-2000:],
        }
    payload = json.loads(outcome["stdout"])
    return {
        "status": "ok", "wall_seconds": outcome["elapsed_seconds"],
        "search_seconds": payload["search_seconds"],
        "total_found": payload["total_found"],
        "scenarios": payload["scenarios"],
        "quality": payload["quality"],
    }


# =====================================================================
#  Агрегація: чесне порівняння ЛИШЕ на спільній підмножині завершених
# =====================================================================

def common_completed_networks(records: dict[str, dict]) -> list[str]:
    """`records` = мережа -> {"A": результат_гілки, "B": результат_гілки}.
    Повертає мережі, де ОБИДВІ гілки завершились у бюджеті (`status ==
    "ok"`) — єдина коректна підмножина для порівняння якості (вимога
    задачі: порівнювати A на всіх 100 з B на підмножині нечесно —
    підмножина зміщена в бік простіших/менших мереж)."""
    return sorted(
        network_id for network_id, branches in records.items()
        if branches["A"]["status"] == "ok" and branches["B"]["status"] == "ok"
    )


def _branch_found_paths(run: dict) -> bool:
    return bool(run.get("scenarios"))


def b_only_reachable_networks(records: dict[str, dict]) -> list[str]:
    """Мережі зі СПІЛЬНОЇ підмножини (обидві гілки завершились), де B
    знайшла хоч один шлях, а A — жодного. Непорожній список — це ЦІНА
    звуження підмножин, названа поіменно (вимога задачі, пункт 5)."""
    return sorted(
        network_id for network_id in common_completed_networks(records)
        if _branch_found_paths(records[network_id]["B"])
        and not _branch_found_paths(records[network_id]["A"])
    )


def a_only_reachable_networks(records: dict[str, dict]) -> list[str]:
    """Симетричний бухгалтерський запис до `b_only_reachable_networks` —
    мережі, де ЗВУЖЕНИЙ простір (A) знайшов шлях, а повний граф (B) у
    межах ТОГО САМОГО `max_paths=5`-DFS — ні (можливо через те, що DFS на
    величезному повному графі "витрачає" ліміт кандидатів на глухі кути
    ще ДО того, як дістається реального шляху)."""
    return sorted(
        network_id for network_id in common_completed_networks(records)
        if _branch_found_paths(records[network_id]["A"])
        and not _branch_found_paths(records[network_id]["B"])
    )


def path_overlap_share(paths_a: list[list[str]], paths_b: list[list[str]]) -> float:
    """Індекс Жаккара між множинами шляхів (кожен шлях — кортеж id вузлів)
    двох гілок: 1.0 — рівно ті самі шляхи, 0.0 — жодного спільного. Обидві
    множини порожні -> 1.0 (гілки узгоджено НІЧОГО не знайшли — це збіг, а
    не "0% спільних шляхів", якого просто немає що ділити)."""
    set_a = {tuple(path) for path in paths_a}
    set_b = {tuple(path) for path in paths_b}
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"sum": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "sum": round(sum(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def summarise_ablation(records: dict[str, dict]) -> dict[str, object]:
    """Зведення абляції по всіх мережах `records`:
      1. скільки мереж кожна гілка завершила в бюджеті / не завершила;
      2. час на мережу (лише завершені; незавершені — окремий факт-список,
         НЕ підставляючи бюджет як "час");
      3. якість ЛИШЕ на спільній підмножині завершених обома гілками;
      4. скільки шляхів знайшла кожна гілка й частка збігу множин шляхів
         (Жаккар) — теж лише на спільній підмножині;
      5. мережі, де B знайшла шлях, а A — ні (і симетрично навпаки)."""
    networks_total = len(records)

    def _status_counts(branch: str) -> dict[str, int]:
        statuses = [records[nid][branch]["status"] for nid in records]
        return {
            "ok": statuses.count("ok"),
            "timeout": statuses.count("timeout"),
            "error": statuses.count("error"),
        }

    def _timing(branch: str) -> dict[str, object]:
        completed = [nid for nid in records if records[nid][branch]["status"] == "ok"]
        times = [records[nid][branch]["search_seconds"] for nid in completed]
        timed_out = sorted(nid for nid in records if records[nid][branch]["status"] == "timeout")
        errored = sorted(nid for nid in records if records[nid][branch]["status"] == "error")
        return {
            "completed_networks": len(completed),
            "search_seconds": _stats(times),
            "timed_out_networks": timed_out,
            "timed_out_count": len(timed_out),
            "errored_networks": errored,
        }

    common = common_completed_networks(records)

    quality: dict[str, dict] = {}
    if common:
        for branch in ("A", "B"):
            quality[branch] = {
                metric: _stats([records[nid][branch]["quality"][metric] for nid in common])
                for metric in _QUALITY_KEYS
            }

    path_counts: dict[str, object] = {}
    overlaps: list[float] = []
    identical = 0
    for network_id in common:
        paths_a = [scenario["path"] for scenario in records[network_id]["A"]["scenarios"]]
        paths_b = [scenario["path"] for scenario in records[network_id]["B"]["scenarios"]]
        overlap = path_overlap_share(paths_a, paths_b)
        overlaps.append(overlap)
        if overlap == 1.0:
            identical += 1
    if common:
        path_counts = {
            "branch_a_mean_paths": round(
                statistics.fmean(len(records[nid]["A"]["scenarios"]) for nid in common), 4,
            ),
            "branch_b_mean_paths": round(
                statistics.fmean(len(records[nid]["B"]["scenarios"]) for nid in common), 4,
            ),
            "mean_jaccard_overlap": round(statistics.fmean(overlaps), 4),
            "identical_path_set_share": round(identical / len(common), 4),
        }

    b_only = b_only_reachable_networks(records)
    a_only = a_only_reachable_networks(records)

    return {
        "networks_total": networks_total,
        "branch_a": {"status_counts": _status_counts("A"), "timing": _timing("A")},
        "branch_b": {"status_counts": _status_counts("B"), "timing": _timing("B")},
        "common_completed_networks": common,
        "common_completed_count": len(common),
        "quality_common_subset": quality,
        "path_counts_common_subset": path_counts,
        "b_only_reachable_networks": b_only,
        "a_only_reachable_networks": a_only,
        # `True`, лише якщо на спільній підмножині є ХОЧ ОДНА мережа, де
        # ширший простір (B) знайшов шлях, а звужений (A) — ні: це і є
        # пряма, вимірна ціна звуження в термінах ДОСЯЖНОСТІ (а не лише
        # швидкості пошуку).
        "narrowing_costs_reachability": bool(b_only),
    }


# =====================================================================
#  Звіт: `ablation_reconcile.md` — жодне число не вписане руками
# =====================================================================

def _fmt(value: object, digits: int = 4) -> str:
    return f"{float(value or 0.0):.{digits}f}"


def render_ablation_report(summary: dict, config: dict) -> str:
    """Повний текст `ablation_reconcile.md`, зібраний ЦІЛКОМ із порахованого
    `summary`/`config` — жодна константа тут не підставляється заднім
    числом (та сама дисципліна, що й `render_readme` у `run_experiment.py`)."""
    a_status = summary["branch_a"]["status_counts"]
    b_status = summary["branch_b"]["status_counts"]
    a_timing = summary["branch_a"]["timing"]
    b_timing = summary["branch_b"]["timing"]
    common_count = summary["common_completed_count"]
    quality = summary.get("quality_common_subset") or {}
    path_counts = summary.get("path_counts_common_subset") or {}
    b_only = summary.get("b_only_reachable_networks") or []
    a_only = summary.get("a_only_reachable_networks") or []

    lines = [
        "# Абляція узгодження підмереж (`reconcile`)",
        "",
        "Вимірюємо внесок формули «пряме розширення ∩ зворотне розширення "
        "-> G_C» (`qa.twin.reconcile`) у якість знайдених шляхів атаки: "
        "гілка A шукає на узгодженій підмережі G_C, гілка B — на ВСЬОМУ "
        "графі моделі без жодного звуження, обидві — тим самим викликом "
        f"`enumerate_scenarios(model, gc, matched=set(), max_paths="
        f"{config.get('max_paths')})` і тим самим ранжуванням.",
        "",
        f"Бюджет на мережу: **{_fmt(config.get('budget_seconds'), 1)} с** "
        "на гілку (окремий процес, жорсткий таймаут + примусове "
        "завершення — Python не вміє перервати обчислювальний потік, "
        "перевірено вимірюванням, що процес після таймауту дійсно мертвий).",
        f"Мереж усього: **{summary['networks_total']}**.",
        "",
        "## 1. Завершеність у бюджеті — головне число",
        "",
        "| Гілка | завершено (ok) | таймаут | помилка |",
        "|---|---|---|---|",
        f"| A (з узгодженням) | {a_status['ok']} | {a_status['timeout']} | {a_status['error']} |",
        f"| B (без узгодження, увесь граф) | {b_status['ok']} | {b_status['timeout']} | {b_status['error']} |",
        "",
    ]
    if b_status["timeout"]:
        lines.append(
            f"Гілка B НЕ завершилась у бюджеті на **{b_status['timeout']}** "
            f"мереж(ах): {', '.join(b_timing['timed_out_networks']) or '—'}."
        )
    else:
        lines.append(
            "Гілка B завершилась у бюджеті на КОЖНІЙ мережі — при "
            f"{_fmt(config.get('budget_seconds'), 1)}с звуження простору "
            "пошуку не було необхідним для завершення в часі (на цьому "
            "датасеті); висновок про комбінаторний вибух ґрунтується на "
            "часі пошуку, а не на самому факті завершення."
        )
    lines.extend([
        "",
        "## 2. Час на мережу (лише завершені; таймаут — окремий факт, "
        "НЕ підставлений як час)",
        "",
        "| Гілка | завершених | сума, с | середнє, с | медіана, с | максимум, с |",
        "|---|---|---|---|---|---|",
        (
            f"| A | {a_timing['completed_networks']} | "
            f"{_fmt(a_timing['search_seconds']['sum'])} | "
            f"{_fmt(a_timing['search_seconds']['mean'])} | "
            f"{_fmt(a_timing['search_seconds']['median'])} | "
            f"{_fmt(a_timing['search_seconds']['max'])} |"
        ),
        (
            f"| B | {b_timing['completed_networks']} | "
            f"{_fmt(b_timing['search_seconds']['sum'])} | "
            f"{_fmt(b_timing['search_seconds']['mean'])} | "
            f"{_fmt(b_timing['search_seconds']['median'])} | "
            f"{_fmt(b_timing['search_seconds']['max'])} |"
        ),
        "",
        "## 3. Якість — ЛИШЕ на спільній підмножині завершених обома "
        "гілками (порівняння A на всіх 100 із B на підмножині було б "
        "нечесним — підмножина зміщена в бік простіших мереж)",
        "",
        f"Мереж у спільній підмножині: **{common_count}**.",
        "",
    ])
    if quality:
        lines.append("| Показник | A (з узгодженням) | B (без узгодження) |")
        lines.append("|---|---|---|")
        for metric in _QUALITY_KEYS:
            lines.append(
                f"| {metric} | {_fmt(quality['A'][metric]['mean'])} | "
                f"{_fmt(quality['B'][metric]['mean'])} |"
            )
    else:
        lines.append(
            "Немає жодної мережі, де завершились ОБИДВІ гілки — порівняння "
            "якості неможливе на цьому прогоні."
        )
    lines.extend(["", "## 4. Скільки шляхів і чи ті самі (спільна підмножина)", ""])
    if path_counts:
        lines.extend([
            f"Середня кількість знайдених шляхів: A={_fmt(path_counts['branch_a_mean_paths'])}, "
            f"B={_fmt(path_counts['branch_b_mean_paths'])}.",
            f"Середній індекс Жаккара між множинами шляхів A і B: "
            f"**{_fmt(path_counts['mean_jaccard_overlap'])}** "
            f"(частка мереж з РІВНО тими самими шляхами: "
            f"{_fmt(path_counts['identical_path_set_share'])}).",
        ])
    else:
        lines.append("Спільної підмножини немає — порівнювати шляхи нема на чому.")
    lines.extend(["", "## 5. Ціна звуження в термінах досяжності", ""])
    if b_only:
        lines.append(
            f"**Так, ціна є**: на {len(b_only)} мереж(ах) повний граф (B) "
            f"знайшов шлях до цілі, а узгоджена підмережа (A) — жодного: "
            f"{', '.join(b_only)}."
        )
    else:
        lines.append(
            "Ціни немає: на СПІЛЬНІЙ підмножині завершених мереж жодного разу "
            "гілка B не знайшла шлях там, де гілка A — ні."
        )
    if a_only:
        lines.append(
            f"Симетрично: на {len(a_only)} мереж(ах) навпаки — A знайшла "
            f"шлях, а B (у межах того самого ліміту DFS-кандидатів на "
            f"величезному повному графі) — ні: {', '.join(a_only)}."
        )
    return "\n".join(lines) + "\n"


# =====================================================================
#  Введення/виведення
# =====================================================================

def _write_ablation_results(
    summary: dict, per_network_records: dict, config: dict, out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config, "summary": summary,
        "networks": {
            network_id: {
                branch: {
                    key: value for key, value in run.items() if key != "stderr_tail"
                }
                for branch, run in branches.items()
            }
            for network_id, branches in per_network_records.items()
        },
    }
    with (out_dir / "ablation_reconcile.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    with (out_dir / "ablation_reconcile.md").open("w", encoding="utf-8") as fh:
        fh.write(render_ablation_report(summary, config))


# =====================================================================
#  Режим воркера: одна (мережа, гілка) в ОДНОМУ процесі
# =====================================================================

def _worker_main(dataset_dir: Path, network_id: str, branch: str) -> int:
    xml_path = dataset_dir / f"{network_id}.xml"
    xml_text = xml_path.read_text(encoding="utf-8")
    network = parse_network(xml_text)   # ЕТАЛОН СЮДИ НЕ ПОТРАПЛЯЄ
    model = build_twin_model(network)    # модель побудована наосліп

    started = time.monotonic()
    branch_result = compute_branch(model, branch, max_paths=MAX_PATHS)
    search_seconds = time.monotonic() - started

    # Межа: усе ВИЩЕ — без жодного звернення до еталона (той самий порядок,
    # що й `evaluate_network` у run_experiment.py). Усе НИЖЧЕ читає
    # `<attacks>`/`answers.json` лише щоб ОЦІНИТИ вже готову відповідь.
    answers = load_answers(str(dataset_dir / "answers.json"))
    gold_attacks = parse_gold_attacks(xml_text)
    answers_entry = answers.get(network_id, {})
    best_attack_id = answers_entry.get("best_attack", "")
    best_gold = next((a for a in gold_attacks if a["id"] == best_attack_id), None)
    gold_cve_sets = gold_attack_cve_sets(best_gold) if best_gold else []

    quality = score_branch_run(model, branch_result["scenarios"], gold_cve_sets)

    payload = {
        "network": network_id, "branch": branch, "search_seconds": search_seconds,
        "total_found": branch_result["total_found"],
        "scenarios": [
            {"path": scenario["path"], "edge_path": scenario["edge_path"]}
            for scenario in branch_result["scenarios"]
        ],
        "quality": quality,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


# =====================================================================
#  Режим оркестратора: усі мережі датасету, дві гілки на кожну
# =====================================================================

def _orchestrator_main(dataset_dir: Path, out_dir: Path, budget_seconds: float) -> int:
    manifest = load_manifest(str(dataset_dir / "manifest.json"))
    python_exe = sys.executable
    script_path = Path(__file__).resolve()

    records: dict[str, dict] = {}
    xml_paths = sorted(glob.glob(str(dataset_dir / "net-*.xml")))
    for index, xml_path in enumerate(xml_paths, start=1):
        network_id = Path(xml_path).stem
        branches: dict[str, dict] = {}
        for branch in ("A", "B"):
            print(
                f"[{index}/{len(xml_paths)}] {network_id} гілка {branch}...",
                file=sys.stderr, flush=True,
            )
            branches[branch] = run_branch_for_network(
                python_exe, script_path, dataset_dir, network_id, branch, budget_seconds,
            )
        records[network_id] = branches

    summary = summarise_ablation(records)
    config = {
        "dataset": str(dataset_dir), "budget_seconds": budget_seconds,
        "max_paths": MAX_PATHS, "networks_total": len(records),
    }
    _write_ablation_results(summary, records, config, out_dir)

    print(
        f"\nГілка A: {summary['branch_a']['status_counts']}; "
        f"гілка B: {summary['branch_b']['status_counts']}; "
        f"спільна підмножина: {summary['common_completed_count']} мереж(і)."
    )
    print(f"Записано: {out_dir / 'ablation_reconcile.json'}, {out_dir / 'ablation_reconcile.md'}")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-branch", nargs=2, metavar=("NETWORK_ID", "BRANCH"),
        help="внутрішній режим воркера: одна гілка (A/B) на одній мережі, JSON у stdout",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--budget", type=float, default=NETWORK_BUDGET_SECONDS)
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    dataset_dir = Path(args.dataset)
    if args.run_branch:
        network_id, branch = args.run_branch
        return _worker_main(dataset_dir, network_id, branch)
    return _orchestrator_main(dataset_dir, Path(args.out), args.budget)


if __name__ == "__main__":
    raise SystemExit(main())
