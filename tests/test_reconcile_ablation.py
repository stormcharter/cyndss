"""Абляція узгодження підмереж (`qa.twin.reconcile`): гілка A (пряме
розширення ∩ зворотне -> G_C) проти гілки B (увесь граф моделі як простір
пошуку, БЕЗ жодного звуження) — вимірює, чи звуження простору пошуку коштує
якості знайдених шляхів.

Головні гарантії, які тут стережемо (вимога задачі):

  1. Гілка B справді дістає ВЕСЬ граф моделі, а не звужений (`select_search_
     space` / `full_graph_scope`) — перевірено моделлю з навмисним
     "острівцем" вузлів поза жодним шляхом S0->ST: reconcile його виключає,
     повний граф — ні.
  2. Таймаут дає ЯВНИЙ маркер "не завершилось" (`completed: False` у
     словнику), а не 0 (виглядало б як "0 шляхів знайдено") і не непіймане
     виключення (впало б на першій же завислій мережі).
  3. Якість (coverage/ordered_coverage/precision/coverage_top_k) рахується
     ЛИШЕ на мережах, де ОБИДВІ гілки завершились у бюджеті — інакше
     порівняння зміщене (гілка B на підмножині простіших мереж).
  4. Після таймауту ОС-процес дійсно мертвий (`Popen.kill()` + повторний
     `communicate()` -> `returncode is not None`), а не лишається висіти у
     фоні й палити CPU (саме так один прогін без узгодження з'їв 1230с CPU
     за 24 хв., поки процес не вбили вручну — контекст задачі).

Синтетичні фікстури тут — навмисно МАЛЕНЬКІ, некеровані моделі, а не реальний
датасет 100 мереж: юніт-тести мають лишатись швидкими й безпечними (жодного
ризику зависання від комбінаторного вибуху реальної мережі всередині тесту).
Реальний датасет торкається рівно один інтеграційний тест (гілка A на
net-001 — швидка за конструкцією, до 40мс за виміром контексту задачі; гілка
B на реальних мережах свідомо НЕ ганяється в юніт-тестах — її час
непередбачуваний, для цього і потрібен окремий процес з жорстким таймаутом).

Ідуть і під pytest, і напряму: `python tests/test_reconcile_ablation.py`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa.twin import TwinEdge, TwinModel, TwinNode  # noqa: E402
from experiments.corporate_networks.run_experiment import path_cves  # noqa: E402
from experiments.htb_reconstruction.metric import (  # noqa: E402
    ordered_coverage, precision_against_gold, step_coverage,
)
from tools.run_reconcile_ablation import (  # noqa: E402
    a_only_reachable_networks,
    b_only_reachable_networks,
    common_completed_networks,
    compute_branch,
    full_graph_scope,
    path_overlap_share,
    render_ablation_report,
    run_subprocess_with_deadline,
    score_branch_run,
    select_search_space,
    summarise_ablation,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_DIR = os.path.join(_REPO_ROOT, "experiments", "corporate_networks", "dataset")
_SCRIPT_PATH = os.path.join(_REPO_ROOT, "tools", "run_reconcile_ablation.py")


# =====================================================================
#  Синтетичні фікстури
# =====================================================================

def _model_with_island() -> TwinModel:
    """S0 -> N1 -> ST — єдиний реальний шлях. ISLAND_A -> ISLAND_B —
    повністю відокремлений острівець поза будь-яким шляхом S0->ST: не
    forward-reachable від S0, не здатний дістатись ST. `reconcile` має його
    виключити, повний граф — включити як є."""
    nodes = [
        TwinNode("S0", "state", "S0"),
        TwinNode("N1", "asset", "N1"),
        TwinNode("ST", "state", "ST"),
        TwinNode("ISLAND_A", "asset", "ISLAND_A"),
        TwinNode("ISLAND_B", "asset", "ISLAND_B"),
    ]
    edges = [
        TwinEdge("S0", "N1", "reaches"),
        TwinEdge("N1", "ST", "reaches"),
        TwinEdge("ISLAND_A", "ISLAND_B", "reaches"),
    ]
    return TwinModel(
        name="island-fixture", nodes=nodes, edges=edges,
        criteria=[], constraints=[],
        current_state_id="S0", goal_state_id="ST",
    )


def _model_with_vuln_chain() -> TwinModel:
    """S0 -> V1(CVE-2020-0001) -> MID -> V2(CVE-2020-0002) -> ST — єдиний
    шлях з двома вузлами-вразливостями, потрібними `path_cves`."""
    nodes = [
        TwinNode("S0", "state", "S0"),
        TwinNode("V1", "vuln", "V1", attributes={"cve": "CVE-2020-0001"}),
        TwinNode("MID", "state", "MID"),
        TwinNode("V2", "vuln", "V2", attributes={"cve": "CVE-2020-0002"}),
        TwinNode("ST", "state", "ST"),
    ]
    edges = [
        TwinEdge("S0", "V1", "exploits"),
        TwinEdge("V1", "MID", "yields"),
        TwinEdge("MID", "V2", "exploits"),
        TwinEdge("V2", "ST", "yields"),
    ]
    return TwinModel(
        name="vuln-chain-fixture", nodes=nodes, edges=edges,
        criteria=[], constraints=[],
        current_state_id="S0", goal_state_id="ST",
    )


def _fake_run(status, search_seconds=None, scenarios=None, quality=None):
    """Фабрикує один результат гілки без реального пошуку чи підпроцесу —
    для тестів агрегації (`summarise_ablation` і сусідні), які не мають
    залежати від реального датасету чи таймінгу."""
    scenarios = scenarios or []
    return {
        "status": status,
        "wall_seconds": (search_seconds or 0.0) + 0.05,
        "search_seconds": search_seconds,
        "total_found": len(scenarios) if status == "ok" else None,
        "scenarios": scenarios,
        "quality": quality,
    }


_ZERO_QUALITY = {
    "coverage": 0.0, "ordered_coverage": 0.0, "precision": 0.0, "coverage_top_k": 0.0,
}


# =====================================================================
#  (1) Гілка B справді отримує ВЕСЬ граф, а не звужений
# =====================================================================

def test_full_graph_scope_returns_every_node_and_every_edge_index():
    model = _model_with_island()
    scope = full_graph_scope(model)
    assert scope["node_ids"] == [node.id for node in model.nodes]
    assert scope["edge_ids"] == list(range(len(model.edges)))


def test_select_search_space_branch_a_excludes_islands_branch_b_includes_them():
    model = _model_with_island()
    space_a = select_search_space(model, "A")
    space_b = select_search_space(model, "B")

    all_node_ids = {node.id for node in model.nodes}
    assert set(space_b["node_ids"]) == all_node_ids, (
        "гілка B має отримати ВЕСЬ граф, а не звужений"
    )
    assert set(space_b["edge_ids"]) == set(range(len(model.edges)))

    # Острівець ISLAND_A/ISLAND_B не лежить на жодному шляху S0->ST, тож
    # узгодження (reconcile) його виключає з простору пошуку гілки A.
    assert "ISLAND_A" not in space_a["node_ids"]
    assert "ISLAND_B" not in space_a["node_ids"]
    assert len(space_a["node_ids"]) < len(space_b["node_ids"])


def test_select_search_space_rejects_unknown_branch_name():
    model = _model_with_island()
    try:
        select_search_space(model, "C")
    except ValueError:
        pass
    else:
        raise AssertionError("невідома гілка мала підняти ValueError")


def test_compute_branch_b_still_finds_the_real_path_despite_the_wider_space():
    """Ширший простір пошуку (увесь граф) не губить єдиний реальний шлях
    S0->ST — лише додає в конкурси зайву, недосяжну гілку (острівець)."""
    model = _model_with_island()
    result_a = compute_branch(model, "A")
    result_b = compute_branch(model, "B")
    assert [s["path"] for s in result_a["scenarios"]] == [["S0", "N1", "ST"]]
    assert [s["path"] for s in result_b["scenarios"]] == [["S0", "N1", "ST"]]


# =====================================================================
#  (2) Таймаут -> явний маркер, не 0 і не виключення; процес дійсно мертвий
# =====================================================================

def test_run_subprocess_with_deadline_returns_completed_true_for_a_quick_command():
    result = run_subprocess_with_deadline(
        [sys.executable, "-c", "print('ok')"], timeout_seconds=10.0,
    )
    assert result["completed"] is True
    assert result["returncode"] == 0
    assert "ok" in result["stdout"]


def test_run_subprocess_with_deadline_gives_an_explicit_marker_on_timeout():
    """Таймаут має дати ЯВНИЙ маркер "не завершилось" (`completed: False` у
    словнику) — не нуль (виглядало б як "0 шляхів знайдено") і не непіймане
    виключення (впало б увесь прогін на першій же завислій мережі)."""
    result = run_subprocess_with_deadline(
        [sys.executable, "-c", "import time; time.sleep(5)"], timeout_seconds=0.5,
    )
    assert isinstance(result, dict)
    assert result["completed"] is False
    assert result != 0


def test_run_subprocess_with_deadline_kills_the_process_and_returns_promptly():
    """Після kill() процес МАЄ бути дійсно мертвим (не висіти у фоні, не
    палити CPU) — `returncode is not None` після `communicate()` є прямим
    доказом цього (ОС-обіцянка `subprocess.Popen`: значення виставляється
    лише ПІСЛЯ фактичного завершення дочірнього процесу). Заодно перевіряємо,
    що функція не чекає повний sleep(5) — інакше "таймаут" був би фікцією."""
    started = time.monotonic()
    result = run_subprocess_with_deadline(
        [sys.executable, "-c", "import time; time.sleep(5)"], timeout_seconds=0.5,
    )
    wall = time.monotonic() - started
    assert wall < 3.0, "мало завершитись швидко (kill), а не чекати повний sleep(5)"
    assert result["returncode"] is not None


# =====================================================================
#  (3) Якість — лише на спільній підмножині завершених мереж
# =====================================================================

def test_common_completed_networks_keeps_only_networks_where_both_branches_finished():
    records = {
        "net-a": {
            "A": _fake_run("ok", 0.01, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
            "B": _fake_run("ok", 0.02, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
        },
        "net-b": {
            "A": _fake_run("ok", 0.01, [], _ZERO_QUALITY),
            "B": _fake_run("timeout"),
        },
        "net-c": {
            "A": _fake_run("ok", 0.01, [], _ZERO_QUALITY),
            "B": _fake_run("error"),
        },
    }
    assert common_completed_networks(records) == ["net-a"]


def test_score_branch_run_delegates_to_the_reused_metric_functions():
    """ЯВНА вимога задачі: coverage/ordered_coverage/precision/coverage_top_k
    рахуються ПЕРЕВИКОРИСТАНИМИ функціями `experiments.htb_reconstruction.
    metric`, а не власною формулою. Тест звіряє вихід `score_branch_run` з
    ПРЯМИМ викликом тих самих функцій на тих самих даних — а не з окремо
    порахованим вручну числом, яке саме собою нічого не довело б про
    перевикористання коду."""
    model = _model_with_vuln_chain()
    scenarios = [{"path": ["S0", "V1", "MID", "V2", "ST"], "edge_path": [0, 1, 2, 3]}]
    gold_cve_sets = [{"CVE-2020-0001"}, {"CVE-2020-0002"}]

    chain = path_cves(model, scenarios[0]["path"])
    expected = {
        "coverage": step_coverage(chain, gold_cve_sets),
        "ordered_coverage": ordered_coverage(chain, gold_cve_sets),
        "precision": precision_against_gold(chain, gold_cve_sets),
        "coverage_top_k": step_coverage(chain, gold_cve_sets),
    }
    got = score_branch_run(model, scenarios, gold_cve_sets)
    assert got == expected


def test_score_branch_run_on_no_scenarios_is_an_honest_zero_not_a_missing_key():
    """Недосяжна ціль -> чесний нуль (та сама конвенція, що й `evaluate_
    network` у `run_experiment.py`), а не відсутній ключ чи `None`."""
    model = _model_with_vuln_chain()
    got = score_branch_run(model, [], [{"CVE-2020-0001"}])
    assert got == _ZERO_QUALITY


def test_summarise_ablation_computes_quality_only_from_the_common_subset():
    """Три мережі: одна — обидві гілки завершились (її якість МАЄ увійти в
    середнє), дві інші — гілка B не завершилась (timeout/error; їхня якість
    НЕ повинна впливати на середнє, попри те що гілка A на них завершилась)."""
    common_quality = {
        "coverage": 0.8, "ordered_coverage": 0.6, "precision": 0.4, "coverage_top_k": 1.0,
    }
    records = {
        "net-ok": {
            "A": _fake_run("ok", 0.01, [{"path": ["S0", "ST"]}], common_quality),
            "B": _fake_run("ok", 0.02, [{"path": ["S0", "ST"]}], common_quality),
        },
        "net-timeout": {
            "A": _fake_run("ok", 0.01, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
            "B": _fake_run("timeout"),
        },
        "net-error": {
            "A": _fake_run("ok", 0.01, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
            "B": _fake_run("error"),
        },
    }
    summary = summarise_ablation(records)

    assert summary["networks_total"] == 3
    assert summary["branch_a"]["status_counts"] == {"ok": 3, "timeout": 0, "error": 0}
    assert summary["branch_b"]["status_counts"] == {"ok": 1, "timeout": 1, "error": 1}
    assert summary["common_completed_networks"] == ["net-ok"]
    assert summary["common_completed_count"] == 1
    # Одна мережа в підмножині -> середнє РІВНО дорівнює її власному числу.
    assert summary["quality_common_subset"]["A"]["coverage"]["mean"] == 0.8
    assert summary["quality_common_subset"]["B"]["precision"]["mean"] == 0.4
    assert summary["branch_b"]["timing"]["timed_out_networks"] == ["net-timeout"]


# =====================================================================
#  (4) Кількість знайдених шляхів і чи це ТІ САМІ шляхи (частка збігу)
# =====================================================================

def test_path_overlap_share_identical_sets_is_one():
    assert path_overlap_share(
        [["a", "b"], ["a", "c"]], [["a", "c"], ["a", "b"]],
    ) == 1.0


def test_path_overlap_share_disjoint_sets_is_zero():
    assert path_overlap_share([["a", "b"]], [["x", "y"]]) == 0.0


def test_path_overlap_share_partial_overlap_is_the_jaccard_index():
    # {P1,P2} проти {P1,P3}: перетин=1, обʼєднання=3 -> 1/3.
    got = path_overlap_share([["a", "b"], ["a", "c"]], [["a", "b"], ["a", "d"]])
    assert abs(got - 1 / 3) < 1e-9


def test_path_overlap_share_both_empty_is_agreement_not_zero():
    """Обидві гілки узгоджено НІЧОГО не знайшли — це збіг (1.0), а не 0%:
    0.0 означало б "жодного спільного шляху", хоча спільна відповідь тут —
    порожня множина в обох гілках."""
    assert path_overlap_share([], []) == 1.0


# =====================================================================
#  (5) Мережі, де B знайшла шлях, а A — ні (ціна звуження)
# =====================================================================

def test_b_only_reachable_networks_names_the_price_of_narrowing():
    records = {
        "net-a": {
            "A": _fake_run("ok", 0.01, [], _ZERO_QUALITY),
            "B": _fake_run("ok", 0.02, [{"path": ["S0", "N1", "ST"]}], _ZERO_QUALITY),
        },
        "net-b": {
            "A": _fake_run("ok", 0.01, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
            "B": _fake_run("ok", 0.02, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
        },
    }
    assert b_only_reachable_networks(records) == ["net-a"]
    assert a_only_reachable_networks(records) == []


def test_b_only_reachable_networks_is_empty_when_narrowing_never_costs_a_path():
    """Якщо жодна мережа не потрапляє в "B знайшла, A — ні" — список
    порожній, і звіт має чітко сказати "ціни немає", а не промовчати."""
    records = {
        "net-a": {
            "A": _fake_run("ok", 0.01, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
            "B": _fake_run("ok", 0.02, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
        },
    }
    assert b_only_reachable_networks(records) == []


# =====================================================================
#  Звіт: `ablation_reconcile.md` — жодне число не вписане руками
# =====================================================================

def test_render_ablation_report_changes_when_branch_b_completion_changes():
    """Той самий регрес-принцип, що й `test_readme_reflects_whether_defense_
    constraints_were_ever_violated` у `run_experiment.py`: текст МАЄ реально
    залежати від переданого `summary`, а не бути статичним шаблоном."""
    base_records = {
        "net-a": {
            "A": _fake_run("ok", 0.01, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
            "B": _fake_run("ok", 0.02, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
        },
    }
    summary_completed = summarise_ablation(base_records)
    timeout_records = {
        "net-a": {
            "A": _fake_run("ok", 0.01, [{"path": ["S0", "ST"]}], _ZERO_QUALITY),
            "B": _fake_run("timeout"),
        },
    }
    summary_timeout = summarise_ablation(timeout_records)
    config = {"budget_seconds": 10.0, "max_paths": 5, "dataset": "test"}

    text_completed = render_ablation_report(summary_completed, config)
    text_timeout = render_ablation_report(summary_timeout, config)
    assert text_completed != text_timeout
    assert "10.0" in text_completed  # бюджет має бути видимий у звіті


def test_render_ablation_report_states_explicitly_whether_narrowing_costs_paths():
    records_with_cost = {
        "net-a": {
            "A": _fake_run("ok", 0.01, [], _ZERO_QUALITY),
            "B": _fake_run("ok", 0.02, [{"path": ["S0", "N1", "ST"]}], _ZERO_QUALITY),
        },
    }
    summary = summarise_ablation(records_with_cost)
    config = {"budget_seconds": 10.0, "max_paths": 5, "dataset": "test"}
    text = render_ablation_report(summary, config)
    assert "net-a" in text


# =====================================================================
#  Інтеграція: воркер-CLI на РЕАЛЬНІЙ (маленькій, швидкій) мережі
# =====================================================================

def test_worker_cli_run_branch_a_on_a_real_network_prints_valid_json():
    """Гілка A навмисно швидка за конструкцією (контекст задачі: максимум
    40мс на мережу з 100) — саме тому лише вона тут ганяється проти
    реального датасету. Гілку B на реальних мережах юніт-тест НЕ запускає:
    її час непередбачуваний (аж до "не завершилось за 24 хвилини"), і саме
    для цього потрібен окремий процес з жорстким таймаутом, а не місце в
    швидкому тестовому наборі."""
    result = subprocess.run(
        [
            sys.executable, _SCRIPT_PATH,
            "--run-branch", "net-001", "A",
            "--dataset", _DATASET_DIR,
        ],
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["network"] == "net-001"
    assert payload["branch"] == "A"
    assert payload["search_seconds"] < 5.0
    assert set(payload["quality"]) == {
        "coverage", "ordered_coverage", "precision", "coverage_top_k",
    }


if __name__ == "__main__":
    import pathlib
    import tempfile
    import traceback

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
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
