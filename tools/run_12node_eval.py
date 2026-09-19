"""Прогін експерименту: 5 сценаріїв захисту на корпоративній 12-вузловій
моделі (`qa/fixtures/corporate_12node.yaml`), оцінені Hermes3 за 7
критеріями фікстури - двічі: при temperature=0 і temperature=0.7, по три
незалежні повтори на кожну температуру.

НЕ ТЕСТ. Це окремий скрипт (`python tools/run_12node_eval.py`), а не
`tests/test_*.py`: він ходить у РЕАЛЬНИЙ Hermes API
(`.env`: HERMES_API_KEY / HERMES_BASE_URL / HERMES_MODEL), і результат -
самозвіт живої моделі, не детермінований і не верифікований зовні. Юніт-
тести логіки живуть окремо (`tests/test_criterion_eval.py`) і працюють на
фальшивому бекенді - жоден з них мережі не торкається.

ЧОМУ ТРИЧІ НА ТЕМПЕРАТУРУ ОКРЕМИМИ ПРОГОНАМИ, А НЕ `repeats=3` ВСЕРЕДИНІ
ОДНОГО ВИКЛИКУ. `qa.criterion_eval.evaluate_scenarios(..., repeats=N)`
зливає N вимірів ОДНОГО критерію в ОДИН голос (медіана) - це вбудований
механізм стійкості одного вимірювання, і дає ОДИН J. Тут вимірюється інша
величина: наскільки розходяться між собою РІЗНІ незалежні прогони - тобто
потрібні три ОКРЕМІ J, а не один змішаний. Тому кожен прогін викликає
`evaluate_scenarios(..., repeats=1)`, а стандартне відхилення J рахується
ЗОВНІ, по трьох прогонах.

СЕКРЕТ. HERMES_API_KEY читається лише з оточення (`.env` через
`python-dotenv`) і передається в конструктор `HermesBackend`. Він НІКОЛИ не
потрапляє у вивід, лог чи файли результатів - усе, що йде в stdout чи на
диск, це URL без ключа, назва моделі та відповіді LLM.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from qa import twin  # noqa: E402
from qa.criterion_eval import evaluate_scenarios  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "qa" / "fixtures" / "corporate_12node.yaml"
RESULTS_DIR = ROOT / "experiments" / "corporate_12node" / "results"

# "виділено п'ять альтернативних сценаріїв" - буквальне число з чернетки
# наукової роботи, не довільний вибір.
MAX_SCENARIOS = 5
TEMPERATURES = (0.0, 0.7)
RUNS_PER_TEMPERATURE = 3
REQUEST_MAX_TOKENS = 200
# Контекст моделі - 4096 токенів і черга запитів (`OLLAMA_NUM_PARALLEL=1`) -
# помірний таймаут не дає одному завислому запиту з'їсти бюджет часу всього
# прогону, але лишає запас на реальні ~3с/запит з чернетки задачі.
REQUEST_TIMEOUT_SECONDS = 45.0


class HermesBackend:
    """Реальна реалізація `qa.criterion_eval.CriterionBackend` - OpenAI-
    сумісний `/chat/completions` на self-hosted Hermes. Сертифікат
    самопідписаний, звідси `verify=False` (той самий компроміс, що і в
    `.env`-описі задачі: приватний бокс, не публічний сервіс). Ключ - лише
    параметр конструктора, що прийшов з оточення в `main()`."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(
            verify=False,
            timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10.0),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )

    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Один запит chat/completions з ретраями на 429/503 (той самий
        прийом, що й `shipper/extractor.py::_post_chat`, лише синхронно)."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": REQUEST_MAX_TOKENS,
            "stream": False,
        }
        delay = 2.0
        response = None
        for attempt in range(4):
            response = self._client.post(f"{self._base_url}/chat/completions", json=payload)
            if response.status_code not in (429, 503):
                break
            retry_after = response.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else delay
            print(
                f"    [rate limited] status={response.status_code} wait={wait:.1f}s "
                f"attempt={attempt + 1}"
            )
            time.sleep(wait)
            delay = min(delay * 2, 30.0)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or [{}]
        content = (choices[0].get("message") or {}).get("content")
        return content or ""

    def close(self) -> None:
        self._client.close()


def load_scenarios() -> tuple[twin.TwinModel, list[dict]]:
    """Парсить фікстуру й виділяє рівно `MAX_SCENARIOS` допустимих шляхів
    S0->ST тим самим детермінованим кодом, що й бойовий шлях UI
    (`expand_forward` / `expand_backward` / `reconcile` / `enumerate_scenarios`).

    `matched=set()` навмисно: тут немає RAG-контексту з реальних документів
    (`qa.twin.actualize` не викликається) - оцінюються шляхи самого двійника,
    без семантичного покриття доказами."""
    model = twin.parse_twin_model(FIXTURE_PATH.read_text(encoding="utf-8"))
    forward = twin.expand_forward(model)
    backward = twin.expand_backward(model)
    gc = twin.reconcile(model, forward, backward)
    found = twin.enumerate_scenarios(model, gc, set(), max_paths=MAX_SCENARIOS)
    return model, found["scenarios"]


def scenario_label(model: twin.TwinModel, scenario: dict) -> str:
    """Короткий людський підпис сценарію для таблиць/логів - мітки заходів,
    не внутрішні id."""
    labels = [
        model.node_by_id[node_id].label
        for node_id in scenario["path"]
        if model.node_by_id[node_id].kind == "control"
    ]
    return " -> ".join(labels) if labels else "(без заходів)"


def build_summary(
    model: twin.TwinModel, scenarios: list[dict], runs: dict[float, list[dict]]
) -> dict:
    """Зводить сирі прогони (`runs[temperature] = [evaluate_scenarios(...), ...]`)
    у таблицю на сценарій: J кожного прогону, середнє, стандартне відхилення
    між прогонами, статус обмежень, ранг за середнім J на кожній температурі."""
    summary_scenarios = []
    for scenario_index, scenario in enumerate(scenarios):
        within = twin.within_constraints(model, scenario["path"], scenario["edge_path"])
        by_temperature: dict[str, dict] = {}
        for temperature, run_results in runs.items():
            j_values: list[float] = []
            unresolved_seen: list[str] = []
            criteria_scores_by_run = []
            for run_result in run_results:
                record = run_result["scenarios"][scenario_index]
                if record["J"] is not None:
                    j_values.append(record["J"])
                unresolved_seen.extend(record["unresolved_criteria"])
                criteria_scores_by_run.append(
                    {
                        name: (entry["merged"]["score"] if entry["merged"] else None)
                        for name, entry in record["criteria"].items()
                    }
                )
            by_temperature[str(temperature)] = {
                "j_values": j_values,
                "j_mean": round(statistics.fmean(j_values), 4) if j_values else None,
                # pstdev, а не sample stdev - рахуємо розкид САМЕ ЦИХ
                # прогонів, а не оцінюємо розкид ширшої гіпотетичної
                # популяції (та сама конвенція, що й `qa.twin.aggregate`).
                "j_stdev": round(statistics.pstdev(j_values), 4) if len(j_values) > 1 else None,
                "unresolved_criteria_seen": sorted(set(unresolved_seen)),
                "criteria_scores_by_run": criteria_scores_by_run,
                "runs_with_valid_j": len(j_values),
                "runs_total": len(run_results),
            }
        summary_scenarios.append(
            {
                "index": scenario_index,
                "label": scenario_label(model, scenario),
                "path_labels": [model.node_by_id[n].label for n in scenario["path"]],
                "within_constraints": within,
                "by_temperature": by_temperature,
            }
        )

    # Ранг за середнім J на кожній температурі окремо (вища J = кращий
    # сценарій захисту; сценарій без жодного валідного J - завжди останній).
    for temperature in runs:
        temp_key = str(temperature)
        ordered = sorted(
            summary_scenarios,
            key=lambda s: (
                s["by_temperature"][temp_key]["j_mean"] is None,
                -(s["by_temperature"][temp_key]["j_mean"] or 0.0),
            ),
        )
        for rank, scenario_summary in enumerate(ordered, start=1):
            scenario_summary["by_temperature"][temp_key]["rank"] = rank

    return {
        "model": model.name,
        "criteria": [c.name for c in model.criteria],
        "max_scenarios": MAX_SCENARIOS,
        "runs_per_temperature": RUNS_PER_TEMPERATURE,
        "temperatures": list(TEMPERATURES),
        "scenarios": summary_scenarios,
    }


def print_summary_table(summary: dict) -> None:
    print("\n=== Зведення ===")
    for scenario in summary["scenarios"]:
        print(f"\nСценарій {scenario['index']}: {scenario['label']}")
        print(f"  within_constraints: {scenario['within_constraints']}")
        for temperature, stats in scenario["by_temperature"].items():
            print(
                f"  T={temperature}: J={stats['j_values']} "
                f"mean={stats['j_mean']} stdev={stats['j_stdev']} "
                f"rank={stats.get('rank')} "
                f"(валідних J: {stats['runs_with_valid_j']}/{stats['runs_total']}), "
                f"unresolved={stats['unresolved_criteria_seen']}"
            )


def main() -> None:
    api_key = os.environ.get("HERMES_API_KEY", "")
    base_url = os.environ.get("HERMES_BASE_URL", "")
    model_name = os.environ.get("HERMES_MODEL", "")
    if not (api_key and base_url and model_name):
        print(
            "Немає HERMES_API_KEY / HERMES_BASE_URL / HERMES_MODEL в оточенні "
            "(перевір .env) - зупиняюсь.",
            file=sys.stderr,
        )
        sys.exit(2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    model, scenarios = load_scenarios()
    print(f"Модель: {model.name}, виділено {len(scenarios)} сценаріїв (max_paths={MAX_SCENARIOS}).")
    for index, scenario in enumerate(scenarios):
        print(f"  [{index}] {scenario_label(model, scenario)}")

    backend = HermesBackend(base_url=base_url, api_key=api_key, model=model_name)
    try:
        runs: dict[float, list[dict]] = {temperature: [] for temperature in TEMPERATURES}
        for temperature in TEMPERATURES:
            for run_index in range(RUNS_PER_TEMPERATURE):
                print(f"--- temperature={temperature} run={run_index + 1}/{RUNS_PER_TEMPERATURE} ---")
                started = time.monotonic()
                result = evaluate_scenarios(model, scenarios, backend, repeats=1, temperature=temperature)
                elapsed = time.monotonic() - started
                print(f"    завершено за {elapsed:.1f}с")
                runs[temperature].append(result)
                # Пишемо сирі дані одразу після КОЖНОГО прогону, а не в кінці -
                # доказова база (сирі відповіді моделі) не має залежати від
                # того, чи доживе процес до останнього прогону.
                raw_path = (
                    RESULTS_DIR
                    / f"raw_{run_started}_t{temperature}_run{run_index + 1}.json"
                )
                raw_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    finally:
        backend.close()

    summary = build_summary(model, scenarios, runs)
    summary_path = RESULTS_DIR / f"summary_{run_started}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nЗведення записано у {summary_path}")
    print_summary_table(summary)


if __name__ == "__main__":
    main()
