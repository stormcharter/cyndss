"""qa/criterion_eval.py: оцінка сценаріїв захисту мовною моделлю за критеріями.

НАВІЩО ОКРЕМИЙ ФАЙЛ. `test_twin_model.py`/`test_twin_defense.py` перевіряють
детермінований шар методу (перебір шляхів, агрегація J). Тут - шар, що дає
`qa.twin.aggregate` числа замість порожнього списку: текстовий опис
сценарію -> запит до LLM за одним критерієм -> стійкий парсинг JSON.
Жоден тест не ходить у мережу - усюди підставляється `FakeBackend` з
наперед заданою чергою відповідей.

Запуск:
    python tests/test_criterion_eval.py
або pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from qa import twin  # noqa: E402
from qa.criterion_eval import (  # noqa: E402
    describe_scenario,
    evaluate_scenarios,
    score_criterion,
)


class FakeBackend:
    """Тестовий бекенд: черга наперед заданих "відповідей моделі".

    Жодного HTTP-виклику тут немає - `generate()` просто повертає наступний
    рядок з черги і записує аргументи виклику для перевірок. Якщо чергу
    вичерпано, а код просить ще одну відповідь - це помилка тесту (не мовчазне
    None), тому кидаємо AssertionError."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str, float]] = []

    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        self.calls.append((system_prompt, user_prompt, temperature))
        if not self._responses:
            raise AssertionError("FakeBackend: черга відповідей вичерпана")
        return self._responses.pop(0)


CRITERION = twin.TwinCriterion(
    name="technical_feasibility",
    description="How technically feasible is implementing this control?",
    weight=0.2,
    direction="benefit",
)


# --- score_criterion: стійкий парсинг -----------------------------------


def test_parses_clean_json():
    """Модель повернула валідний JSON без нічого зайвого навколо - парситься
    з першої спроби, ретраїв не було."""
    backend = FakeBackend(
        ['{"score": 0.7, "confidence": 0.9, "evidence_state": "sufficient", "rationale": "ok"}']
    )
    result = score_criterion("mock scenario text", CRITERION, backend)
    assert result["score"] == pytest.approx(0.7)
    assert result["confidence"] == pytest.approx(0.9)
    assert result["evidence_state"] == "sufficient"
    assert result["rationale"] == "ok"
    assert "parse_failed" not in result
    assert len(backend.calls) == 1


def test_parses_json_wrapped_in_surrounding_text():
    """Модель обгорнула JSON текстом ('Sure, here is...') - парсер мусить
    витягти сам об'єкт, а не впасти чи вимагати рівно чистого JSON."""
    raw = (
        "Sure, here is my evaluation:\n"
        '{"score": 0.4, "confidence": 0.5, "evidence_state": "insufficient", '
        '"rationale": "not enough data"}\n'
        "Hope this helps!"
    )
    backend = FakeBackend([raw])
    result = score_criterion("mock scenario text", CRITERION, backend)
    assert result["score"] == pytest.approx(0.4)
    assert result["evidence_state"] == "insufficient"
    assert result["rationale"] == "not enough data"


def test_broken_json_after_three_attempts_is_an_explicit_failure():
    """Три спроби поспіль - і жодна не розбирається. Це має дати ЯВНИЙ маркер
    невдачі (score=None, parse_failed=True), а НЕ тихо підставлений 0 чи
    середнє - інакше провал моделі виглядав би як реальна низька оцінка."""
    backend = FakeBackend(["not json at all", "{score: 0.5 (no quotes)", "still no json here"])
    result = score_criterion("mock scenario text", CRITERION, backend)
    assert result["score"] is None
    assert result["parse_failed"] is True
    assert result["evidence_state"] == "parse_failed"
    assert len(backend.calls) == 3
    assert len(result["raw_responses"]) == 3


def test_out_of_range_score_above_one_is_clipped_and_flagged():
    backend = FakeBackend(
        ['{"score": 1.4, "confidence": 0.8, "evidence_state": "sufficient", "rationale": "great"}']
    )
    result = score_criterion("mock scenario text", CRITERION, backend)
    assert result["score"] == pytest.approx(1.0)
    assert result["clipped"] is True


def test_out_of_range_score_below_zero_is_clipped_and_flagged():
    backend = FakeBackend(
        ['{"score": -0.3, "confidence": 0.8, "evidence_state": "sufficient", "rationale": "bad"}']
    )
    result = score_criterion("mock scenario text", CRITERION, backend)
    assert result["score"] == pytest.approx(0.0)
    assert result["clipped"] is True


def test_third_attempt_succeeding_still_returns_a_valid_result():
    """Перші дві спроби - сміття, третя - валідний JSON: маємо отримати
    успішний результат (ретраї існують саме для цього випадку)."""
    backend = FakeBackend(
        [
            "garbage",
            "more garbage",
            '{"score": 0.55, "confidence": 0.6, "evidence_state": "sufficient", "rationale": "recovered"}',
        ]
    )
    result = score_criterion("mock scenario text", CRITERION, backend)
    assert result["score"] == pytest.approx(0.55)
    assert "parse_failed" not in result
    assert len(backend.calls) == 3


# --- describe_scenario: компактний опис, без внутрішніх id --------------

_MINI_MODEL_YAML = """
name: mini
current_state: s0
goal_state: st
nodes:
  - {id: s0, kind: state, label: "S0: Start"}
  - {id: mid, kind: state, label: "Perimeter Hardened"}
  - {id: st, kind: state, label: "ST: Secured"}
  - {id: ctrl-fw, kind: control, label: "Firewall Rules", cost: 2000, personnel: 1}
edges:
  - {source: s0, target: ctrl-fw, relation: applies}
  - {source: ctrl-fw, target: mid, relation: enables}
  - {source: mid, target: st, relation: enables}
criteria:
  - {name: risk, weight: 1.0, description: "residual risk"}
"""


def _mini_scenario() -> tuple[twin.TwinModel, list[str], list[int]]:
    """Модель з одним допустимим шляхом s0 -> ctrl-fw -> mid -> st, через
    той самий перебір, що й бойовий код (а не вручну підібрані індекси)."""
    model = twin.parse_twin_model(_MINI_MODEL_YAML)
    forward = twin.expand_forward(model)
    backward = twin.expand_backward(model)
    gc = twin.reconcile(model, forward, backward)
    found = twin.enumerate_scenarios(model, gc, set())
    scenario = found["scenarios"][0]
    return model, scenario["path"], scenario["edge_path"]


def test_describe_scenario_contains_measure_labels_not_ids():
    model, path, edge_path = _mini_scenario()
    text = describe_scenario(model, path, edge_path)
    assert "Firewall Rules" in text
    assert "ctrl-fw" not in text
    assert "s0" not in text.split()  # id "s0" не має з'являтись як токен
    assert "state-vulnerable" not in text


def test_describe_scenario_includes_measure_attributes():
    model, path, edge_path = _mini_scenario()
    text = describe_scenario(model, path, edge_path)
    assert "cost=2000" in text
    assert "personnel=1" in text


def test_describe_scenario_lists_traversed_states():
    model, path, edge_path = _mini_scenario()
    text = describe_scenario(model, path, edge_path)
    assert "S0: Start" in text
    assert "Perimeter Hardened" in text
    assert "ST: Secured" in text


# --- evaluate_scenarios: агрегація узгоджена з qa.twin.aggregate --------

_TWO_CRITERIA_MODEL_YAML = """
name: two-criteria
current_state: s0
goal_state: st
nodes:
  - {id: s0, kind: state, label: "S0: Start"}
  - {id: st, kind: state, label: "ST: Secured"}
criteria:
  - {name: safety, weight: 1.0, description: "higher is better", direction: benefit}
  - {name: time, weight: 1.0, description: "how long it takes", direction: cost}
edges:
  - {source: s0, target: st, relation: enables}
"""


def _two_criteria_scenario():
    model = twin.parse_twin_model(_TWO_CRITERIA_MODEL_YAML)
    forward = twin.expand_forward(model)
    backward = twin.expand_backward(model)
    gc = twin.reconcile(model, forward, backward)
    scenario = twin.enumerate_scenarios(model, gc, set())["scenarios"][0]
    return model, scenario


def test_evaluate_scenarios_j_matches_manual_aggregate_call():
    """Ручний приклад: два критерії, одна оцінка на кожен (repeats=1). J,
    порахований `evaluate_scenarios`, має збігатися з прямим викликом
    `qa.twin.aggregate` на тих самих числах - модуль НЕ повинен рахувати
    інтеграл по-своєму вдруге, а зобов'язаний перевикористати наявну функцію."""
    model, scenario = _two_criteria_scenario()
    backend = FakeBackend(
        [
            '{"score": 0.9, "confidence": 0.9, "evidence_state": "sufficient", "rationale": "r1"}',
            '{"score": 0.3, "confidence": 0.8, "evidence_state": "sufficient", "rationale": "r2"}',
        ]
    )
    result = evaluate_scenarios(model, [scenario], backend, repeats=1, temperature=0.0)
    record = result["scenarios"][0]

    expected_scores = [
        {"criterion": "safety", "score": 0.9},
        {"criterion": "time", "score": 0.3},
    ]
    expected_j, expected_disagreement = twin.aggregate(expected_scores, model.criteria)

    assert record["J"] == pytest.approx(expected_j)
    assert record["disagreement"] == pytest.approx(expected_disagreement)
    # cost-критерій "time" мав інвертуватись усередині aggregate - перевірка,
    # що інтегральна оцінка не дорівнює наївному середньому сирих балів.
    assert record["J"] != pytest.approx((0.9 + 0.3) / 2)


def test_evaluate_scenarios_reports_unresolved_criterion_without_faking_a_score():
    """Якщо ВСІ повтори одного критерію не розібрались - сценарій не має
    отримати оманливий J, ніби нічого не сталось. J=None і критерій
    перелічений у unresolved_criteria - явно, а не мовчки нулем/середнім."""
    model, scenario = _two_criteria_scenario()
    backend = FakeBackend(
        [
            '{"score": 0.9, "confidence": 0.9, "evidence_state": "sufficient", "rationale": "r1"}',
            "garbage", "garbage", "garbage",  # "time" - три невдалі спроби
        ]
    )
    result = evaluate_scenarios(model, [scenario], backend, repeats=1, temperature=0.0)
    record = result["scenarios"][0]
    assert record["J"] is None
    assert "time" in record["unresolved_criteria"]


def test_repeats_collects_multiple_measurements_per_criterion():
    """repeats=3 має дати три сирі виміри на критерій (а не один, помножений
    на три виклики і мовчки відкинутий)."""
    model = twin.parse_twin_model(
        """
name: one-criterion
current_state: s0
goal_state: st
nodes:
  - {id: s0, kind: state, label: "S0: Start"}
  - {id: st, kind: state, label: "ST: Secured"}
criteria:
  - {name: safety, weight: 1.0, description: "higher is better"}
edges:
  - {source: s0, target: st, relation: enables}
"""
    )
    forward = twin.expand_forward(model)
    backward = twin.expand_backward(model)
    gc = twin.reconcile(model, forward, backward)
    scenario = twin.enumerate_scenarios(model, gc, set())["scenarios"][0]

    backend = FakeBackend(
        [
            '{"score": 0.2, "confidence": 0.5, "evidence_state": "sufficient", "rationale": "a"}',
            '{"score": 0.4, "confidence": 0.6, "evidence_state": "sufficient", "rationale": "b"}',
            '{"score": 0.6, "confidence": 0.7, "evidence_state": "sufficient", "rationale": "c"}',
        ]
    )
    result = evaluate_scenarios(model, [scenario], backend, repeats=3, temperature=0.0)
    breakdown = result["scenarios"][0]["criteria"]["safety"]
    assert len(breakdown["raw_votes"]) == 3
    assert [vote["score"] for vote in breakdown["raw_votes"]] == [
        pytest.approx(0.2), pytest.approx(0.4), pytest.approx(0.6),
    ]
    # медіана трьох вимірів (qa.twin.merge_panel_votes), не перший і не останній
    assert breakdown["merged"]["score"] == pytest.approx(0.4)
    assert len(backend.calls) == 3


# --- ручний запуск без pytest --------------------------------------------
if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
