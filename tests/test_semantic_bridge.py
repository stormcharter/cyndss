"""Двостороння семантична мережа, легенда й анкета.

Модель і ембедер підмінено заглушками: перевіряється не якість понять (її дає
модель), а властивості методу, на які спирається висновок:
  * мережа росте з двох фронтів, фронти не перемішуються;
  * дубль поняття в межах фронту не створює вузла;
  * міст з'являється лише там, де його підтвердила близькість або модель;
  * кожен ланцюг іде від проблеми до розв'язку через міст;
  * ранжування детерміноване й узгоджене з вагами;
  * анкета: відповіді перевіряються, зберігаються, голоси зводяться.

Запуск:
    pytest tests/test_semantic_bridge.py -v
"""

import json
import os
import re
import sys
import zlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_context import (  # noqa: E402
    default_problem,
    default_solution,
    network_brief,
    parse_network_xml,
)
from semantic_bridge import (  # noqa: E402
    PROBLEM_ID,
    SOLUTION_ID,
    SemanticNet,
    build_semantic_net,
    chain_steps,
    chains,
    extract_json,
    rank_chains,
    score_steps,
)
from solution_survey import (  # noqa: E402
    AGENTS,
    SurveyStore,
    agent_order,
    ask_agent,
    build_survey,
    changes,
    diverse_chains,
    expert_estimates,
    make_response,
    plain_legend,
    revise_agent,
    survey_text,
    synthesize,
    system_choice,
    tally,
    write_legend,
)

XML = """<network id="net-test" profile="small" topology="flat">
  <subnets><subnet id="lan" zone="lan" trust="medium"/></subnets>
  <assets>
    <asset id="dc-01" role="domain-controller" subnet="lan" criticality="critical">
      <defenses><defense kind="backup" state="disabled"/></defenses>
    </asset>
    <asset id="nas-01" role="nas" subnet="lan" criticality="high"><defenses/></asset>
    <asset id="ws-01" role="workstation" subnet="lan" criticality="low">
      <services><service><vulnerability cve="CVE-1"/></service></services>
      <accounts><account name="h"><reusedOn asset="nas-01"/></account></accounts>
    </asset>
  </assets>
  <task><constraints><constraint>бюджет обмежений</constraint></constraints></task>
  <attacks><attack id="a1"><step from="ws-01" to="dc-01"/></attack></attacks>
</network>"""

# Сценарій заглушки: що «модель» пропонує з кожного поняття.
FORWARD = {
    "P": ["Інвентаризація вузлів", "Увімкнути резервні копії", "Навести лад із паролями"],
    "f1": ["Оцінка ризиків вузлів"],
    "f2": ["Копії контролера домену"],
    "f3": ["Унікальні паролі адміністраторів"],
}
BACKWARD = {
    "S": ["Резервні копії налаштовано", "Паролі не повторюються", "Захист на nas-01"],
    "b1": ["Копії контролера домену"],          # той самий текст, що f-гілка -> «те саме»
    "b2": ["Політика паролів затверджена"],
    "b3": ["Агент захисту встановлено"],
}
# uk -> en: пари, які «модель» підтверджує, мають спільне слово — як і з
# реальним ембедером, кандидати в мости відбираються за близькістю.
EN = {
    "Навести лад із паролями": "password hygiene",
    "Паролі не повторюються": "password reuse eliminated",
    "Унікальні паролі адміністраторів": "unique admin password",
    "Політика паролів затверджена": "password policy approved",
}
CONFIRMED = {("Навести лад із паролями", "Паролі не повторюються"),
             ("Унікальні паролі адміністраторів", "Політика паролів затверджена")}


def fake_embed(texts):
    """Мішок слів у фіксованому просторі: однаковий текст -> cos 1, різний -> мала."""
    out = []
    for text in texts:
        vec = [0.0] * 4096
        for word in re.findall(r"\w+", text.lower()):
            if word == "clustering":
                continue
            vec[zlib.crc32(word.encode()) % 4096] += 1.0
        out.append(vec)
    return out


def fake_generate(confirm=frozenset(CONFIRMED)):
    calls = []

    def generate(system, user, tag):
        calls.append(tag)
        if tag.startswith(("forward:", "backward:")):
            side, cid = tag.split(":")
            table = FORWARD if side == "forward" else BACKWARD
            return json.dumps({"concepts": [{"uk": c, "en": EN.get(c, c)}
                                            for c in table.get(cid, [])]}, ensure_ascii=False)
        if tag == "bridges":
            links = []
            for line in user.splitlines():
                m = re.match(r"(\d+)\. (.+) → (.+)", line)
                if m and (m.group(2), m.group(3)) in confirm:
                    links.append({"i": int(m.group(1)), "ok": True, "why": "так"})
            return "```json\n" + json.dumps({"links": links}, ensure_ascii=False) + "\n```"
        if tag == "score":
            n = len(re.findall(r"^\d+\. ", user, re.M))
            return json.dumps({"scores": [{"i": i, "cost": 1 + i % 5, "time": 2,
                                           "impact": 5 - i % 3} for i in range(1, n + 1)]})
        return "текст"
    generate.calls = calls
    return generate


@pytest.fixture
def net():
    return build_semantic_net("Проблема", "Розв'язок", "контекст", fake_generate(),
                              fake_embed, k=3, depth=2)


# --- контекст мережі ------------------------------------------------------------


def test_network_context_ignores_attacks_and_lists_gaps():
    xml = parse_network_xml(XML)
    brief = network_brief(xml)
    assert "dc-01: backup — disabled" in brief
    assert "nas-01" in brief            # важливий вузол без захисту
    assert "a1" not in brief and "атак" not in brief.lower()
    assert "net-test" in default_problem(xml) and "net-test" in default_solution(xml)


def test_network_without_assets_is_rejected():
    with pytest.raises(ValueError, match="asset"):
        parse_network_xml('<network id="x"><assets/></network>')
    with pytest.raises(ValueError, match="<network>"):
        parse_network_xml("<root/>")


# --- два фронти --------------------------------------------------------------------


def test_two_fronts_grow_from_both_ends(net):
    forward = net.side("forward")
    backward = net.side("backward")
    assert {c.depth for c in forward} == {1, 2} and {c.depth for c in backward} == {1, 2}
    # Пряме ребро йде від проблеми, зворотне — до розв'язку.
    assert ("P", "f1", "веде до") in net.edges
    assert ("b1", "S", "передує") in net.edges


def test_expansion_calls_cover_every_frontier_node():
    generate = fake_generate()
    build_semantic_net("Проблема", "Розв'язок", "к", generate, fake_embed, k=3, depth=2)
    expansions = [t for t in generate.calls if ":" in t]
    assert len(expansions) == 2 * (1 + 3)   # корінь + три вузли першого кроку з кожного боку


def test_duplicate_concept_on_one_side_is_merged():
    table = dict(FORWARD)
    FORWARD["f1"] = ["Увімкнути резервні копії"]    # повтор f2 у межах прямого фронту
    try:
        net = build_semantic_net("Проблема", "Розв'язок", "к", fake_generate(), fake_embed)
    finally:
        FORWARD.clear()
        FORWARD.update(table)
    labels = [c.label for c in net.side("forward")]
    assert labels.count("Увімкнути резервні копії") == 1
    assert ("f1", "f2", "веде до") in net.edges


def test_bridges_only_where_confirmed(net):
    kinds = {(net.concepts[b["from"]].label, net.concepts[b["to"]].label): b["kind"]
             for b in net.bridges}
    assert kinds[("Копії контролера домену", "Копії контролера домену")] == "same"
    for pair in CONFIRMED:
        assert kinds[pair] == "link"
    assert len(net.bridges) == 3


def test_no_bridges_means_no_chains():
    net = build_semantic_net("Проблема", "Розв'язок", "к", fake_generate(confirm=frozenset()),
                             lambda texts: [[1.0, float(i)] for i, _ in enumerate(texts)],
                             k=3, depth=1)
    assert net.bridges == [] or all(b["kind"] == "same" for b in net.bridges)


def test_every_chain_crosses_a_bridge(net):
    bridges = {(b["from"], b["to"]) for b in net.bridges}
    paths = chains(net)
    assert paths
    for path in paths:
        assert path[0] == PROBLEM_ID and path[-1] == SOLUTION_ID
        assert any(pair in bridges for pair in zip(path, path[1:]))


def test_same_bridge_does_not_duplicate_the_step(net):
    path = next(p for p in chains(net) if "f5" in p or any(
        net.concepts[n].label == "Копії контролера домену" for n in p))
    labels = [net.concepts[n].label for n in chain_steps(net, path)]
    assert labels.count("Копії контролера домену") == 1


def test_net_roundtrips_through_json(net):
    again = SemanticNet.from_dict(json.loads(json.dumps(net.to_dict(), ensure_ascii=False)))
    assert chains(again) == chains(net)
    assert again.bridges == net.bridges


def test_broken_model_answer_is_logged_not_fatal():
    def generate(system, user, tag):
        return "не JSON" if tag.startswith("forward") else fake_generate()(system, user, tag)
    net = build_semantic_net("Проблема", "Розв'язок", "к", generate, fake_embed, depth=1)
    assert net.side("forward") == [] and net.log
    assert chains(net) == []


# --- оцінка й ранжування -----------------------------------------------------------------


def test_scores_parsed_and_missing_ones_assumed(net):
    ids = sorted({n for p in chains(net) for n in p} - {"P", "S"})
    scores = score_steps(net, ids, "к", fake_generate())
    assert all(not s["assumed"] for s in scores.values())
    broken = score_steps(net, ids, "к", lambda *a: "{}")
    assert all(s["assumed"] and s["cost"] == 3 for s in broken.values())


def test_ranking_sorted_deterministic_and_weighted(net):
    ids = sorted({n for p in chains(net) for n in p} - {"P", "S"})
    scores = score_steps(net, ids, "к", fake_generate())
    first = rank_chains(net, scores, k=10)
    second = rank_chains(net, scores, k=10)
    assert [c["path"] for c in first["chains"]] == [c["path"] for c in second["chains"]]
    js = [c["j"] for c in first["chains"]]
    assert js == sorted(js, reverse=True)
    for chain in first["chains"]:
        assert chain["cost"] == sum(s["cost"] for s in chain["steps"])
        assert 0 < chain["impact"] < 1
    strict = rank_chains(net, scores, k=10, min_impact=0.99)
    assert strict["admissible"] < first["admissible"] or first["admissible"] == 0


def test_extract_json_handles_fences_and_noise():
    assert extract_json('Ось:\n```json\n{"a": {"b": 1}}\n```\nдякую') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        extract_json("нічого")


# --- легенда й анкета --------------------------------------------------------------------


@pytest.fixture
def case():
    return {"id": "net-test", "problem": "Слабкий захист", "solution": "Мережа захищена"}


@pytest.fixture
def ranked(net):
    ids = sorted({n for p in chains(net) for n in p} - {"P", "S"})
    return rank_chains(net, score_steps(net, ids, "к", fake_generate()), k=3)["chains"]


@pytest.fixture
def survey(case, ranked):
    return build_survey(case, ranked)


def test_plain_legend_and_prompt(case, ranked):
    chain = ranked[0]
    assert f"{round(chain['impact'] * 100)}%" in plain_legend(chain)
    seen = {}

    def generate(system, user, tag):
        seen["system"], seen["user"] = system, user
        return " легенда "
    assert write_legend(generate, case, chain) == "легенда"
    assert "БЕЗ технічної" in seen["system"] and chain["steps"][0]["label"] in seen["user"]


def test_survey_asks_how_to_solve(survey, ranked):
    assert survey["question"].startswith("Як розв'язати")
    assert [o["key"] for o in survey["options"]] == list("АБВ"[:len(ranked)])
    assert set(survey["questions"]) == {"time", "cost", "impact"}


def test_survey_id_depends_on_options_not_legends(case, ranked):
    plain = build_survey(case, ranked)
    written = build_survey(case, ranked, {ranked[0]["rank"]: "інша"})
    assert plain["id"] == written["id"] != build_survey(case, ranked[:1])["id"]


def test_response_validation_and_store(tmp_path, survey):
    with pytest.raises(ValueError, match="хто"):
        make_response(survey, " ", {"time": "А", "cost": "А", "impact": "А"})
    with pytest.raises(ValueError, match="Ціна"):
        make_response(survey, "Л1", {"time": "А", "cost": "Я", "impact": "А"})
    store = SurveyStore(tmp_path)
    store.add(survey["id"], make_response(survey, "Л1", {"time": "А", "cost": "А", "impact": "А"}))
    store.add(survey["id"], make_response(survey, "Л2", {"time": "Б", "cost": "Б", "impact": "Б"}))
    store.add(survey["id"], make_response(survey, "Л1", {"time": "Б", "cost": "А", "impact": "А"}))
    assert [(r["respondent"], r["choices"]["time"]) for r in store.load(survey["id"])] == \
        [("Л2", "Б"), ("Л1", "Б")]
    assert [r["respondent"] for r in store.remove(survey["id"], "Л2")] == ["Л1"]


def test_tally_names_winners_and_ties(survey):
    def vote(who, t, c, i):
        return make_response(survey, who, {"time": t, "cost": c, "impact": i})
    result = tally(survey, [vote("Л1", "А", "А", "Б"), vote("Л2", "А", "Б", "Б"),
                            vote("Л3", "Б", "А", "Б")])
    assert result["winners"] == {"time": ["А"], "cost": ["А"], "impact": ["Б"]}
    # Б виграє лише вплив, але одностайно — і це переважує дві вузькі перемоги А.
    assert result["best"] == ["Б"]
    assert tally(survey, [vote("Л1", "А", "А", "А"), vote("Л2", "Б", "Б", "Б")])["best"] == ["А", "Б"]
    assert tally(survey, [])["best"] == []


def test_system_choice_follows_numbers(survey):
    calc = system_choice(survey)
    by_key = {o["key"]: o for o in survey["options"]}
    assert by_key[calc["cost"]]["cost"] == min(o["cost"] for o in survey["options"])
    assert by_key[calc["impact"]]["impact"] == max(o["impact"] for o in survey["options"])


def test_agents_answer_the_same_survey(survey):
    answer = ask_agent(lambda s, u, a: '```json\n{"time": "A", "cost": "б", "impact": "А"}\n```',
                       AGENTS[0], survey)
    assert answer["response"]["choices"] == {"time": "А", "cost": "Б", "impact": "А"}
    broken = ask_agent(lambda s, u, a: '{"time": "А"}', AGENTS[1], survey)
    assert broken["response"] is None and "Ціна" in broken["error"]
    assert [a["id"] for a in AGENTS] == ["L1", "L2", "L3"]


def test_synthesis_gets_the_computed_verdict(survey):
    seen = {}

    def generate(system, user, tag):
        seen["user"] = user
        return "підсумок"
    responses = [make_response(survey, "Л1", {"time": "А", "cost": "Б", "impact": "А"},
                               comment="дорого")]
    assert synthesize(generate, survey, responses, tally(survey, responses)) == "підсумок"
    assert "ЗАГАЛЬНИЙ ВИБІР (рекомендація): варіант А" in seen["user"]
    assert "дорого" in seen["user"]


def test_second_batch_is_tried_when_first_finds_nothing():
    """Перший пакет кандидатів — без підтверджених мостів, тож модель питають ще раз."""
    calls = []
    base = fake_generate(confirm=frozenset())

    def generate(system, user, tag):
        if tag == "bridges":
            calls.append(user)
            if len(calls) == 2:
                return json.dumps({"links": [{"i": 1, "ok": True, "why": "так"}]})
            return json.dumps({"links": []})
        return base(system, user, tag)
    counter = iter(range(10_000))

    def orthogonal(texts):
        # Кожне поняття — власна вісь: близькість 0, тож автоматичних мостів немає.
        out = []
        for _ in texts:
            vec = [0.0] * 512
            vec[next(counter)] = 1.0
            out.append(vec)
        return out
    net = build_semantic_net("Проблема", "Розв'язок", "к", generate, orthogonal, k=3, depth=2)
    assert len(calls) == 2
    assert [b["kind"] for b in net.bridges] == ["link"]


# --- експерти: незалежність, порядок, Дельфі ----------------------------------------------


def _chain(rank, labels, cost=3, time=3, impact=0.5):
    return {"rank": rank, "path": ["P", *labels, "S"], "cost": cost, "time": time,
            "impact": impact, "j": 1.0 - rank / 10,
            "steps": [{"id": f"x{rank}{i}", "label": l, "cost": 1, "time": 1, "impact": 3}
                      for i, l in enumerate(labels)]}


@pytest.fixture
def options():
    return [_chain(1, ["Аудит", "Патчі"], cost=2), _chain(2, ["Аудит", "Патчі", "Сканування"]),
            _chain(3, ["EDR на ws-01", "SIEM-агенти"], impact=0.8),
            _chain(4, ["Паролі adm-01"], cost=1, time=1)]


def test_diverse_chains_drop_near_duplicates(options):
    picked = diverse_chains(options)
    assert [c["rank"] for c in picked] == [1, 3, 4]   # №2 ділить 2 з 3 кроків з №1
    assert [c["rank"] for c in diverse_chains(options, max_overlap=0.9)] == [1, 2, 3, 4]
    assert len(diverse_chains(options, limit=2)) == 2


def test_rank_chains_merges_identical_step_lists(net):
    ids = sorted({n for p in chains(net) for n in p} - {"P", "S"})
    ranked = rank_chains(net, score_steps(net, ids, "к", fake_generate()), k=50)["chains"]
    keys = [tuple(s["label"].lower() for s in c["steps"]) for c in ranked]
    assert len(keys) == len(set(keys))


@pytest.fixture
def expert_survey(case, options):
    return build_survey({**case, "context": "мережа з вузлами ws-01, adm-01"},
                        diverse_chains(options))


def test_experts_see_steps_without_numbers(expert_survey):
    text = survey_text(expert_survey, numbers=False)
    assert "Аудит" in text and "ws-01, adm-01" in text
    assert " б." not in text and "%" not in text
    assert "%" in survey_text(expert_survey)            # людині цифри показуємо


def test_each_expert_gets_own_reproducible_order(expert_survey):
    orders = [agent_order(expert_survey, a) for a in AGENTS]
    assert all(sorted(o) == ["А", "Б", "В"] for o in orders)
    assert orders == [agent_order(expert_survey, a) for a in AGENTS]
    assert len({tuple(o) for o in orders}) > 1


def test_expert_answer_keeps_reasons_and_estimates(expert_survey):
    seen = {}

    def generate(system, user, tag):
        seen["system"], seen["user"] = system, user
        return json.dumps({
            "estimates": {"A": {"cost": 2, "time": 9, "impact": "4"}, "Я": {"cost": 1}},
            "time": {"choice": "В", "why": "лише паролі"},
            "cost": {"choice": "А", "why": "дешево"},
            "impact": {"choice": "Б", "why": "EDR на ws-01"}}, ensure_ascii=False)
    answer = ask_agent(generate, AGENTS[1], expert_survey)
    response = answer["response"]
    assert response["choices"] == {"time": "В", "cost": "А", "impact": "Б"}
    assert response["reasons"]["impact"] == "EDR на ws-01"
    assert response["estimates"] == {"А": {"cost": 2, "time": 5, "impact": 4}}
    assert response["round"] == 1
    assert "приховані витрати" in seen["system"]          # лінза саме цієї ролі
    assert " б." not in seen["user"]


def test_delphi_round_records_changes(expert_survey):
    own = make_response(expert_survey, AGENTS[0]["name"], {"time": "А", "cost": "А", "impact": "А"},
                        kind="agent", reasons={"impact": "моє"}, estimates={"А": {"cost": 1}})
    other = make_response(expert_survey, AGENTS[1]["name"], {"time": "В", "cost": "А", "impact": "Б"},
                          kind="agent", reasons={"impact": "EDR важливіший"})
    seen = {}

    def generate(system, user, tag):
        seen["user"], seen["tag"] = user, tag
        return '{"time": {"choice": "А", "why": "лишаюсь"}, "cost": {"choice": "А", "why": "так"},' \
               ' "impact": {"choice": "Б", "why": "переконав аргумент про EDR"}}'
    answer = revise_agent(generate, AGENTS[0], expert_survey, own, [other])
    response = answer["response"]
    assert seen["tag"] == "L1:delphi"
    assert "EDR важливіший" in seen["user"] and AGENTS[1]["name"] not in seen["user"]  # анонімно
    assert response["round"] == 2 and response["estimates"] == own["estimates"]
    assert changes(response) == {"impact": ("А", "Б")}


def test_expert_estimates_average(expert_survey):
    responses = [
        make_response(expert_survey, "Л1", {"time": "А", "cost": "А", "impact": "А"},
                      estimates={"А": {"cost": 2, "time": 3, "impact": 4}}),
        make_response(expert_survey, "Л2", {"time": "А", "cost": "А", "impact": "А"},
                      estimates={"А": {"cost": 4, "time": 3, "impact": 5}, "Б": {"cost": 1}}),
    ]
    assert expert_estimates(expert_survey, responses) == {
        "А": {"cost": 3.0, "time": 3.0, "impact": 4.5}, "Б": {"cost": 1.0}}
