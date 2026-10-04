"""Сторінка методу «від проблеми до розв'язку» — спільна для ONYX і локального стенда.

НАВІЩО ОКРЕМИЙ МОДУЛЬ. Той самий екран потрібен у двох місцях: у бойовому
`ui.py` (режим «🧭 Від проблеми до розв'язку», за логіном, на моделях сервера й
на корпусі ONYX) і в `scenario_lab.py` (локально, без Mongo/Manticore). Що
відрізняється — звідки модель, ембедер і корпус, — передається ззовні в
`SolutionEnv`; сам екран про інфраструктуру не знає.

Ключі стану сесії й віджетів мають префікс `sol_`: у `ui.py` ця сторінка живе
поруч із чатом і стрічкою, і їхні ключі не повинні перетинатися.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network as VisNetwork

from network_context import default_problem, default_solution, network_brief, parse_network_xml
from semantic_bridge import (
    PROBLEM_ID,
    SOLUTION_ID,
    Embed,
    Generate,
    Retrieve,
    SemanticNet,
    build_semantic_net,
    chains,
    rank_chains,
    score_steps,
)
from solution_survey import (
    AGENTS,
    CRITERION_TITLE,
    OPTION_KEYS,
    SurveyStore,
    ask_agent,
    build_survey,
    changes,
    diverse_chains,
    expert_estimates,
    level,
    make_response,
    revise_agent,
    synthesize,
    system_choice,
    tally,
    write_legend,
)

# Помилки, які означають «модель/ембедер/сховище недоступні», а не баг коду.
SERVICE_ERRORS = (httpx.HTTPError, ConnectionError, TimeoutError)


@dataclass
class SolutionEnv:
    """Усе, що сторінка бере з середовища.

    make_generate(backend, temperature) -> Generate — виклик моделі; backend —
        ключ із `backends` (None — модель за замовчуванням), temperature None —
        типова для виклику (0 для перевірки мостів і оцінок, 0.3 для решти).
    backends — {ключ: підпис} для вибору моделі кожного ШІ-експерта.
    embed — ембедер (nomic: поняття порівнюються за англійською формою).
    retrieve — RAG-опора для розкриття фронтів або None.
    """
    make_generate: Callable[[str | None, float | None], Generate]
    backends: dict[str, str]
    embed: Embed
    networks_dir: Path
    data_dir: Path
    model_tag: str
    retrieve: Retrieve | None = None
    retrieve_label: str = ""
    notes: list[str] = field(default_factory=list)


def K(name: str) -> str:
    return f"sol_{name}"


# =========================================================================
#  Граф: два фронти й мости
# =========================================================================

SIDE_STYLE = {
    "problem": ("#d93f3f", "triangle", 34),
    "solution": ("#2e9d5b", "star", 38),
    "forward": ("#3b6fd1", "dot", 18),
    "backward": ("#2e9d8f", "dot", 18),
}


def _wrap(text: str, width: int = 22, lines_max: int = 4) -> str:
    """Перенос підпису по словах, щоб вузол ріс у висоту, а не в ширину."""
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    if len(lines) > lines_max:
        lines = lines[:lines_max]
        lines[-1] += "…"
    return "\n".join(lines)


def semantic_graph_html(net: SemanticNet, highlight: list[str]) -> str:
    """Проблема зліва, розв'язок справа; прямий фронт росте праворуч від
    проблеми, зворотний — ліворуч від розв'язку, мости з'єднують їх посередині.
    Розкладка вручну: кожен крок фронту — свій стовпець."""
    depth = max((c.depth for c in net.concepts.values()), default=0)
    column_x, row_y = 260, 120
    last = 2 * depth + 1
    columns: dict[int, list[str]] = {}
    for concept in net.concepts.values():
        if concept.side == "problem":
            col = 0
        elif concept.side == "solution":
            col = last
        elif concept.side == "forward":
            col = concept.depth
        else:
            col = last - concept.depth
        columns.setdefault(col, []).append(concept.id)
    position = {}
    for col, ids in columns.items():
        # Порядок у стовпці — за батьком, щоб гілки не перехрещувались.
        ids.sort(key=lambda cid: (net.concepts[cid].parent, cid))
        for i, cid in enumerate(ids):
            position[cid] = (col * column_x, int((i - (len(ids) - 1) / 2) * row_y))

    on_chain = set(highlight)
    chain_edges = set(zip(highlight, highlight[1:]))
    vis = VisNetwork(height="680px", width="100%", bgcolor="#ffffff", font_color="#1b1b2b",
                     notebook=True, directed=True, cdn_resources="remote")
    vis.set_options(json.dumps({"interaction": {"hover": True, "navigationButtons": True},
                                "physics": {"enabled": False}}))
    for concept in net.concepts.values():
        color, shape, size = SIDE_STYLE[concept.side]
        if highlight and concept.id not in on_chain and concept.side not in {"problem", "solution"}:
            color = "#d9dde3"
        x, y = position[concept.id]
        title = concept.label + (f"\n({concept.en})" if concept.en != concept.label else "")
        vis.add_node(concept.id, label=_wrap(concept.label), title=title, color=color,
                     shape=shape, size=size, x=x, y=y, physics=False, font={"size": 14})
    for src, dst, relation in net.edges:
        bridge = relation in {"міст", "те саме"}
        bold = (src, dst) in chain_edges
        if bold:
            color, width = "#d93f3f", 4
        elif bridge:
            color, width = "#f0a43a", 3
        else:
            color, width = "#c3c9d2", 1.2
        vis.add_edge(src, dst, title=relation, color=color, width=width,
                     dashes=bridge, arrows="to", smooth={"enabled": True, "type": "curvedCW",
                                                         "roundness": 0.08})
    return vis.generate_html(notebook=True)


# =========================================================================
#  Сторінка
# =========================================================================

def render_solution_mode(env: SolutionEnv) -> None:
    generate = env.make_generate(None, None)
    cache_dir = env.data_dir / "semantic_cache"
    store = SurveyStore(env.data_dir / "surveys")

    st.markdown("### 🧭 Від проблеми до розв'язку: двостороння семантична мережа")
    st.caption("1. фізична мережа (контекст) → 2. RAG → 3. проблема й розв'язок → 4. мережа "
               "понять з двох боків → 5. ланцюги через мости → 6. ранжування (ціна, час, вплив) "
               "→ 7. легенди простою мовою → 8. анкета для респондентів, синтез")
    for note in env.notes:
        st.caption(note)

    # --- 1. Фізична мережа як контекст ---------------------------------------
    files = sorted(p.name for p in env.networks_dir.glob("*.xml")) \
        if env.networks_dir.exists() else []
    col_src, col_up = st.columns([2, 3])
    chosen = col_src.selectbox("1. Фізична мережа (XML) — контекст", files,
                               key=K("network")) if files else None
    uploaded = col_up.file_uploader("…або завантажте свій XML", type=["xml"], key=K("upload"))
    xml_text = (uploaded.read().decode("utf-8", "replace") if uploaded
                else (env.networks_dir / chosen).read_text(encoding="utf-8") if chosen else "")
    if not xml_text:
        st.info(f"Немає XML-мереж у {env.networks_dir} — завантажте файл.")
        return
    try:
        phys = parse_network_xml(xml_text)
    except ValueError as exc:
        st.error(f"Мережу не прийнято: {exc}")
        return
    context = network_brief(phys)
    with st.expander("Що модель знає про цю мережу (контекст для побудови)"):
        st.text(context)

    # --- 3. Проблема й розв'язок ----------------------------------------------
    st.markdown("#### 3. Проблема і бажаний розв'язок")
    p1, p2 = st.columns(2)
    problem = p1.text_area("Проблема (звідки йдемо)", default_problem(phys), height=100,
                           key=K(f"problem_{phys.id}"))
    solution = p2.text_area("Розв'язок (куди йдемо)", default_solution(phys), height=100,
                            key=K(f"solution_{phys.id}"))
    case = {"id": phys.id, "problem": problem.strip(), "solution": solution.strip(),
            "context": context}

    # --- 4. Побудова мережі -----------------------------------------------------
    st.markdown("#### 4. Семантична мережа з двох боків")
    b1, b2, b3, b4 = st.columns([1, 1, 1, 2])
    k = b1.slider("Понять на кожному кроці", 2, 4, 3, key=K("k"))
    depth = b2.slider("Кроків з кожного боку", 1, 3, 2, key=K("depth"))
    use_rag = b3.checkbox(f"Опора на корпус{(' (' + env.retrieve_label + ')') if env.retrieve_label else ''}",
                          value=env.retrieve is not None, disabled=env.retrieve is None,
                          key=K("rag"))
    calls = 2 * sum(k ** d for d in range(depth)) + 2
    b4.caption(f"Прямий фронт росте від проблеми, зворотний — від розв'язку; де вони "
               f"зустрічаються, модель підтверджує мости. Це ≈{calls} викликів моделі. "
               "Побудована мережа кешується на диску — повторно її не рахуємо.")

    cache_key = hashlib.sha1(json.dumps(
        [case["problem"], case["solution"], context, k, depth, env.model_tag, bool(use_rag)],
        ensure_ascii=False).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"{phys.id}-{cache_key}.json"
    net: SemanticNet | None = None
    scores: dict = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        net = SemanticNet.from_dict(cached["net"])
        scores = cached.get("scores", {})

    if st.button("🔄 Побудувати заново" if net else "🧠 Побудувати семантичну мережу",
                 type="secondary" if net else "primary", key=K("build")):
        status = st.status("Будую мережу…", expanded=True)
        try:
            net = build_semantic_net(case["problem"], case["solution"], context, generate,
                                     env.embed, k=k, depth=depth,
                                     retrieve=env.retrieve if use_rag else None,
                                     progress=lambda m: status.write(m))
            status.write("Оцінюю кроки за ціною, часом і впливом…")
            ids = sorted({n for p in chains(net) for n in p} - {PROBLEM_ID, SOLUTION_ID})
            scores = score_steps(net, ids, context, generate)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"net": net.to_dict(), "scores": scores},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
            status.update(label="Мережу побудовано", state="complete", expanded=False)
        except SERVICE_ERRORS as exc:
            status.update(label="Модель недоступна", state="error")
            st.error(f"Модель або ембедер недоступні: {exc}")
            return
        st.rerun()

    if net is None:
        st.info("Натисніть «Побудувати семантичну мережу».")
        return

    m = st.columns(4)
    m[0].metric("Понять від проблеми", len(net.side("forward")))
    m[1].metric("Понять від розв'язку", len(net.side("backward")))
    m[2].metric("Мостів", len(net.bridges))
    m[3].metric("Ланцюгів", len(chains(net)))
    for line in net.log:
        st.caption(f"⚠️ {line}")

    with st.expander("⚖️ Параметри ранжування (пункт 6)", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        w_cost = c1.slider("Вага: ціна", 0.0, 1.0, 0.3, 0.05, key=K("w_cost"))
        w_time = c2.slider("Вага: час", 0.0, 1.0, 0.3, 0.05, key=K("w_time"))
        w_impact = c3.slider("Вага: вплив", 0.0, 1.0, 0.4, 0.05, key=K("w_impact"))
        top_k = c4.slider("Скільки ланцюгів показати", 1, 10, 5, key=K("top_k"))
        min_impact = c5.slider("Мінімальний вплив", 0.0, 1.0, 0.0, 0.05, key=K("min_impact"))
    weights = {"cost": w_cost, "time": w_time, "impact": w_impact}
    ranking = rank_chains(net, scores, weights, k=top_k, min_impact=min_impact)
    ranked = ranking["chains"]

    g1, g2 = st.columns([1, 3])
    pick = g1.selectbox("Підсвітити ланцюг", [0] + [c["rank"] for c in ranked],
                        format_func=lambda r: "—" if r == 0 else f"№{r}",
                        index=1 if ranked else 0, key=K("pick"))
    g2.caption("🔺 проблема · ★ розв'язок · 🔵 поняття від проблеми · 🟢 поняття від розв'язку · "
               "помаранчевий пунктир — міст між фронтами. Наведіть на вузол — повний текст.")
    highlight = next((c["path"] for c in ranked if c["rank"] == pick), [])
    components.html(semantic_graph_html(net, highlight), height=700)

    if net.bridges:
        with st.expander(f"Мости між фронтами ({len(net.bridges)})"):
            for bridge in net.bridges:
                a, b = net.concepts[bridge["from"]], net.concepts[bridge["to"]]
                kind = "те саме поняття" if bridge["kind"] == "same" else "підтверджено моделлю"
                why = f". {bridge['why']}" if bridge.get("why") and bridge["kind"] != "same" else ""
                st.markdown(f"`{a.label}` ⇄ `{b.label}` — {kind}, близькість "
                            f"{bridge['cos']:.2f}{why}")

    # --- 5. Ланцюги -----------------------------------------------------------------
    st.markdown("#### 5. Ланцюги від проблеми до розв'язку")
    if not net.bridges:
        st.warning("Фронти не зустрілись — мостів немає. Збільште кількість кроків або понять "
                   "на кроці й побудуйте заново.")
        return
    st.caption(f"Усього ланцюгів: {ranking['total']} · допустимих: {ranking['admissible']} · "
               f"відкинуто за впливом: {ranking['rejected_impact']}")
    if not ranked:
        st.warning("Жоден ланцюг не проходить поріг впливу — послабте його.")
        return
    # В анкету йдуть лише різні за суттю ланцюги: майже однакові плани (≥60%
    # спільних кроків) змушують респондента обирати між тим самим.
    options_chains = diverse_chains(ranked)
    letter = {c["rank"]: OPTION_KEYS[i] for i, c in enumerate(options_chains)}
    skipped = len(ranked) - len(options_chains)
    if skipped:
        st.caption(f"В анкету йдуть {len(options_chains)} різних варіантів; ще {skipped} "
                   "відсіяно як майже однакові з кращими (мають ≥60% спільних кроків).")
    for chain in ranked:
        st.markdown(f"**{letter.get(chain['rank'], '')} · №{chain['rank']}** · "
                    + "  →  ".join(f"`{s['label']}`" for s in chain["steps"]))

    # --- 6. Ранжування ----------------------------------------------------------------
    st.markdown("#### 6. Ранжування за впливом, ціною й часом")
    st.dataframe(pd.DataFrame([{
        "Варіант": letter.get(c["rank"], "—"), "№": c["rank"], "Кроків": len(c["steps"]),
        "Ціна, б.": c["cost"], "Час, б.": c["time"],
        "Розв'язує, %": round(c["impact"] * 100), "J": c["j"],
        "Оцінки": "частково припущені" if c["assumed"] else "від моделі",
    } for c in ranked]), hide_index=True, width="stretch")
    st.caption("Кожен крок модель оцінила від 1 до 5. Ціна й час ланцюга — сума балів кроків; "
               "вплив — 1 − Π(1 − 0.15·вплив кроку). J — зважена сума нормованих критеріїв "
               "(`qa.twin.aggregate`), ціна й час інвертовані.")
    with st.expander("Оцінки кожного кроку"):
        st.dataframe(pd.DataFrame([{
            "Варіант": letter.get(c["rank"]), "Крок": s["label"],
            "Бік": "від проблеми" if s["side"] == "forward" else "від розв'язку",
            "Ціна": s["cost"], "Час": s["time"], "Вплив": s["impact"],
        } for c in ranked for s in c["steps"]]), hide_index=True, width="stretch")

    # --- 7. Легенди ----------------------------------------------------------------
    st.markdown("#### 7. Легенди: кожен варіант простою мовою")
    chain_key = (cache_key, tuple(tuple(c["path"]) for c in ranked))
    legend_state = st.session_state.get(K("legends"))
    if not legend_state or legend_state.get("key") != chain_key:
        legend_state = {"key": chain_key, "by_rank": {}, "errors": []}
        st.session_state[K("legends")] = legend_state

    if st.button("✍️ Переписати легенди за допомогою ШІ", key=K("legend_btn")):
        bar = st.progress(0.0, text="Пишу легенди…")
        legend_state["errors"] = []
        for i, chain in enumerate(options_chains):
            bar.progress(i / len(options_chains),
                         text=f"Легенда варіанта {letter.get(chain['rank'])}…")
            try:
                legend_state["by_rank"][chain["rank"]] = write_legend(generate, case, chain)
            except SERVICE_ERRORS as exc:
                legend_state["errors"].append(f"Модель недоступна: {exc}. Лишаю шаблонні легенди.")
                break
        bar.empty()
        st.rerun()
    for error in legend_state["errors"]:
        st.error(error)
    if not legend_state["by_rank"]:
        st.caption("Зараз показано шаблонні легенди. Кнопка вище попросить ШІ переписати їх "
                   "живою мовою — вони ж підуть в анкету.")

    # --- 8. Анкета ------------------------------------------------------------------
    survey = build_survey(case, options_chains, legend_state["by_rank"])
    responses = store.load(survey["id"])
    st.markdown("#### 8. Анкета: як розв'язати проблему?")
    st.caption("Респондент читає варіанти й обирає, який найкращий за часом, який за ціною "
               "і який за впливом. Відповіді зберігаються у файл, тож анкету можуть заповнити "
               f"кілька людей. Ідентифікатор анкети: `{survey['id']}`.")
    st.error(f"**Проблема.** {survey['problem']}")
    st.success(f"**Бажаний результат.** {survey['solution']}")
    st.markdown(f"##### {survey['question']}")
    for option in survey["options"]:
        with st.container(border=True):
            head, m1, m2, m3 = st.columns([4, 1, 1, 1])
            head.markdown(f"**Варіант {option['key']}** · {option['title']}")
            m1.metric("Ціна", level(option["cost"], option["n"]), f"{option['cost']} б.",
                      delta_color="off")
            m2.metric("Час", level(option["time"], option["n"]), f"{option['time']} б.",
                      delta_color="off")
            m3.metric("Розв'язує", f"{round(option['impact'] * 100)}%")
            st.markdown(option["legend"])
            with st.expander("Кроки плану"):
                st.markdown("\n".join(f"{i}. {step}"
                                      for i, step in enumerate(option["steps"], 1)))

    keys = [o["key"] for o in survey["options"]]
    with st.form(K("survey_form"), clear_on_submit=True):
        st.markdown("##### Ваш вибір")
        respondent = st.text_input("Хто відповідає (ім'я або роль)",
                                   placeholder="Напр.: Л1 · керівник ІТ-відділу",
                                   key=K("respondent"))
        picks = {criterion: st.radio(question, keys, index=None, horizontal=True,
                                     format_func=lambda key: f"Варіант {key}",
                                     key=K(f"pick_{criterion}"))
                 for criterion, question in survey["questions"].items()}
        comment = st.text_area("Коментар (необов'язково)", height=80, key=K("comment"))
        if st.form_submit_button("Надіслати відповідь", type="primary"):
            try:
                store.add(survey["id"], make_response(survey, respondent, picks, comment))
                st.session_state[K("flash")] = f"Відповідь «{respondent.strip()}» збережено."
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if st.session_state.get(K("flash")):
        st.success(st.session_state.pop(K("flash")))

    _render_experts(env, survey, responses, store)
    _render_results(survey, responses, store, weights, generate)


def _render_experts(env: SolutionEnv, survey: dict, responses: list[dict],
                    store: SurveyStore) -> None:
    st.markdown("##### ШІ-експерти Л1–Л3")
    st.caption("Раунд 1 — кожен експерт відповідає незалежно: бачить лише кроки планів (без "
               "балів розрахунку) у власному порядку, сам оцінює кожен варіант і пояснює вибір "
               "окремо для кожного питання. Раунд 2 (метод Дельфі) — бачить анонімні відповіді "
               "інших і може змінити думку, пояснивши чому.")
    names = list(env.backends)
    conf = {}
    with st.expander("Налаштування експертів: модель для кожного"):
        st.caption("Різні моделі роблять думки експертів незалежнішими: на одній моделі ролі "
                   "дають різні аргументи, але здебільшого той самий вибір.")
        for agent in AGENTS:
            a1, a2 = st.columns([3, 1])
            conf[agent["id"]] = (
                a1.selectbox(agent["name"], names, format_func=lambda n: env.backends[n],
                             key=K(f"backend_{agent['id']}")),
                a2.number_input("t°", 0.0, 1.5, float(agent["temperature"]), 0.1,
                                key=K(f"temp_{agent['id']}")),
            )

    by_name = {r["respondent"]: r for r in responses}
    round1_ready = all(agent["name"] in by_name for agent in AGENTS)
    r1, r2 = st.columns(2)
    run_round1 = r1.button("🤖 Раунд 1: незалежні відповіді", width="stretch", key=K("round1"))
    run_round2 = r2.button("🔁 Раунд 2: Дельфі", width="stretch", disabled=not round1_ready,
                           key=K("round2"),
                           help=None if round1_ready else "Спершу потрібні відповіді всіх "
                                                          "трьох експертів у раунді 1.")
    if run_round1 or run_round2:
        bar = st.progress(0.0, text="Експерти працюють…")
        problems = []
        # Раунд 2 бачить відповіді раунду 1 усіх — знімок до початку, щоб перший,
        # хто переглянув думку, не впливав на наступних у тому самому раунді.
        snapshot = {agent["id"]: by_name.get(agent["name"]) for agent in AGENTS}
        for i, agent in enumerate(AGENTS):
            bar.progress(i / len(AGENTS), text=f"{agent['name']}…")
            backend, temperature = conf[agent["id"]]
            agent_generate = env.make_generate(backend, temperature)
            try:
                if run_round1:
                    answer = ask_agent(agent_generate, agent, survey)
                else:
                    own = snapshot[agent["id"]]
                    others = [r for aid, r in snapshot.items() if aid != agent["id"] and r]
                    answer = revise_agent(agent_generate, agent, survey, own, others)
            except SERVICE_ERRORS as exc:
                problems.append(f"{agent['name']}: модель недоступна ({exc})")
                continue
            if answer["response"]:
                store.add(survey["id"], answer["response"])
            else:
                problems.append(f"{agent['name']}: {answer['error']}. Відповідь моделі: "
                                f"{answer['raw'][:300]}")
        bar.empty()
        st.session_state[K("agent_problems")] = problems
        st.rerun()
    for problem_text in st.session_state.pop(K("agent_problems"), []):
        st.warning(problem_text)


def _render_results(survey: dict, responses: list[dict], store: SurveyStore,
                    weights: dict, generate: Generate) -> None:
    st.markdown(f"##### Відповіді ({len(responses)})")
    if not responses:
        st.info("Відповідей ще немає.")
        return
    st.dataframe(pd.DataFrame([{
        "Респондент": r["respondent"],
        "Хто": "ШІ" if r.get("kind") == "agent" else "людина",
        "За часом": r["choices"].get("time"),
        "За ціною": r["choices"].get("cost"),
        "За впливом": r["choices"].get("impact"),
        "Раунд": r.get("round", 1),
        "Змінив думку": "; ".join(f"{CRITERION_TITLE[c]} {a}→{b}"
                                  for c, (a, b) in changes(r).items()),
        "Коментар": r.get("comment", ""),
    } for r in responses]), hide_index=True, width="stretch")
    reasoned = [r for r in responses if r.get("reasons")]
    if reasoned:
        with st.expander("Обґрунтування експертів по кожному питанню", expanded=True):
            for r in reasoned:
                st.markdown(f"**{r['respondent']}** (раунд {r.get('round', 1)})")
                for criterion in survey["questions"]:
                    st.markdown(f"- {CRITERION_TITLE[criterion]}: **варіант "
                                f"{r['choices'][criterion]}** — "
                                f"{r['reasons'].get(criterion, '')}")
    estimates = expert_estimates(survey, responses)
    if estimates:
        with st.expander("Власні оцінки експертів проти оцінок моделі"):
            st.caption("Модель — середній бал на крок плану; експерти — середня оцінка "
                       "всього плану (1-5). Великий розрив показує, де розрахунку не довіряють.")
            st.dataframe(pd.DataFrame([{
                "Варіант": o["key"],
                "Ціна: модель": round(o["cost"] / o["n"], 1),
                "Ціна: експерти": estimates.get(o["key"], {}).get("cost"),
                "Час: модель": round(o["time"] / o["n"], 1),
                "Час: експерти": estimates.get(o["key"], {}).get("time"),
                "Вплив: модель, %": round(o["impact"] * 100),
                "Вплив: експерти": estimates.get(o["key"], {}).get("impact"),
            } for o in survey["options"]]), hide_index=True, width="stretch")
    with st.expander("Видалити відповідь"):
        who = st.selectbox("Чию відповідь видалити", [r["respondent"] for r in responses],
                           key=K("delete_who"))
        if st.button("Видалити", key=K("delete_btn")):
            store.remove(survey["id"], who)
            st.rerun()

    result = tally(survey, responses, weights)
    calc = system_choice(survey)
    st.dataframe(pd.DataFrame([{
        "Питання": CRITERION_TITLE[criterion],
        **{f"Варіант {key}": v for key, v in result["counts"][criterion].items()},
        "Обрали респонденти": ", ".join(result["winners"][criterion]) or "—",
        "Обрав би розрахунок": calc.get(criterion, "—"),
    } for criterion in survey["questions"]]), hide_index=True, width="stretch")
    best = result["best"]
    if len(best) == 1:
        st.success(f"**Загальний вибір респондентів: варіант {best[0]}** "
                   f"(оцінка {result['overall'][0]['score']:.2f} з урахуванням ваг ціни, "
                   "часу й впливу з пункту 6).")
    else:
        st.warning(f"Нічия між варіантами {', '.join(best)} — потрібні ще відповіді.")

    synth_state = st.session_state.get(K("synth")) or {}
    if st.button("🧾 Підсумок простою мовою (ШІ)", key=K("synth_btn")):
        try:
            text = synthesize(generate, survey, responses, result)
            st.session_state[K("synth")] = {"id": survey["id"], "n": len(responses), "text": text}
        except SERVICE_ERRORS as exc:
            st.session_state[K("synth")] = {"id": survey["id"], "n": len(responses),
                                            "text": "", "error": str(exc)}
        st.rerun()
    if synth_state.get("id") == survey["id"]:
        if synth_state.get("error"):
            st.error(f"Модель недоступна: {synth_state['error']}")
        elif synth_state.get("text"):
            if synth_state.get("n") != len(responses):
                st.caption("Підсумок зроблено до останніх відповідей — оновіть його.")
            st.info(synth_state["text"])
