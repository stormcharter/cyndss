"""Захисне плече: шляхи атаки (`qa.twin.enumerate_scenarios`) -> набори засобів.

НАВІЩО. `qa/twin.py` разом з `twin_bridge.py`/`twin_scenarios.py` знаходять
шляхи КОМПРОМЕТАЦІЇ від периметра до цілі. Аналітику мало знати, ЩО можливо -
треба знати, ЯКИМИ засобами це розірвати. Формально це задача про мінімальну
трансверсаль (hitting set): кожен шлях атаки проходить через певні
вразливості; кожен `control`-вузол закриває певну множину вразливостей
(ребра `mitigates` / `detects` / `recovers_from`, `qa.twin.TREATMENT_RELATIONS`);
треба вибрати набір control-вузлів так, щоб у кожному шляху була закрита
хоча б одна вразливість.

ЧОМУ `detects`/`recovers_from` НЕ розривають шлях. Обидва відношення
реагують ПІСЛЯ того, як перехід уже стався: детектор бачить факт
експлуатації постфактум, відновлення повертає стан після інциденту. Жодне з
них не робить крок атаки недосяжним - вони зменшують НАСЛІДОК, а не
ЙМОВІРНІСТЬ переходу. Для трансверсалі (розрив шляху) рахується лише
`mitigates`. Зафіксовано тестом `test_coverage_excludes_detects_and_recovers_from`.

Задача NP-важка (двоїста до set cover), тож перебір мінімальних за
включенням наборів обмежений і за розміром (`max_size`), і за кількістю
переглянутих кандидатів (`MAX_CANDIDATES`). Якщо повного розриву в цих межах
не існує, повертається один прозоро позначений (`exact: False`) частковий
набір, а шляхи, що лишились нерозірваними, названі явно - мовчати про
неповноту тут гірше, ніж чесно сказати "далі не пощастило".
"""
from __future__ import annotations

import dataclasses
import itertools

from qa.twin import TwinCriterion, TwinModel, aggregate, within_constraints

#: Жорстка межа кількості переглянутих комбінацій control-вузлів у точному
#: пошуку мінімальних наборів. При розумній кількості засобів (фікстура має
#: 8) перебір і близько не підходить до цієї межі; вона лише рятує від
#: комбінаторного вибуху на значно більших моделях.
MAX_CANDIDATES = 20_000


def coverage(model: TwinModel) -> dict[str, frozenset[str]]:
    """control-вузол -> множина вразливостей, які він РОЗРИВАЄ (лише `mitigates`).

    Дивись пояснення в докстрінгу модуля: `detects`/`recovers_from` теж
    `TREATMENT_RELATIONS`, але жодне з них не унеможливлює перехід, тож у
    трансверсаль вони не рахуються."""
    result: dict[str, set[str]] = {}
    by_id = model.node_by_id
    for edge in model.edges:
        if edge.relation != "mitigates":
            continue
        source, target = by_id.get(edge.source), by_id.get(edge.target)
        if source is None or target is None:
            continue
        if source.kind != "control" or target.kind != "vuln":
            continue
        result.setdefault(source.id, set()).add(target.id)
    return {control_id: frozenset(vulns) for control_id, vulns in result.items()}


def vulnerabilities_on_path(model: TwinModel, edge_path: list[int]) -> frozenset[str]:
    """Вразливості (`kind=vuln`), які трапляються серед вузлів шляху.

    Шлях подається як послідовність ІНДЕКСІВ ребер (`edge_path` з
    `enumerate_scenarios`), а не вузлів: id вузла тут виводиться з ребра, бо
    це єдине, що зовнішній виклик стабільно передає."""
    by_id = model.node_by_id
    touched: set[str] = set()
    for index in edge_path or []:
        edge = model.edges[index]
        touched.add(edge.source)
        touched.add(edge.target)
    return frozenset(
        node_id
        for node_id in touched
        if (node := by_id.get(node_id)) is not None and node.kind == "vuln"
    )


def unevaluable_constraints(model: TwinModel) -> list[str]:
    """Критерії обмежень (`TwinConstraint.criterion`), яких МОДЕЛЬ у принципі
    НЕ може підтвердити - не тому, що конкретному набору бракує засобів, а
    тому, що в моделі немає ЖОДНОГО вузла, здатного нести потрібний доказ.

    ЧОМУ ЦЕ ОКРЕМИЙ СТАН, А НЕ ПОРУШЕННЯ. `regulatory`-обмеження вимагає, щоб
    КОЖЕН control-вузол набору мав атрибут `regulatory_approved`
    (`qa.twin._within_constraints`). У джерельних XML корпоративного
    датасету (`experiments/corporate_networks/dataset.py::_parse_defense`)
    цього поля НЕМАЄ взагалі - тож жоден control-вузол НІКОЛИ не матиме
    атрибута, і будь-який непорожній набір автоматично "порушував" би вимогу,
    якої фізично нема з чим звірити. Мовчати про це так само небезпечно, як і
    вигадати атрибут: аналітик має побачити ЧЕСНО "нічим підтвердити", а не
    "порушено" чи "пройдено".

    ПРАВИЛО. Вимога вважається неоцінюваною, лише якщо ЖОДЕН control-вузол
    УСІЄЇ моделі (не лише розглянутого набору) не має відповідного атрибута
    ВЗАГАЛІ - незалежно від значення (`False` теж рахується присутністю: це
    явне "не затверджено", а не мовчання). Якщо атрибут є хоч на одному
    control-вузлі БУДЬ-ДЕ в моделі, дані існують і вимога лишається
    ОЦІНЮВАНОЮ для всіх наборів - навіть тих, чиї власні control-вузли
    атрибута не мають (це вже РЕАЛЬНЕ порушення, не брак даних).

    `budget`/`months`/`personnel` сюди НЕ потрапляють: відсутність числового
    атрибута на вузлі вже трактується як 0 (узгоджена конвенція обчислення
    суми, а не брак даних, що робив би обмеження неоцінюваним) - ці три
    обмеження лишаються завжди оцінюваними, і послаблювати їх тут не можна
    (реальний бюджет і далі має відкидати набір, що в нього не вкладається)."""
    result: list[str] = []
    for constraint in model.constraints:
        if constraint.criterion != "regulatory" or constraint.limit < 1:
            continue
        has_regulatory_data = any(
            node.kind == "control" and "regulatory_approved" in node.attributes
            for node in model.nodes
        )
        if not has_regulatory_data:
            result.append(constraint.criterion)
    return sorted(set(result))


def defense_within_constraints(model: TwinModel, control_ids) -> bool:
    """Чи вкладається набір control-вузлів у ВСІ ОЦІНЮВАНІ обмеження моделі
    (бюджет/місяці/персонал/regulatory, `TwinConstraint`,
    `qa.twin.ALLOWED_CONSTRAINTS`).

    Рахунок (сума `cost`/`months`/`personnel` по вузлах, вимога
    `regulatory_approved` при активному й ОЦІНЮВАНОМУ regulatory-обмеженні)
    той самий, що й для шляху сценарію в `enumerate_scenarios`.
    Перевикористовуємо публічну `qa.twin.within_constraints` замість другої
    копії арифметики бюджету: дві копії розійдуться, щойно хтось поправить
    лише одну з них. Ребер тут немає - вартість/персонал засобів захисту
    лежать на самих control-вузлах.

    НЕОЦІНЮВАНІ обмеження (`unevaluable_constraints`) свідомо ВИКЛЮЧАЮТЬСЯ з
    перевірки тут - вони не можуть ні підтвердити, ні спростувати набір, тож
    не повинні брати участі в рішенні "проходить/не проходить". Класифікація
    "проходить усе оцінюване, але має неоцінювану вимогу" - окрема турбота
    `rank_defenses` (третій кошик, `unevaluable_requirements`), не цієї
    булевої функції: збереження булевого контракту тут - свідомий вибір, щоб
    наявні виклики (і тести на них) не зламались, коли модель взагалі не має
    неоцінюваних вимог."""
    unevaluable = set(unevaluable_constraints(model))
    if not unevaluable:
        return within_constraints(model, list(control_ids), [])
    evaluable_model = dataclasses.replace(
        model,
        constraints=[c for c in model.constraints if c.criterion not in unevaluable],
    )
    return within_constraints(evaluable_model, list(control_ids), [])


def _control_hits(
    model: TwinModel, attack_paths: list[list[int]]
) -> dict[str, set[int]]:
    """control-вузол -> індекси шляхів (у `attack_paths`), які він розриває."""
    cov = coverage(model)
    path_vulns = [vulnerabilities_on_path(model, path) for path in attack_paths]
    hits: dict[str, set[int]] = {}
    for control_id, vulns in cov.items():
        broken = {i for i, pv in enumerate(path_vulns) if pv & vulns}
        if broken:
            hits[control_id] = broken
    return hits


def _greedy_partial_cover(
    hits: dict[str, set[int]], breakable: set[int], max_size: int
) -> tuple[list[str], set[int]]:
    """Жадібне наближення set cover: на кожному кроці додаємо control, що
    розриває найбільше ЩЕ не розірваних шляхів; за рівності - менший id
    (детермінізм). Не доведений мінімум (задача NP-важка) - прозоро
    позначений best-effort для випадку, коли точного повного розриву в межах
    `max_size` не існує."""
    remaining = set(breakable)
    pool = set(hits)
    chosen: list[str] = []
    while remaining and pool and len(chosen) < max_size:
        best = min(pool, key=lambda cid: (-len(hits[cid] & remaining), cid))
        gained = hits[best] & remaining
        pool.discard(best)
        if not gained:
            # `best` - максимум за покриттям серед пулу; якщо навіть він
            # нічого не додає, решта пулу теж нічого не додасть.
            break
        chosen.append(best)
        remaining -= gained
    return chosen, remaining


def cut_sets(
    model: TwinModel, attack_paths: list[list[int]], max_size: int = 4
) -> dict[str, object]:
    """Усі МІНІМАЛЬНІ за включенням набори control-вузлів, що розривають усі
    подані `attack_paths` (кожен шлях - `edge_path` зі `enumerate_scenarios`).

    Формально - трансверсаль (hitting set) гіперграфа "шлях -> вразливості на
    ньому"; рахуємо її як подвійну до set cover "control -> шляхи, які він
    розриває". Перебір точний, але обмежений і розміром (`max_size`), і
    кількістю переглянутих кандидатів (`MAX_CANDIDATES`) - задача NP-важка.

    Повертає:
        sets: знайдені набори, детерміновано впорядковані за розміром, тоді
            за id control-вузлів. Якщо існує точний повний розрив у межах
            `max_size` - усі такі мінімальні набори (`exact=True`,
            `covers_all=True`). Якщо ні - ОДИН жадібний частковий набір
            (`exact=False`), з явним переліком шляхів, які він не розірвав.
        unbreakable_path_indices: шляхи, які не розірве жоден наявний засіб
            узагалі, незалежно від вибору набору - варто показати аналітику
            окремо, бо жоден набір цього не виправить.
        truncated: чи впирався пошук у `MAX_CANDIDATES` (знайдені набори
            лишаються коректними, але могли лишитися неперевірені комбінації
            більшого розміру).
    """
    total = len(attack_paths)
    if total == 0:
        return {"sets": [], "unbreakable_path_indices": [], "truncated": False}

    hits = _control_hits(model, attack_paths)
    breakable = {index for indices in hits.values() for index in indices}
    unbreakable = sorted(set(range(total)) - breakable)

    if not breakable:
        return {"sets": [], "unbreakable_path_indices": unbreakable, "truncated": False}

    relevant = sorted(hits)  # сортуємо за id - результат не має залежати
    # від порядку вставки в `dict` (а той, своєю чергою, від порядку ребер у
    # вихідному документі).
    size_limit = max(1, int(max_size))

    accepted: list[frozenset[str]] = []
    examined = 0
    truncated = False
    for size in range(1, size_limit + 1):
        if truncated:
            break
        for combo in itertools.combinations(relevant, size):
            examined += 1
            if examined > MAX_CANDIDATES:
                truncated = True
                break
            candidate = frozenset(combo)
            if any(done <= candidate for done in accepted):
                continue  # надмножина вже прийнятого меншого набору - не мінімальний
            covered: set[int] = set()
            for control_id in combo:
                covered |= hits[control_id]
            if covered >= breakable:
                accepted.append(candidate)

    if accepted:
        ordered = sorted(accepted, key=lambda combo: (len(combo), tuple(sorted(combo))))
        sets = [
            {
                "controls": tuple(sorted(combo)),
                "size": len(combo),
                "exact": True,
                "covers_all": True,
                "uncovered_path_indices": unbreakable,
            }
            for combo in ordered
        ]
        return {"sets": sets, "unbreakable_path_indices": unbreakable, "truncated": truncated}

    chosen, remaining = _greedy_partial_cover(hits, breakable, size_limit)
    sets = (
        [
            {
                "controls": tuple(sorted(chosen)),
                "size": len(chosen),
                "exact": False,
                "covers_all": not remaining,
                "uncovered_path_indices": sorted(unbreakable + list(remaining)),
            }
        ]
        if chosen
        else []
    )
    return {"sets": sets, "unbreakable_path_indices": unbreakable, "truncated": truncated}


# =====================================================================
#  Ранжування наборів захисту інтегральною оцінкою J (формули 12-13)
# =====================================================================
#: Критерії за замовчуванням - та сама інтегральна оцінка (`aggregate` /
#: `oriented_score`), що й для шляхів атаки в `twin_bridge.DEFAULT_CRITERIA`,
#: щоб числа лишались зіставними в одному UI.
#:
#: - coverage (benefit): частка ПОДАНИХ `attack_paths`, які набір реально
#:   розриває. Рахується від ПОВНОГО списку шляхів, а не лише "розривних":
#:   шлях, який не розірве ЖОДЕН наявний засіб, лишається реальним ризиком і
#:   не повинен зникати з оцінки, наче це не провина набору.
#: - cost (cost): сумарна вартість control-вузлів у наборі відносно ліміту
#:   бюджету моделі (або, якщо бюджету не задано, відносно найдорожчого
#:   кандидата серед знайдених наборів - щоб критерій лишався в [0, 1]).
#: - size (cost): кількість заходів відносно `max_size`. За однакового
#:   покриття й вартості менше окремих ініціатив простіше впровадити й
#:   контролювати - розмір набору вартий власного критерію, не похідної
#:   вартості.
DEFAULT_DEFENSE_CRITERIA = (
    TwinCriterion("coverage", "частка шляхів атаки, які набір розриває", 0.5, "benefit"),
    TwinCriterion("cost", "сумарна вартість заходів відносно бюджету", 0.3, "cost"),
    TwinCriterion("size", "кількість окремих заходів у наборі", 0.2, "cost"),
)


def _cost_of(node) -> float:
    value = node.attributes.get("cost")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _defense_scores(
    model: TwinModel,
    controls: tuple[str, ...],
    total_paths: int,
    uncovered_count: int,
    cost_divisor: float,
    size_limit: int,
) -> list[dict[str, object]]:
    """Оцінки набору за кожним критерієм за замовчуванням, у шкалі [0, 1]."""
    coverage_score = (total_paths - uncovered_count) / total_paths if total_paths else 0.0
    total_cost = sum(_cost_of(model.node_by_id[control_id]) for control_id in controls)
    cost_score = min(total_cost / cost_divisor, 1.0) if cost_divisor else 0.0
    size_score = min(len(controls) / size_limit, 1.0) if size_limit else 0.0
    return [
        {"criterion": "coverage", "score": round(coverage_score, 6)},
        {"criterion": "cost", "score": round(cost_score, 6)},
        {"criterion": "size", "score": round(size_score, 6)},
    ]


def rank_defenses(
    model: TwinModel,
    attack_paths: list[list[int]],
    criteria: list[TwinCriterion] | None = None,
    max_size: int = 4,
) -> dict[str, object]:
    """Кандидати з `cut_sets`, оцінені інтегральним J і впорядковані так, щоб
    результат можна було показати аналітику, а не лише одне число.

    Обмеження моделі (`TwinConstraint`) НЕ звужують сам пошук покриття в
    `cut_sets` - покриття шляхів і бюджет це два різні питання ("чи розриває"
    і "чи можемо собі дозволити"). Змішати їх означало б мовчки ховати набір,
    що розриває шляхи, але не влазить у бюджет, замість чесно показати його
    аналітику окремо (`excluded_by_constraints`).

    ТРИ КОШИКИ, А НЕ ДВА. `defense_within_constraints` відповідає лише на
    питання "проходить усі ОЦІНЮВАНІ обмеження чи ні" - воно свідомо не
    розрізняє "пройшов, бо дані сходяться" від "пройшов, бо перевіряти
    нічим". Тут ця різниця стає видимою:
      - `ranked` - проходить усе, і жодна вимога моделі не лишилась
        неоцінюваною;
      - `excluded_by_constraints` - порушує ОЦІНЮВАНЕ обмеження (є дані, і
        вони не сходяться) - те саме "порушено", що й раніше;
      - `unevaluable_requirements` - проходить УСЕ оцінюване, але модель має
        вимогу (`unevaluable_constraints`), якої НІЧИМ підтвердити - третій,
        чесний стан замість автоматичного "порушено" від браку даних.
    Порушення ОЦІНЮВАНОГО обмеження має пріоритет над неоцінюваністю: набір,
    що одночасно виходить за бюджет І має неоцінювану regulatory-вимогу,
    лишається в `excluded_by_constraints` - "не послаблюй перевірку там, де
    дані є" (умова задачі). Список неоцінюваних вимог моделі однаковий для
    ВСІХ кандидатів (це властивість моделі, не конкретного набору), тож
    записується в кожен запис як `unevaluable_constraints` - аналітик бачить
    обидві причини одразу, навіть коли кошик визначила лише одна з них."""
    criteria = list(criteria) if criteria else list(DEFAULT_DEFENSE_CRITERIA)
    found = cut_sets(model, attack_paths, max_size=max_size)
    total_paths = len(attack_paths)
    size_limit = max(1, int(max_size))
    unevaluable = unevaluable_constraints(model)

    # Нормуємо вартість на ліміт бюджету моделі, якщо він заданий; інакше -
    # відносно найдорожчого кандидата серед знайдених наборів, щоб критерій
    # лишався в [0, 1] і без явного бюджетного обмеження в моделі.
    budget_limit = next(
        (
            constraint.limit
            for constraint in model.constraints
            if constraint.criterion == "budget" and constraint.limit > 0
        ),
        None,
    )
    candidate_costs = [
        sum(_cost_of(model.node_by_id[cid]) for cid in entry["controls"])
        for entry in found["sets"]
    ]
    cost_divisor = budget_limit or (max(candidate_costs) if candidate_costs else 0.0) or 1.0

    ranked: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    unevaluable_requirements: list[dict[str, object]] = []
    for entry in found["sets"]:
        controls = entry["controls"]
        scores = _defense_scores(
            model, controls, total_paths, len(entry["uncovered_path_indices"]),
            cost_divisor, size_limit,
        )
        integral, disagreement = aggregate(scores, criteria)
        record = {**entry, "j": integral, "disagreement": disagreement, "criteria": scores}
        within = defense_within_constraints(model, controls)
        record["within_constraints"] = within
        record["unevaluable_constraints"] = list(unevaluable)
        if not within:
            excluded.append(record)
        elif unevaluable:
            unevaluable_requirements.append(record)
        else:
            ranked.append(record)

    ranked.sort(key=lambda item: (-item["j"], item["size"], item["controls"]))
    excluded.sort(key=lambda item: (-item["j"], item["size"], item["controls"]))
    unevaluable_requirements.sort(key=lambda item: (-item["j"], item["size"], item["controls"]))

    return {
        "ranked": ranked,
        "excluded_by_constraints": excluded,
        "unevaluable_requirements": unevaluable_requirements,
        "unbreakable_path_indices": found["unbreakable_path_indices"],
        "truncated": found["truncated"],
    }
