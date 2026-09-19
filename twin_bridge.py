"""Міст: модель машини HTB -> типізований ОРІЄНТОВАНИЙ граф (`qa/twin.py`).

НАВІЩО. У `semantic_reasoner` граф неорієнтований, а ребро означає просту
співзустрічальність понять в одному документі. Наслідок видно на атласі:
«OpenSSL» і `root` стають прямими сусідами, обхід проскакує проміжні кроки, а
стрілка на схемі не означає переходу — це лише порядок читання. Через це
твердження про орієнтовану семантичну мережу й причинність спиралися на
модель, якої в коді не було.

ЗВІДКИ БЕРЕТЬСЯ НАПРЯМ. З іменованих полів вихідної моделі, і більше нізвідки:

    parent_id                     компонент -> що його містить
    runs_as                       компонент -> принципал, під яким він працює
    affected_component            компонент -> вразливість, яка в ньому є
    resulting_access              вразливість -> здобутий стан доступу
    permission_edges.subject/object  стан -> об'єкт, дія як назва ребра
    security_consequence          об'єкт -> стан, до якого веде дія
    artifacts.hosted_on           артефакт -> компонент
    exposure                      периметр -> компоненти, видимі ззовні

Жодне ребро не вигадується: ані LLM, ані статистикою. Після історії з
вигаданим T1190 (375 міток із 380 без опори в тексті) це не педантизм, а
єдиний спосіб не отримати той самий клас помилки на рівень вище — у структурі.

ЩО СЮДИ НЕ ПОТРАПЛЯЄ. `attack_path` — це еталон. Він не впливає на граф ніяк;
на цьому стоїть окремий тест (`test_gold_never_leaks_into_the_model`), бо без
такої гарантії порівняння з еталоном нічого не варте.
"""

from __future__ import annotations

from qa.twin import (
    MAX_DEPTH, TwinConstraint, TwinCriterion, TwinEdge, TwinModel, TwinNode,
    aggregate,
)
from semantic_reasoner import extract_terms


def _clean(value) -> str:
    """Порожні поля в описі позначені `-`; прибираємо, щоб не смітити текст."""
    text = str(value or "").strip()
    return "" if text in {"-", "n/a", "N/A"} else text

#: Синтетичний вузол периметра — позиція, з якої починає зовнішній аналітик.
#: Власних понять НЕ має: це не факт із моделі, і в метрику він лізти не мусить.
EXTERNAL_ID = "S_EXTERNAL"

#: Те саме визначення «видно ззовні», що й для точок входу неорієнтованої
#: гілки (`corpus.entry_concepts`) — інакше гілки отримали б різний периметр
#: і порівняння вимірювало б не структуру графа, а різні стартові умови.
_EXPOSED = ("public", "vpn", "internet", "external", "anonymous", "unauthenticated")

#: СТРУКТУРНІ відношення: описують устрій системи, але не є кроком атаки.
#: `hosted_on` каже «сервіс лежить на хості», а не «зламавши сервіс, ти дістав
#: хост». Якщо дати обходу по них їхати, він піднімається по вкладеності й
#: «здобуває» SYSTEM просто тому, що хост під ним працює під SYSTEM — та сама
#: вигадка, що й неорієнтованого графа, лише в орієнтованому одязі. Типізація
#: тим і корисна, що дозволяє назвати винувате ребро й відсікти його явно.
STRUCTURAL_RELATIONS = frozenset({"hosted_on", "stored_on"})

#: Відношення, які НЕ називають механізму переходу. `reaches` — це лише мережева
#: видимість, `runs_as` — статична властивість компонента. Крок, що називає
#: експлойт або дію, здійсненніший за твердження «сервіс просто працює під root».
GENERIC_RELATIONS = frozenset({"reaches", "runs_as"})

#: Критерії за замовчуванням для інтегральної оцінки J (формули 12-13).
#: Ваги сумуються в одиницю, полярність задана явно: «витрата» інвертується
#: перед згорткою, інакше моделі з різним формулюванням критеріїв незіставні.
DEFAULT_CRITERIA = (
    TwinCriterion("feasibility", "технічна здійсненність переходів", 0.4, "benefit"),
    TwinCriterion("evidence", "опертість ланцюга на опис інфраструктури", 0.3, "benefit"),
    TwinCriterion("effort", "витрати на реалізацію", 0.3, "cost"),
)


def _strip_domain(text: str) -> str:
    """`ACTIVE\Administrator` -> `Administrator`.

    Доменний префікс однаковий у всіх акаунтів машини й до того ж збігається з
    її назвою. Якщо його лишити, він стає «магнітом»: від `ACTIVE\SVC_TGS`
    після фільтра довжини лишається сам лише токен `active`, він виявляється
    підмножиною {active, administrator} — і два РІЗНІ принципали зливаються в
    один вузол.

    Ріжемо ЛИШЕ по зворотному слешу: це роздільник домену у Windows-принципалі.
    Прямий слеш у імені значить інше («charix/attacker» — «charix, тобто
    атакувальник»), і різати по ньому означає порвати ланцюг."""
    return text.rsplit("\\", 1)[-1] if "\\" in text else text


def _concepts(*parts) -> list[str]:
    """Поняття вузла в ТОМУ САМОМУ токенному просторі, що й еталон.

    Метрика зіставляє токени `extract_terms`; якщо вузли орієнтованого графа
    жили б в іншому просторі, його результат не можна було б порівняти з
    неорієнтованою гілкою."""
    seen: list[str] = []
    for part in parts:
        for token in extract_terms(_clean(part)):
            if token not in seen:
                seen.append(token)
    return seen


def _is_exposed(component: dict) -> bool:
    exposure = _clean(component.get("exposure")).lower()
    return any(marker in exposure for marker in _EXPOSED)


class _StateRegistry:
    """Стани доступу, зведені до названого принципала.

    «SYSTEM» з `identities`, «SYSTEM execution» з `security_consequence` і
    «NETWORK SERVICE write» з прав на артефакт — це ті самі принципали з
    уточненнями. Якби кожне формулювання народжувало власний вузол, граф
    рвався б рівно там, де це найдорожче: ціль ставала б недосяжною, і
    орієнтована гілка програвала б не по суті, а через дублікати.

    ПРАВИЛО ЗВЕДЕННЯ: два формулювання позначають той самий стан, якщо множина
    понять одного вкладена в множину понять іншого. Канонічною лишається
    КОРОТША — вона називає принципала, довша додає обставину переходу
    («execution», «write»), а обставина — властивість ребра, не стану.
    Правило застосовується однаково до всіх полів і еталона не бачить."""

    def __init__(self) -> None:
        self.order: list[TwinNode] = []

    def _match(self, concepts: list[str]) -> TwinNode | None:
        incoming = set(concepts)
        for node in self.order:
            existing = set(node.attributes.get("concepts", []))
            if existing <= incoming or incoming <= existing:
                return node
        return None

    def ensure(self, text, state_type: str = "privilege") -> str | None:
        # Префікс знімаємо ТІЛЬКИ тут: це імена принципалів. Назва компонента
        # цілком може містити «/» («browse.php / listfiles.php»), а шлях
        # артефакта — «\»; різати їх тим самим правилом означає викинути
        # половину назви й порвати ланцюг.
        concepts = _concepts(_strip_domain(_clean(text)))
        if not concepts:
            return None
        node = self._match(concepts)
        if node is not None:
            # Коротше формулювання перемагає: воно й є іменем принципала.
            if len(concepts) < len(node.attributes.get("concepts", [])):
                merged = TwinNode(id=node.id, kind=node.kind, label=_clean(text),
                                  attributes={**node.attributes,
                                              "concepts": concepts})
                self.order[self.order.index(node)] = merged
            return node.id
        node = TwinNode(
            id=f"S{len(self.order) + 1}",
            kind="state",
            label=_clean(text),
            attributes={"concepts": concepts, "state_type": state_type},
        )
        self.order.append(node)
        return node.id


def build_twin_model(machine: dict, goal_token: str = "") -> TwinModel:
    """Опис інфраструктури -> `TwinModel` з орієнтованими типізованими ребрами.

    `goal_token` — нормалізоване поняття цілі (`root`, `system`,
    `administrator`). У проді його називає аналітик, в експерименті воно
    береться з останнього кроку еталона. Модуль сам еталона НЕ читає й нічого
    про `attack_path` не знає — це і є та межа, яку не можна розмивати."""
    nodes: list[TwinNode] = []
    edges: list[TwinEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()

    kind_of: dict[str, str] = {}

    def link(source: str | None, relation: str, target: str | None) -> None:
        # Петлі й дублікати відкидаємо; зустрічне ребро НЕ додаємо ніде —
        # орієнтованість тут не декларація, а властивість побудови.
        if not source or not target or source == target:
            return
        key = (source, relation, target)
        if key in seen_edges:
            return
        seen_edges.add(key)
        # `runs_as` від ХОСТА переходом не є: дійти до хоста мережею — не те
        # саме, що виконати на ньому код. Ідентичність хоста здобувається
        # вразливістю або названою дією, тобто іншим ребром.
        transition = relation not in STRUCTURAL_RELATIONS
        if relation == "runs_as" and kind_of.get(source) == "asset":
            transition = False
        edges.append(TwinEdge(
            source=source, target=target, relation=relation,
            attributes={"weight": 1.0, "transition": transition},
        ))

    states = _StateRegistry()
    external = TwinNode(id=EXTERNAL_ID, kind="state",
                        label="external position",
                        attributes={"concepts": [], "state_type": "known_identity"})
    nodes.append(external)

    # --- принципали з `identities` ----------------------------------------
    for ident in machine.get("identities", []) or []:
        states.ensure(ident.get("principal"), "known_identity")

    # --- компоненти -------------------------------------------------------
    components = machine.get("components", []) or []
    for c in components:
        cid = _clean(c.get("component_id"))
        if not cid:
            continue
        kind = "asset" if "host" in _clean(c.get("layer_or_type")).lower() else "service"
        kind_of[cid] = kind
        nodes.append(TwinNode(
            id=cid, kind=kind, label=_clean(c.get("name")),
            attributes={"concepts": _concepts(c.get("name"), c.get("role"),
                                              c.get("software_or_version")),
                        "exposure": _clean(c.get("exposure"))},
        ))

    component_ids = {n.id for n in nodes if n.kind in ("asset", "service")}

    for c in components:
        cid = _clean(c.get("component_id"))
        if cid not in component_ids:
            continue
        parent = _clean(c.get("parent_id"))
        if parent in component_ids:
            link(cid, "hosted_on", parent)
        link(cid, "runs_as", states.ensure(c.get("runs_as"), "execution_context"))
        if _is_exposed(c):
            link(EXTERNAL_ID, "reaches", cid)

    # --- артефакти --------------------------------------------------------
    for i, a in enumerate(machine.get("artifacts", []) or []):
        aid = f"A{i + 1}"
        nodes.append(TwinNode(
            id=aid, kind="asset", label=_clean(a.get("artifact_or_path")),
            attributes={"concepts": _concepts(a.get("artifact_or_path"),
                                              a.get("purpose"))},
        ))
        host = _clean(a.get("hosted_on"))
        if host in component_ids:
            link(aid, "stored_on", host)
        writer = states.ensure(a.get("write_or_execute_access"), "execution_context")
        link(writer, "can_write", aid)

    artifact_ids = {f"A{i + 1}" for i in range(len(machine.get("artifacts", []) or []))}

    # --- вразливості ------------------------------------------------------
    for i, v in enumerate(machine.get("vulnerabilities", []) or []):
        vid = _clean(v.get("finding_id")) or f"V{i + 1}"
        nodes.append(TwinNode(
            id=vid, kind="vuln", label=_clean(v.get("issue")),
            attributes={"concepts": _concepts(v.get("issue"),
                                              v.get("affected_version")),
                        "external_ids": _clean(v.get("reference"))},
        ))
        affected = _clean(v.get("affected_component"))
        if affected in component_ids:
            link(affected, "has_vulnerability", vid)
        link(vid, "grants", states.ensure(v.get("resulting_access")))

    # --- права доступу ----------------------------------------------------
    for e in machine.get("permission_edges", []) or []:
        subject = states.ensure(e.get("subject"))
        obj = _clean(e.get("object"))
        if obj not in component_ids and obj not in artifact_ids:
            obj = None
        relation = _clean(e.get("permission_or_action")) or "acts_on"
        link(subject, relation, obj)
        link(obj, "yields", states.ensure(e.get("security_consequence")))

    nodes.extend(states.order)

    # --- ціль -------------------------------------------------------------
    goal_token = _clean(goal_token).lower()
    goal_id = ""
    for node in states.order:
        if goal_token and goal_token in node.attributes.get("concepts", []):
            goal_id = node.id
            break

    return TwinModel(
        name=_clean(machine.get("machine")) or "machine",
        nodes=nodes,
        edges=edges,
        criteria=list(DEFAULT_CRITERIA),
        constraints=[TwinConstraint("budget", 0.0)] if False else [],
        current_state_id=EXTERNAL_ID,
        goal_state_id=goal_id,
    )


def path_concepts(model: TwinModel, path) -> list[str]:
    """Шлях вузлами -> ланцюг токенів у порядку проходження.

    Потрібно, щоб орієнтовану гілку можна було зміряти ТІЄЮ САМОЮ метрикою
    (`metric.py`), що й неорієнтовану: та працює зі списком токенів."""
    by_id = model.node_by_id
    chain: list[str] = []
    for node_id in path:
        node = by_id.get(node_id)
        if node is None:
            continue
        for token in node.attributes.get("concepts", []) or []:
            if token not in chain:
                chain.append(token)
    return chain


def transition_edge_ids(model: TwinModel) -> list[int]:
    """Індекси ребер, якими взагалі можна ЙТИ як кроками сценарію.

    Структурні відношення лишаються в моделі — вони описують устрій системи й
    потрібні для читання результату, — але шляхом не стають."""
    return [i for i, edge in enumerate(model.edges)
            if edge.attributes.get("transition")]


# =====================================================================
#  Ранжування сценаріїв інтегральною оцінкою J (формули 12-13)
# =====================================================================
# НАВІЩО ВЛАСНЕ РАНЖУВАННЯ. `enumerate_scenarios` сортує кандидатів за
# `_treatment_count` — кількістю вузлів kind=`control`, що лікують вразливості.
# Це критерій із ІНШОГО застосування модуля (планування усунення), а в цьому
# графі control-вузлів немає взагалі, тож він завжди нуль і сортування
# вироджується в алфавітне за id вузлів. На Poison це ставило першим
# «external -> OpenSSH -> root» (покриття 0,5), тоді як справжній ланцюг із
# покриттям 1,0 стояв другим. «Перший шлях» тоді вимірює порядок ідентифікаторів,
# а не якість методу.
#
# ЖОДЕН критерій нижче не бачить еталона — рахуються лише зі структури шляху.
# Це перевіряється тестом `test_scenario_scores_never_look_at_the_gold`.

def score_scenario(model: TwinModel, path, edge_path) -> list[dict]:
    """Оцінки сценарію за кожним оголошеним критерієм, у шкалі [0, 1]."""
    relations = [model.edges[i].relation for i in edge_path or []]
    named = [r for r in relations if r not in GENERIC_RELATIONS]
    feasibility = len(named) / len(relations) if relations else 0.0

    real = [nid for nid in path if nid != EXTERNAL_ID]
    evidence = len(real) / len(path) if path else 0.0

    hops = max(len(path) - 1, 0)
    effort = min(hops / MAX_DEPTH, 1.0)

    return [
        {"criterion": "feasibility", "score": round(feasibility, 6)},
        {"criterion": "evidence", "score": round(evidence, 6)},
        {"criterion": "effort", "score": round(effort, 6)},
    ]


def rank_scenarios(model: TwinModel, scenarios) -> list[dict]:
    """Сценарії, упорядковані за спаданням J; дублі шляхів прибрані.

    Тай-брейк — сам шлях, щоб порядок не залежав від порядку надходження."""
    seen: set[tuple] = set()
    scored = []
    for scenario in scenarios or []:
        key = tuple(scenario.get("path") or ())
        if key in seen:
            continue
        seen.add(key)
        integral, disagreement = aggregate(
            score_scenario(model, scenario.get("path") or [],
                           scenario.get("edge_path") or []),
            model.criteria,
        )
        scored.append({**scenario, "j": integral, "disagreement": disagreement})
    scored.sort(key=lambda item: (-item["j"], tuple(item.get("path") or ())))
    return scored
