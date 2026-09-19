"""Адаптер: XML-опис корпоративної мережі -> `qa.twin.TwinModel`.

ЩО ЦЕ. `experiments/corporate_networks/dataset/` — 100 згенерованих мереж
(`net-001.xml` … `net-100.xml`), кожна описує топологію (підмережі, активи,
служби, вразливості, облікові записи, засоби захисту, потоки, політику) і
задачу обходу (`<task>`: звідки атакувати, якого стану досягти). Цей модуль
перетворює один XML-файл на типізований ОРІЄНТОВАНИЙ граф `TwinModel` —
той самий формат, що `twin_bridge.py` будує для HTB-машин (звідти й
запозичено стиль: напрям і тип ребра беруться лише з НАЗВАНИХ полів вихідних
даних, ніколи не вигадуються).

ПРИХОВАНИЙ ЕТАЛОН. Кожна мережа несе блок `<attacks>` — фактичні ланцюги
компрометації з бальною оцінкою в `answers.json`. Він не повинен впливати на
побудову моделі НІЯК: ні вузлом, ні ребром, ні атрибутом. Тому XML читають
ДВІ окремі функції з різними іменами:

    parse_network(xml_text)       -> все, КРІМ `<attacks>`
    parse_gold_attacks(xml_text)  -> ЛИШЕ `<attacks>`

`parse_network` фізично вилучає вузол `<attacks>` з дерева ДО того, як
починається будь-який обхід — так навіть майбутня необережна правка коду
не зможе випадково зачепити його. Гарантує це
`tests/test_corporate_dataset.py::test_gold_never_leaks_into_the_model`
(та сама дисципліна, що вже діє для `attack_path` у
`experiments/htb_reconstruction/corpus.py`).

ЗІСТАВЛЕННЯ MITIGATES. У схемі немає жодного поля, яке напряму пов'язує
`<defense>` з конкретною вразливістю чи exploitClass (ні CVE, ні спільного
ідентифікатора). Вигадувати таку прив'язку (наприклад «edr закриває
privilege-escalation») означало б повторити історію з вигаданим T1190 —
мітка виглядає правдоподібно, але не опирається на дані. Єдине, що дані
РЕАЛЬНО стверджують, — це належність засобу захисту й вразливості ОДНОМУ Й
ТОМУ САМОМУ активу (обидва — діти одного `<asset>`), а РІД відношення
береться за категорією захисту (`kind`), а не за CVE: `siem-agent`
спостерігає, `backup` відновлює, решта — запобігають. Це той самий поділ
mitigates/detects/recovers_from, що вже закладений у `qa.twin.TREATMENT_
RELATIONS` і використаний у `qa/fixtures/corporate_12node.yaml`
(ctrl-siem -> detects, ctrl-backup -> recovers_from, решта -> mitigates).
Ребро додається лише для `state="enabled"`: вимкнений засіб не захищає
нічого, а для «partial» ефективність неоднозначна — «краще не додати, ніж
додати хибне».
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from qa.twin import TwinConstraint, TwinCriterion, TwinEdge, TwinModel, TwinNode


def _clean(value) -> str:
    return str(value or "").strip()


def _as_bool(value) -> bool:
    return _clean(value).lower() == "true"


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# =====================================================================
#  parse_network — усе, КРІМ прихованого еталона
# =====================================================================

def _parse_vulnerability(elem: ET.Element) -> dict:
    return {
        "cve": _clean(elem.get("cve")),
        "exploit_class": _clean(elem.get("exploitClass")),
        "exploitability": _as_float(elem.get("exploitability")),
        "cvss": _as_float(elem.get("cvss")),
        "severity": _clean(elem.get("severity")),
        "cvss_version": _clean(elem.get("cvssVersion")),
        "cvss_source": _clean(elem.get("cvssSource")),
        "cvss_vector": _clean(elem.get("cvssVector")),
        "attack_vector": _clean(elem.get("attackVector")),
        "privileges_required": _clean(elem.get("privilegesRequired")),
        "user_interaction": _clean(elem.get("userInteraction")),
        "mitre": [_clean(m.get("technique")) for m in elem.findall("mitre")],
        "cwe": [_clean(c.get("id")) for c in elem.findall("cwe")],
    }


def _parse_service(elem: ET.Element) -> dict:
    return {
        "name": _clean(elem.get("name")),
        "product": _clean(elem.get("product")),
        "version": _clean(elem.get("version")),
        "port": _clean(elem.get("port")),
        "protocol": _clean(elem.get("protocol")),
        "vulnerabilities": [_parse_vulnerability(v) for v in elem.findall("vulnerability")],
    }


def _parse_account(elem: ET.Element) -> dict:
    return {
        "name": _clean(elem.get("name")),
        "privilege": _clean(elem.get("privilege")),
        "domain": _as_bool(elem.get("domain")),
        # `<reusedOn asset="..."/>` — той самий обліковий запис валідний і на
        # іншому активі (наприклад вкрадений локальний адмін-пароль
        # повторюється). Беремо ЛИШЕ те, що поле стверджує напряму.
        "reused_on": [_clean(r.get("asset")) for r in elem.findall("reusedOn")],
    }


def _parse_defense(elem: ET.Element) -> dict:
    return {
        "kind": _clean(elem.get("kind")),
        "product": _clean(elem.get("product")),
        "state": _clean(elem.get("state")),
    }


def _parse_asset(elem: ET.Element) -> dict:
    os_elem = elem.find("os")
    services_elem = elem.find("services")
    osvuln_elem = elem.find("osVulnerabilities")
    accounts_elem = elem.find("accounts")
    defenses_elem = elem.find("defenses")
    return {
        "id": _clean(elem.get("id")),
        "role": _clean(elem.get("role")),
        "hostname": _clean(elem.get("hostname")),
        "subnet": _clean(elem.get("subnet")),
        "ip": _clean(elem.get("ip")),
        "criticality": _clean(elem.get("criticality")),
        "os": {
            "family": _clean(os_elem.get("family")) if os_elem is not None else "",
            "product": _clean(os_elem.get("product")) if os_elem is not None else "",
            "version": _clean(os_elem.get("version")) if os_elem is not None else "",
        },
        "services": (
            [_parse_service(s) for s in services_elem.findall("service")]
            if services_elem is not None else []
        ),
        "os_vulnerabilities": (
            [_parse_vulnerability(v) for v in osvuln_elem.findall("vulnerability")]
            if osvuln_elem is not None else []
        ),
        "accounts": (
            [_parse_account(a) for a in accounts_elem.findall("account")]
            if accounts_elem is not None else []
        ),
        "defenses": (
            [_parse_defense(d) for d in defenses_elem.findall("defense")]
            if defenses_elem is not None else []
        ),
    }


def parse_network(xml_text: str) -> dict:
    """XML однієї мережі -> словник БЕЗ прихованого еталона.

    `<attacks>` вилучається з дерева ПЕРШИМ рядком, до будь-якого обходу —
    навіть якщо нижче з'явиться помилка чи код почне обходити дерево
    цілком, вирізаного вузла фізично немає й прочитати його неможливо."""
    root = ET.fromstring(xml_text)
    attacks_elem = root.find("attacks")
    if attacks_elem is not None:
        root.remove(attacks_elem)

    subnets = []
    subnets_elem = root.find("subnets")
    if subnets_elem is not None:
        for s in subnets_elem.findall("subnet"):
            subnets.append({
                "id": _clean(s.get("id")),
                "cidr": _clean(s.get("cidr")),
                "zone": _clean(s.get("zone")),
                "trust": _clean(s.get("trust")),
            })

    assets = []
    assets_elem = root.find("assets")
    if assets_elem is not None:
        for a in assets_elem.findall("asset"):
            assets.append(_parse_asset(a))

    links = []
    links_elem = root.find("links")
    if links_elem is not None:
        for link_elem in links_elem.findall("link"):
            links.append({
                "src": _clean(link_elem.get("src")),
                "dst": _clean(link_elem.get("dst")),
                "kind": _clean(link_elem.get("kind")),
            })

    flows = []
    flows_elem = root.find("flows")
    if flows_elem is not None:
        for f in flows_elem.findall("flow"):
            flows.append({
                "src": _clean(f.get("src")),
                "dst": _clean(f.get("dst")),
                "kind": _clean(f.get("kind")),
                "port": _clean(f.get("port")),
            })

    policy = []
    policy_elem = root.find("policy")
    if policy_elem is not None:
        for r in policy_elem.findall("rule"):
            policy.append({
                "src_zone": _clean(r.get("srcZone")),
                "dst_zone": _clean(r.get("dstZone")),
                "protocol": _clean(r.get("protocol")),
                "ports": _clean(r.get("ports")),
                "action": _clean(r.get("action")),
            })

    task: dict[str, object] = {}
    task_elem = root.find("task")
    if task_elem is not None:
        init_elem = task_elem.find("initialState")
        goal_elem = task_elem.find("goal")
        constraints_elem = task_elem.find("constraints")
        task = {
            "initial_asset": _clean(init_elem.get("asset")) if init_elem is not None else "",
            "initial_condition": _clean(init_elem.get("condition")) if init_elem is not None else "",
            "initial_cve": _clean(init_elem.get("viaCve")) if init_elem is not None else "",
            "goal_asset": _clean(goal_elem.get("asset")) if goal_elem is not None else "",
            "goal_condition": _clean(goal_elem.get("condition")) if goal_elem is not None else "",
            "constraints": (
                [_clean(c.text) for c in constraints_elem.findall("constraint")]
                if constraints_elem is not None else []
            ),
        }

    return {
        "id": _clean(root.get("id")),
        "profile": _clean(root.get("profile")),
        "topology": _clean(root.get("topology")),
        "subnets": subnets,
        "assets": assets,
        "links": links,
        "flows": flows,
        "policy": policy,
        "task": task,
    }


# =====================================================================
#  parse_gold_attacks — ЛИШЕ прихований еталон
# =====================================================================

def parse_gold_attacks(xml_text: str) -> list[dict]:
    """XML однієї мережі -> список атак з `<attacks>`, і БІЛЬШЕ НІЧОГО.

    Симетрична пара до `parse_network`: якщо той будує модель "наосліп"
    щодо еталона, ця функція, навпаки, читає ВИКЛЮЧНО еталон — оцінювання
    результату і побудова моделі фізично не можуть переплутати джерела."""
    root = ET.fromstring(xml_text)
    attacks_elem = root.find("attacks")
    if attacks_elem is None:
        return []
    result = []
    for attack in attacks_elem.findall("attack"):
        steps = []
        for step in attack.findall("step"):
            steps.append({
                "n": int(_clean(step.get("n")) or 0),
                "from": _clean(step.get("from")),
                "to": _clean(step.get("to")),
                "cve": _clean(step.get("cve")),
                "exploit_class": _clean(step.get("exploitClass")),
                "via": _clean(step.get("via")),
                "mitre": [_clean(m.get("technique")) for m in step.findall("mitre")],
            })
        result.append({"id": _clean(attack.get("id")), "steps": steps})
    return result


# =====================================================================
#  build_twin_model — словник із parse_network -> TwinModel
# =====================================================================

#: Критерії за замовчуванням для J (формули 12-13 паперу), за зразком
#: `twin_bridge.DEFAULT_CRITERIA`: та сама структурна ідея (рахуються ЛИШЕ
#: зі структури шляху), бо жодних інших даних (вартість, довіра оцінювача)
#: цей датасет не постачає.
DEFAULT_CRITERIA = (
    TwinCriterion(
        "feasibility",
        "частка переходів, що називають конкретний механізм (вразливість, "
        "дозволений потік чи політику), а не просто мережеву досяжність",
        0.4, "benefit",
    ),
    TwinCriterion(
        "evidence",
        "частка вузлів шляху, що є реальними активами мережі, а не "
        "синтетичними вузлами периметра чи стану",
        0.3, "benefit",
    ),
    TwinCriterion(
        "effort",
        "довжина шляху, нормована на MAX_DEPTH",
        0.3, "cost",
    ),
)

#: Категорія лікування за `kind` захисту. Обґрунтування — у docstring
#: модуля («ЗІСТАВЛЕННЯ MITIGATES»). Значення обмежені `TREATMENT_RELATIONS`
#: з `qa/twin.py`.
_DEFENSE_RELATION = {
    "firewall-host": "mitigates",
    "ids": "mitigates",
    "edr": "mitigates",
    "av": "mitigates",
    "mfa": "mitigates",
    "siem-agent": "detects",
    "backup": "recovers_from",
}

#: Синтетичний вузол периметра: "internet" — зона з правил політики, якій
#: НЕ відповідає жодна `<subnet>` (на відміну від dmz/lan/server/management/
#: wifi). Той самий прийом, що й `EXTERNAL_ID` у `twin_bridge.py`.
ZONE_INTERNET = "internet"
_INTERNET_NODE_ID = "ZONE:internet"


# =====================================================================
#  Переходи атаки: досяжність активу з активу + exploitClass -> умова
# =====================================================================
# ПРОБЛЕМА, яку цей блок закриває: `build_twin_model` описував СТРУКТУРУ
# мережі (де що лежить, які вразливості є), але жодне ребро не вело від
# стану, який атакувальник УЖЕ здобув, до наступного кроку — виміряно, що
# `STATE:web-01:compromised` мав 0 вихідних ребер, а `enumerate_scenarios`
# повертав 0 шляхів на ВСІХ 100 мережах. Нижче з ПУБЛІЧНИХ полів (підмережі,
# `<flow>`, `<policy>`, облікові записи) виводиться, який актив досяжний з
# якого і яку умову дає експлуатація вразливості — і БІЛЬШЕ НІЧОГО з
# `<attacks>` (`test_reachability_transitions_do_not_depend_on_the_attacks_
# block` стереже саме це).

#: МОДЕЛЬНЕ ПРИПУЩЕННЯ (не факт із даних): яку умову атакувальник здобуває,
#: експлуатуючи вразливість певного `exploitClass`. Датасет називає лише
#: КАТЕГОРІЮ техніки (initial-access/lateral-movement/credential-access) —
#: не конкретний наслідок на активі. `initial-access` і `lateral-movement`
#: зведені до однієї умови ("compromised"), бо exploitClass їх за силою
#: впливу не розрізняє; `credential-access` веде до "data-exfiltration" —
#: єдиної умови датасету, що семантично про вилучення й каже.
#: `privilege-escalation` СВІДОМО не тут — його наслідок залежить від
#: активу (`_privilege_escalation_condition`), а не є константою.
#: ЗАМІННІСТЬ: це єдина точка, де живе припущення — підміна словника (і, за
#: потреби, функції нижче) змінює ВЕСЬ подальший висновок моделі без правок
#: деінде.
EXPLOIT_CLASS_CONDITION = {
    "initial-access": "compromised",
    "lateral-movement": "compromised",
    "credential-access": "data-exfiltration",
}


def _privilege_escalation_condition(asset: dict) -> str:
    """`privilege-escalation` -> яку умову дає ескалація на КОНКРЕТНОМУ активі.

    МОДЕЛЬНЕ ПРИПУЩЕННЯ (не факт із даних): якщо на активі є обліковий запис
    з `privilege="domain-admin"`, ескалація приносить саме його — сам актив
    підтверджує, що такий рівень тут узагалі існує; інакше — узагальнений
    "local-admin". Дані НЕ називають, який саме привілей дає КОНКРЕТНА
    вразливість (лише exploitClass), тож це наближення, а не цитата з
    джерела — воно може помилятися там, де реальна ескалація обмежена
    нижчим рівнем навіть за наявності domain-admin облікового запису на
    активі."""
    return "domain-admin" if any(
        a.get("privilege") == "domain-admin" for a in asset.get("accounts", [])
    ) else "local-admin"


def _exploit_class_condition(exploit_class: str, asset: dict) -> str:
    """`exploitClass` (+ актив, для privilege-escalation) -> здобута умова.

    Порожній рядок означає "немає відомого наслідку": для exploitClass поза
    чотирма очікуваними значеннями код НЕ вгадує умову й просто не будує
    перехід — мовчати безпечніше, ніж вигадати."""
    if exploit_class == "privilege-escalation":
        return _privilege_escalation_condition(asset)
    return EXPLOIT_CLASS_CONDITION.get(exploit_class, "")


#: Назви трьох правил досяжності активу з активу — іменовані константи, а
#: не рядкові літерали в коді нижче, щоб звіт "скільки ребер дало кожне
#: правило" і сама побудова моделі не могли розійтися в написанні.
REACH_SAME_SUBNET = "same_subnet"
REACH_FLOW = "flow"
REACH_POLICY = "policy"


def _same_subnet_pairs(assets: list[dict]) -> set[tuple[str, str]]:
    """Правило 1: активи ОДНІЄЇ підмережі досяжні один з одного (L2-сегмент).

    Найчастіший випадок у цьому датасеті — і водночас той, якого побудова
    моделі раніше НЕ емітила: `located_in` каже лише "актив лежить у
    підмережі", а не "з підмережі видно інший актив"."""
    by_subnet: dict[str, list[str]] = {}
    for asset in assets:
        by_subnet.setdefault(asset["subnet"], []).append(asset["id"])
    return {
        (a, b)
        for members in by_subnet.values()
        for a in members
        for b in members
        if a != b
    }


def _flow_pairs(flows: list[dict], asset_ids: set[str]) -> set[tuple[str, str]]:
    """Правило 2: явний `<flow src dst>` між двома активами — найконкретніше
    з трьох тверджень (називає точну пару хостів, а не сегмент чи зону)."""
    return {
        (f["src"], f["dst"])
        for f in flows
        if f["src"] in asset_ids and f["dst"] in asset_ids and f["src"] != f["dst"]
    }


def _policy_pairs(
    policy: list[dict], assets: list[dict], subnet_zone: dict[str, str]
) -> set[tuple[str, str]]:
    """Правило 3: `<rule srcZone dstZone action="allow"/>` — найширше з
    трьох тверджень: правило називає ЗОНИ, а не активи, тож застосовується
    до кожної впорядкованої пари активів цих двох зон."""
    assets_by_zone: dict[str, list[str]] = {}
    for asset in assets:
        zone = subnet_zone.get(asset["subnet"], "")
        if zone:
            assets_by_zone.setdefault(zone, []).append(asset["id"])
    pairs: set[tuple[str, str]] = set()
    for rule in policy:
        if rule["action"] != "allow":
            continue
        for a in assets_by_zone.get(rule["src_zone"], []):
            for b in assets_by_zone.get(rule["dst_zone"], []):
                if a != b:
                    pairs.add((a, b))
    return pairs


def _asset_reachability(network: dict) -> dict[tuple[str, str], str]:
    """Досяжність активу з активу за трьома публічними правилами -> яке
    правило її підтвердило.

    Кожне правило обчислюється НЕЗАЛЕЖНО, а тег на результат — правило з
    найвищим пріоритетом серед тих, що підтвердили пару (`flow` >
    `same_subnet` > `policy`, від найконкретнішого твердження до
    найширшого). Без цього одна пара породжувала б кілька паралельних
    ребер до того самого вузла вразливості, а звіт "скільки ребер дало
    кожне правило" перестав би бути розбиттям без перетину."""
    assets = network.get("assets", [])
    asset_ids = {a["id"] for a in assets}
    subnet_zone = {s["id"]: s["zone"] for s in network.get("subnets", [])}
    result: dict[tuple[str, str], str] = {}
    for pair in _policy_pairs(network.get("policy", []), assets, subnet_zone):
        result[pair] = REACH_POLICY
    for pair in _same_subnet_pairs(assets):
        result[pair] = REACH_SAME_SUBNET
    for pair in _flow_pairs(network.get("flows", []), asset_ids):
        result[pair] = REACH_FLOW
    return result


# =====================================================================
#  Обмеження задачі: `<task><constraints>` -> `qa.twin.TwinConstraint` /
#  текст. Перед написанням цього коду перевірено ВЕСЬ набір формулювань,
#  що трапляються в 100 файлах (`grep` по `<constraint>` дослівно, а не
#  вгадано наперед) — частоти:
#      46 без зупинки виробничих сервісів
#      44 збереження доказової бази для розслідування
#      44 без залучення зовнішнього підрядника
#      40 час реагування не більше 4 годин
#      29 відповідність ISO/IEC 27035
#  Жодних інших формулювань і жодної варіації числа чи одиниці всередині
#  кожного (усі 40 "час реагування" — дослівно "4 годин").
# =====================================================================

#: Єдине формулювання, що зводиться ОДНОЗНАЧНО: числа в ньому немає, але
#: за змістом це вимога регуляторної відповідності стандарту (ISO/IEC
#: 27035 — управління інцидентами ІБ), тож природно лягає на готовий
#: прапорець `regulatory` з `qa.twin.ALLOWED_CONSTRAINTS` (`limit=1.0`
#: активує перевірку `regulatory_approved` в `qa.twin._within_constraints`).
REGULATORY_CONSTRAINT_TEXT = "відповідність ISO/IEC 27035"


def classify_task_constraints(texts: list[str]) -> tuple[list[TwinConstraint], list[str]]:
    """Сирі рядки `<constraint>` задачі (`parse_network`) -> (числові
    `TwinConstraint`, текст, що НЕ звівся до числа).

    Зведення ОДНОЗНАЧНЕ лише для `REGULATORY_CONSTRAINT_TEXT` (докстрінг
    вище). РЕШТА свідомо НЕ зводиться — "краще не звести, ніж звести
    навмання":

      - "час реагування не більше 4 годин" — МАЄ явне число й одиницю
        (4, години), але одиниця не має відповідника серед
        `qa.twin.ALLOWED_CONSTRAINTS`: там лише "months" (місяці) як
        часовий критерій. Перерахувати години в частку місяця означало б
        вигадати курс перерахунку, якого дані не дають. Додати обмеження
        з ВЛАСНОЮ назвою (наприклад "response_time_hours") теж не рятує:
        `qa.twin._within_constraints` перевіряє ЛИШЕ жорстко зашиті рядки
        "budget"/"months"/"personnel" (`CONSTRAINT_ATTRIBUTE`) плюс
        окрему гілку "regulatory" — обмеження з будь-якою ІНШОЮ назвою
        мовчки НІКОЛИ ніде не спрацює (`qa/twin.py` редагувати заборонено
        умовою задачі), а в `model.constraints` виглядатиме як перевірене.
        Мертвий, ніким не застосовуваний запис гірший за чесний текст.
      - "без зупинки виробничих сервісів", "без залучення зовнішнього
        підрядника", "збереження доказової бази для розслідування" — не
        мають числа взагалі й не описують жодної з чотирьох категорій
        `ALLOWED_CONSTRAINTS` (бюджет/термін/персонал/регуляторність) —
        це правила-виключення чи процедурні вимоги, не ліміти.

    Лишений текст призначений для вузлів kind="requirement"
    (`_task_requirement_nodes` нижче) — вимога не зникає з моделі, лише
    не претендує на число, якого дані не підтверджують."""
    constraints: list[TwinConstraint] = []
    leftover: list[str] = []
    for raw in texts:
        text = _clean(raw)
        if not text:
            continue
        if text == REGULATORY_CONSTRAINT_TEXT:
            constraints.append(TwinConstraint(criterion="regulatory", limit=1.0))
            continue
        leftover.append(text)
    return constraints, leftover


def _task_requirement_nodes(network_id: str, texts: list[str]) -> list[TwinNode]:
    """Текстові вимоги задачі, що НЕ звелися до числа, -> вузли
    kind="requirement" (категорія вже передбачена `qa.twin.ALLOWED_KINDS`
    саме для такого випадку). Вузли навмисно НЕ приєднуються жодним
    ребром: це метадана про задачу, а не крок атаки чи захисту, і
    приєднання до графа вигадало б відношення, якого дані не називають."""
    return [
        TwinNode(
            id=f"REQ:{network_id}:{i}",
            kind="requirement",
            label=text[:256],
            attributes={"text": text},
        )
        for i, text in enumerate(texts)
    ]


def build_twin_model(network: dict) -> TwinModel:
    """Словник із `parse_network` -> `TwinModel` з орієнтованими типізованими
    ребрами. Модуль НЕ приймає й не бачить `<attacks>` — вхідний словник
    його фізично не містить (гарантія `parse_network`)."""
    nodes: list[TwinNode] = []
    edges: list[TwinEdge] = []
    seen_edges: set[tuple] = set()

    def link(source, relation, target, transition, **attrs) -> None:
        # Петлі відкидаємо; зустрічне ребро НЕ додаємо ніде — орієнтованість
        # тут властивість побудови, а не декларація.
        if not source or not target or source == target:
            return
        key = (source, relation, target, tuple(sorted(attrs.items())))
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append(TwinEdge(
            source=source, target=target, relation=relation,
            attributes={"weight": 1.0, "transition": transition, **attrs},
        ))

    # STATE:<asset>:<condition> — рівно ОДИН вузол на пару (актив, умова),
    # незалежно від того, скільки різних джерел її називають (задача через
    # <initialState>/<goal>, вразливість через exploitClass). Потрібно
    # заздалегідь, бо і той, і той шлях створення мають сходитись в одному
    # вузлі, а не дублювати його (тест `test_no_duplicate_node_ids`).
    condition_state_nodes: dict[tuple[str, str], str] = {}

    def ensure_condition_state(asset_id: str, condition: str) -> str:
        key = (asset_id, condition)
        node_id = condition_state_nodes.get(key)
        if node_id is not None:
            return node_id
        node_id = f"STATE:{asset_id}:{condition}"
        condition_state_nodes[key] = node_id
        # `security_state` — єдиний тип у ALLOWED_STATE_TYPES, що не
        # претендує на конкретну природу стану (сесія/контекст виконання/
        # привілей): `condition` датасету буває і привілеєм
        # ("domain-admin"), і наслідком дії ("data-exfiltration",
        # "phishing-landing"), тож форсувати його у вужчу категорію
        # означало б вигадувати те, чого поле не каже.
        nodes.append(TwinNode(
            id=node_id, kind="state", label=f"{asset_id}: {condition}",
            attributes={
                "state_type": "security_state", "asset": asset_id,
                "condition": condition,
            },
        ))
        return node_id

    # --- периметр: синтетична зона "internet" -----------------------------
    nodes.append(TwinNode(
        id=_INTERNET_NODE_ID, kind="resource", label="Internet (периметр)",
        attributes={"zone": ZONE_INTERNET},
    ))
    # Дві РІЗНІ мапи навмисно: `<asset subnet="...">` посилається на id
    # підмережі, а `<rule srcZone/dstZone>` — на її ЗОНУ; у net-001 підмережа
    # mgmt має zone="management" — id і зона розходяться, тож зводити їх в
    # одну мапу означало б випадково ламати правила саме там, де вони
    # різняться.
    subnet_node_by_id: dict[str, str] = {}
    zone_node: dict[str, str] = {ZONE_INTERNET: _INTERNET_NODE_ID}

    for subnet in network.get("subnets", []):
        node_id = f"SUBNET:{subnet['id']}"
        nodes.append(TwinNode(
            id=node_id, kind="resource",
            label=f"{subnet['zone']} ({subnet['cidr']})",
            attributes={"cidr": subnet["cidr"], "zone": subnet["zone"], "trust": subnet["trust"]},
        ))
        subnet_node_by_id[subnet["id"]] = node_id
        zone_node[subnet["zone"]] = node_id

    asset_ids = {a["id"] for a in network.get("assets", [])}
    asset_vuln_ids: dict[str, list[str]] = {a["id"]: [] for a in network.get("assets", [])}
    # vuln-вузол -> його вихідний словник (cve/exploit_class/…) — потрібно
    # нижче, у "переходах атаки", щоб зіставити exploitClass з умовою, яку
    # вразливість дає, не перечитуючи XML вдруге.
    vuln_by_id: dict[str, dict] = {}

    # --- активи -------------------------------------------------------
    for asset in network.get("assets", []):
        nodes.append(TwinNode(
            id=asset["id"], kind="asset",
            label=f"{asset['hostname']} ({asset['role']})",
            attributes={
                "role": asset["role"], "criticality": asset["criticality"],
                "ip": asset["ip"], "subnet": asset["subnet"],
                "os_family": asset["os"]["family"], "os_product": asset["os"]["product"],
                "os_version": asset["os"]["version"],
                # Служби НЕ стають окремими вузлами: у цьому датасеті flow й
                # policy оперують на рівні активу/зони, а не порту, тож
                # окремий вузол на кожен сервіс (до 80 на мережу) роздував би
                # граф, не додаючи жодного нового відношення для обходу.
                # Текст лишаємо як атрибут, щоб інформація не губилась.
                "services": [
                    f"{s['name']} {s['product']} {s['version']} :{s['port']}/{s['protocol']}".strip()
                    for s in asset["services"]
                ],
            },
        ))

    # --- ребро "розміщення активу в підмережі" -----------------------------
    for asset in network.get("assets", []):
        target = subnet_node_by_id.get(asset["subnet"])
        if target:
            link(asset["id"], "located_in", target, transition=False)

    # --- вразливості ----------------------------------------------------
    def add_vulnerabilities(asset_id: str, vulns: list[dict], context: str) -> None:
        for v in vulns:
            if not v["cve"]:
                continue
            vid = f"VULN:{asset_id}:{v['cve']}"
            nodes.append(TwinNode(
                id=vid, kind="vuln", label=f"{v['cve']} ({v['exploit_class']})",
                attributes={
                    "cve": v["cve"], "cvss": v["cvss"], "exploitability": v["exploitability"],
                    "exploit_class": v["exploit_class"], "mitre": v["mitre"], "cwe": v["cwe"],
                    "severity": v["severity"], "context": context,
                    "external_ids": [v["cve"]],
                },
            ))
            asset_vuln_ids[asset_id].append(vid)
            vuln_by_id[vid] = v
            link(asset_id, "has_vuln", vid, transition=True)

    for asset in network.get("assets", []):
        aid = asset["id"]
        for service in asset["services"]:
            add_vulnerabilities(
                aid, service["vulnerabilities"],
                context=f"service:{service['name']}:{service['port']}",
            )
        add_vulnerabilities(aid, asset["os_vulnerabilities"], context="os")

    # --- потоки -------------------------------------------------------
    for flow in network.get("flows", []):
        if flow["src"] in asset_ids and flow["dst"] in asset_ids:
            link(flow["src"], "flow", flow["dst"], transition=True,
                 kind=flow["kind"], port=flow["port"])

    # --- дозволи політики (лише allow; deny доступу не надає) --------------
    for rule in network.get("policy", []):
        if rule["action"] != "allow":
            continue
        src = zone_node.get(rule["src_zone"])
        dst = zone_node.get(rule["dst_zone"])
        if src and dst:
            link(src, "policy_allows", dst, transition=True,
                 protocol=rule["protocol"], ports=rule["ports"])

    # --- облікові записи ------------------------------------------------
    # domain=true -> ідентичність рівня домену: те саме ім'я на різних
    # активах МЕРЕЖІ — той самий принципал (доменний обліковий запис єдиний
    # в AD), тож зводимо в один вузол і додаємо ребро "has_account" від
    # КОЖНОГО активу, що його називає (перевірено емпірично на всіх 100
    # мережах: жодного випадку, де те саме ім'я з domain=true мало б різний
    # `privilege` в межах однієї мережі). domain=false -> локальний обліковий
    # запис, прив'язаний до конкретного хоста за конвенцією (`svc-<host>`);
    # однакове ім'я на різних хостах тут НЕ факт спільної ідентичності.
    domain_account_node: dict[str, str] = {}

    def account_state_id(account: dict, asset_id: str) -> str:
        if account["domain"]:
            node_id = domain_account_node.get(account["name"])
            if node_id is None:
                node_id = f"ACCT:{account['name']}"
                domain_account_node[account["name"]] = node_id
                nodes.append(TwinNode(
                    id=node_id, kind="state",
                    label=f"{account['name']} ({account['privilege']})",
                    attributes={
                        "state_type": "privilege", "privilege": account["privilege"],
                        "domain": True,
                    },
                ))
            return node_id
        node_id = f"ACCT:{asset_id}:{account['name']}"
        nodes.append(TwinNode(
            id=node_id, kind="state",
            label=f"{account['name']} ({account['privilege']}) @ {asset_id}",
            attributes={
                "state_type": "privilege", "privilege": account["privilege"],
                "domain": False,
            },
        ))
        return node_id

    for asset in network.get("assets", []):
        aid = asset["id"]
        for account in asset["accounts"]:
            node_id = account_state_id(account, aid)
            link(aid, "has_account", node_id, transition=False)
            # `reusedOn`: той самий обліковий запис валідний і на іншому
            # активі. Ребро додається ЛИШЕ тому, що поле це стверджує прямо —
            # навіть якщо на цільовому активі власного <account> з такою
            # назвою немає.
            for reused_asset in account["reused_on"]:
                if reused_asset in asset_ids:
                    link(reused_asset, "has_account", node_id, transition=False)

    # --- засоби захисту -------------------------------------------------
    for asset in network.get("assets", []):
        aid = asset["id"]
        for i, defense in enumerate(asset["defenses"]):
            cid = f"CTRL:{aid}:{i}"
            nodes.append(TwinNode(
                id=cid, kind="control",
                label=f"{defense['product']} ({defense['kind']})",
                attributes={"kind": defense["kind"], "state": defense["state"]},
            ))
            if defense["state"] != "enabled":
                continue
            relation = _DEFENSE_RELATION.get(defense["kind"])
            if not relation:
                continue
            for vid in asset_vuln_ids.get(aid, []):
                link(cid, relation, vid, transition=False)

    # --- початковий і цільовий стан ------------------------------------
    task = network.get("task", {}) or {}
    initial_asset = task.get("initial_asset", "")
    initial_condition = task.get("initial_condition", "")
    goal_asset = task.get("goal_asset", "")
    goal_condition = task.get("goal_condition", "")

    current_state_id = ensure_condition_state(initial_asset, initial_condition)
    goal_state_id = ensure_condition_state(goal_asset, goal_condition)

    # --- обмеження задачі: <task><constraints> -> TwinConstraint / текст ---
    task_constraints, unreduced_constraint_texts = classify_task_constraints(
        task.get("constraints", []) or []
    )
    nodes.extend(_task_requirement_nodes(network.get("id", ""), unreduced_constraint_texts))

    # --- переходи атаки: досяжність + експлуатація -----------------------
    # Тут з'являється ланцюг "стан -> вразливість -> стан": вразливість
    # лежить НА шляху свідомо, а не одним ребром "актив досяжний з активу",
    # бо саме вона потім розриває `twin_defense.cut_sets` — control-вузол
    # лікує вразливість, а не досяжність саму по собі.
    asset_by_id = {a["id"]: a for a in network.get("assets", [])}
    states_by_asset: dict[str, set[str]] = {}

    def register_state(asset_id: str, state_id: str) -> None:
        if asset_id:
            states_by_asset.setdefault(asset_id, set()).add(state_id)

    register_state(initial_asset, current_state_id)
    register_state(goal_asset, goal_state_id)

    # Крок 1: кожна вразливість, чий exploitClass маємо чим зіставити (див.
    # `_exploit_class_condition`), дає ребро "yields" до умови, яку вона
    # приносить НА ТОМУ САМОМУ активі, де лежить. Окреме від уже наявного
    # "grants" нижче: те відношення лишається ЛИШЕ для явно названого
    # `viaCve` задачі (факт), "yields" — для цього модельного припущення
    # (гіпотеза) — дві заяви різної впевненості не повинні жити під одним
    # іменем відношення.
    for asset in network.get("assets", []):
        aid = asset["id"]
        for vid in asset_vuln_ids.get(aid, []):
            condition = _exploit_class_condition(vuln_by_id[vid]["exploit_class"], asset)
            if not condition:
                continue  # exploitClass поза відомими чотирма — без здогадок
            state_id = ensure_condition_state(aid, condition)
            link(vid, "yields", state_id, transition=True)
            register_state(aid, state_id)

    # Крок 2: з КОЖНОГО стану, здобутого на активі A, — до КОЖНОЇ
    # вразливості на активі B, досяжному з A за одним із трьох публічних
    # правил (`_asset_reachability`). Порядок Крок 1 -> Крок 2 навмисний:
    # джерела для Кроку 2 мають бути вже зареєстровані, включно з тими, що
    # щойно з'явилися в Кроці 1. `A != B` завжди (реачебіліті не містить
    # пар активу з самим собою) — саме тому переходи "reaches"/"exploits"
    # ніколи не можуть утворити симетричну пару з "yields"/"grants"
    # (`test_edges_are_named_and_directed_not_symmetric`).
    # Переходи спершу збираються, а потім укладаються в канонічному порядку.
    # Причина не косметична: `_asset_reachability` і реєстр станів будуються
    # через `set`, а порядок обходу множини рядків у CPython залежить від
    # хеш-сіда процесу. Набір ребер від цього не змінюється, зате
    # змінюється їхня послідовність, а `qa.twin.enumerate_scenarios` обриває
    # пошук у глибину на межі кандидатів — тож у вибірку потрапляли різні
    # шляхи, і середнє покриття на 100 мережах гуляло від 0,2900 до 0,3247
    # між сідами. Робота заявляє відтворюваність, отже порядок закріплено.
    transitions: list[tuple[str, str, str]] = []
    for (src_asset, dst_asset), rule in _asset_reachability(network).items():
        target_asset = asset_by_id.get(dst_asset)
        if target_asset is None:
            continue
        for state_id in states_by_asset.get(src_asset, ()):
            for vid in asset_vuln_ids.get(dst_asset, []):
                if not _exploit_class_condition(vuln_by_id[vid]["exploit_class"], target_asset):
                    continue  # вразливість без відомого наслідку — глухий кут
                transitions.append((state_id, vid, rule))
    for state_id, vid, rule in sorted(transitions):
        link(state_id, "exploits", vid, transition=True, reachability_rule=rule)

    initial_cve = task.get("initial_cve", "")
    if initial_cve:
        vid = f"VULN:{initial_asset}:{initial_cve}"
        if vid in asset_vuln_ids.get(initial_asset, []):
            link(vid, "grants", current_state_id, transition=True)

    return TwinModel(
        name=network.get("id", "") or "corporate-network",
        nodes=nodes,
        edges=edges,
        criteria=list(DEFAULT_CRITERIA),
        # `ALLOWED_CONSTRAINTS` (budget/months/personnel/regulatory) із
        # захисних засобів (kind/product/state) і далі нічим наповнити —
        # жодної вартості, термінів чи персоналу дані не називають. Єдине
        # джерело числового обмеження — `<task><constraints>`, зведене
        # `classify_task_constraints` вище (обґрунтування — докстрінг тієї
        # функції); решта текстових вимог лишається вузлами
        # kind="requirement", а не мовчки губиться чи вигаданим числом.
        constraints=task_constraints,
        current_state_id=current_state_id,
        goal_state_id=goal_state_id,
    )


# =====================================================================
#  Еталон: answers.json (окремий файл, не всередині жодного XML)
# =====================================================================

def load_answers(path: str) -> dict[str, dict]:
    """`answers.json` -> мережа -> еталонний запис (`best_attack`,
    `effectiveness`, `lengths`, `margin`, …).

    Це окремий файл, що описує лише ЗВЕДЕННЯ; сам він не є частиною
    `<attacks>` жодного конкретного XML і не бере участі в побудові моделі —
    ця функція взагалі не викликається з `build_twin_model`."""
    with open(path, "r", encoding="utf-8") as f:
        document = json.load(f)
    return {entry["network"]: entry for entry in document.get("networks", [])}


def cve_exploitability_index(network: dict) -> dict[str, float]:
    """CVE -> exploitability за ПУБЛІЧНИМИ даними активів (сервіси + ОС).

    Джерело — виключно `<service><vulnerability>` та `<osVulnerabilities>
    <vulnerability>` зі словника `parse_network`, тобто те, що бачить
    аналітик ДО будь-якого еталона; сюди НЕ потрапляє нічого з `<attacks>`."""
    index: dict[str, float] = {}
    for asset in network.get("assets", []):
        for service in asset.get("services", []):
            for vuln in service.get("vulnerabilities", []):
                if vuln["cve"]:
                    index[vuln["cve"]] = vuln["exploitability"]
        for vuln in asset.get("os_vulnerabilities", []):
            if vuln["cve"]:
                index[vuln["cve"]] = vuln["exploitability"]
    return index


def reproduce_effectiveness(gold_attacks: list[dict], cve_index: dict[str, float]) -> dict[str, float]:
    """Відтворення метрики з `answers.json["scoring"]` — "product of
    per-step CVSS exploitability": добуток exploitability по кроках атаки.

    `exploitability` кожного кроку береться НЕ з самого кроку (крок називає
    лише CVE), а з публічного реєстру вразливостей активів
    (`cve_exploitability_index`)."""
    result: dict[str, float] = {}
    for attack in gold_attacks:
        product = 1.0
        for step in attack["steps"]:
            product *= cve_index.get(step["cve"], 0.0)
        result[attack["id"]] = round(product, 6)
    return result
