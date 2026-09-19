"""Перетворення HTB-моделі машини на корпус документів + еталонний ланцюг.

Ключове рішення: одна машина розкладається на БАГАТО дрібних документів —
по одному на компонент, ідентичність, артефакт, право доступу та вразливість.
Це навмисно: `SemanticReasoner` будує ребро між кожною парою понять У МЕЖАХ
одного документа. Якщо подати машину одним суцільним текстом, усі поняття
співзустрінуться з усіма, граф виродиться у повний підграф з однаковими вагами,
і будь-який «ланцюг» стане артефактом порядку сортування, а не структури.
Дрібні документи дають те саме, що дає реальний OSINT-потік: кожен пост
згадує кілька сутностей, а структура виникає з перетинів між постами.

Друге ключове рішення: `attack_path` НЕ бере участі в побудові корпусу.
Він — прихований еталон, з яким звіряють результат. Стереже це
`tests/test_htb_experiment.py::test_corpus_does_not_leak_attack_path`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from semantic_reasoner import extract_terms  # noqa: E402


def _clean(value) -> str:
    """Порожні поля в датасеті позначені `-`; прибираємо, щоб не смітити текст."""
    text = str(value or "").strip()
    return "" if text in {"-", "n/a", "N/A"} else text


def _join(*parts) -> str:
    return " ".join(p for p in (_clean(p) for p in parts) if p)


def _name_map(machine: dict) -> dict[str, str]:
    """`component_id` / `identity_id` / `artifact_id` → людська назва.

    Кроки `attack_path` цілять у сутність за ідентифікатором (`C3`, `A1`),
    а граф оперує словами з назв. Без цієї мапи еталон неможливо зіставити
    з передбаченням."""
    names: dict[str, str] = {}
    for item in machine.get("components", []) or []:
        names[_clean(item.get("component_id"))] = _clean(item.get("name"))
    for item in machine.get("identities", []) or []:
        names[_clean(item.get("identity_id"))] = _clean(item.get("name"))
    for item in machine.get("artifacts", []) or []:
        names[_clean(item.get("artifact_id"))] = _clean(item.get("name"))
    names.pop("", None)
    return names


def _resolve(ref, names: dict[str, str]) -> str:
    """`C3` → «OpenSSL heartbeat». Невідоме посилання лишаємо як є."""
    key = _clean(ref)
    return names.get(key, key)


def _cve_ids(reference) -> str:
    """У полі `reference` буває або CVE, або текст на кшталт
    «Configuration weakness». Якорем робимо лише справжні CVE."""
    out = []
    for part in _clean(reference).replace(";", ",").split(","):
        part = part.strip().upper()
        if part.startswith("CVE-"):
            out.append(part)
    return ",".join(out)


def build_records(machine: dict) -> list[dict]:
    """Модель машини → список записів у форматі корпусу Manticore.

    Формат (`title`, `summary`, `verified_cve_ids`, `mongo_id`) навмисно той
    самий, що й у бойового `events_enriched`: експеримент має ганяти ТОЙ ЖЕ
    код `SemanticReasoner`, що й прод, без окремої гілки «для статті»."""
    names = _name_map(machine)
    slug = _clean(machine.get("machine")).lower().replace(" ", "-") or "machine"
    records: list[dict] = []

    def add(kind: str, key: str, title: str, summary: str, cves: str = "") -> None:
        if not _clean(title):
            return
        records.append({
            "mongo_id": f"{slug}:{kind}:{key}",
            "title": _clean(title),
            "summary": _clean(summary),
            "verified_cve_ids": cves,
        })

    host = machine.get("host") or {}
    add("host", "0", _clean(machine.get("machine")),
        _join(host.get("operating_system_or_build"), host.get("deployment_role")))

    for i, c in enumerate(machine.get("components", []) or []):
        add("component", _clean(c.get("component_id")) or str(i), c.get("name"),
            _join(c.get("layer_or_type"), c.get("role"), c.get("binding_or_path"),
                  c.get("software_or_version"), c.get("runs_as"),
                  c.get("exposure"), c.get("analysis_note"),
                  _resolve(c.get("parent_id"), names)))

    for i, ident in enumerate(machine.get("identities", []) or []):
        add("identity", _clean(ident.get("identity_id")) or str(i), ident.get("name"),
            _join(ident.get("type"), ident.get("scope_or_privilege"),
                  ident.get("analysis_note"), _resolve(ident.get("parent_id"), names)))

    for i, a in enumerate(machine.get("artifacts", []) or []):
        add("artifact", _clean(a.get("artifact_id")) or str(i), a.get("name"),
            _join(a.get("location_or_path"), a.get("content_or_purpose"),
                  a.get("analysis_note"), _resolve(a.get("parent_id"), names)))

    for i, e in enumerate(machine.get("permission_edges", []) or []):
        target = _resolve(e.get("object"), names)
        add("permission", str(i),
            _join(e.get("subject"), target),
            _join(e.get("permission_or_action"), target,
                  e.get("condition_or_scope"), e.get("security_consequence")))

    for i, v in enumerate(machine.get("vulnerabilities", []) or []):
        affected = _resolve(v.get("affected_component"), names)
        add("vulnerability", _clean(v.get("finding_id")) or str(i),
            _join(v.get("issue"), affected),
            _join(affected, v.get("affected_version"), v.get("preconditions"),
                  v.get("vulnerable_behavior"), v.get("resulting_access"),
                  v.get("failed_control")),
            cves=_cve_ids(v.get("reference")))

    return records


def gold_chain(machine: dict) -> list[set[str]]:
    """Прихований еталон: по множині токенів на кожен крок `attack_path`.

    Вузол кроку — це ЦІЛЬ (компонент/артефакт) і ЗДОБУТИЙ доступ. Слова самої
    дії (`action`) свідомо НЕ беремо: дія — це ребро. Якби ми втягнули її
    формулювання, збіг зараховувався б за перефразування тексту, а не за
    відтворену структуру."""
    names = _name_map(machine)
    chain: list[set[str]] = []
    for step in machine.get("attack_path", []) or []:
        text = _join(
            _resolve(step.get("target_component_or_artifact"), names),
            step.get("resulting_principal_or_access"),
        )
        tokens = set(extract_terms(text))
        if tokens:
            chain.append(tokens)
    return chain


def scoring_gold(machine: dict) -> list[set[str]]:
    """Еталон для оцінювання — без ОСТАННЬОГО кроку.

    Останній крок і є ціль, яку ми самі подали на вхід обходу. Зараховувати її
    як «відтворену» означало б додавати до кожної метрики безкоштовну одиницю."""
    chain = gold_chain(machine)
    return chain[:-1] if chain else []


# Що вважаємо доступним ззовні. Виводиться з поля `exposure` опису, тобто з
# ТОГО, ЩО ЗНАЄ ЗАХИСНИК, — не з прихованого `attack_path`.
_EXPOSED = ("public", "vpn", "internet", "external", "anonymous", "unauthenticated")


def entry_concepts(machine: dict) -> set[str]:
    """Точки входу для двофронтового пошуку: поняття з назв компонентів,
    виставлених назовні.

    Це чесний старт для `bidirectional_path`: аналітик знає периметр, але не
    знає ланцюга. Якби ми брали старт із першого кроку еталона, порівняння
    одно- і двонапрямного обходу було б підіграванням другому."""
    concepts: set[str] = set()
    for c in machine.get("components", []) or []:
        exposure = _clean(c.get("exposure")).lower()
        if any(marker in exposure for marker in _EXPOSED):
            concepts.update(extract_terms(_clean(c.get("name"))))
    return concepts


def goal_concept(machine: dict) -> str | None:
    """Ціль обходу — здобутий доступ на ОСТАННЬОМУ кроці (типово `root`
    або `system`). Беремо останній токен, бо у фразах на кшталт
    «NT AUTHORITY SYSTEM» значущим є саме він."""
    steps = machine.get("attack_path", []) or []
    for step in reversed(steps):
        tokens = extract_terms(_clean(step.get("resulting_principal_or_access")))
        if tokens:
            return tokens[-1]
    return None
