"""Детерміновані алгоритми методу над графом сценаріїв сесії.

Реалізує метод із статті як обмежені чисті операції: розбір моделі двійника,
актуалізація свідчень із RAG, двонапрямне розкриття підмереж, їх узгодження,
перебір допустимих шляхів і детерміноване агрегування.
Виклики LLM та HTTP-пошук лишаються в ``qa.agent``; цей модуль імпортується
з обох без циклів.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field

import yaml

MAX_NODES = 200
MAX_EDGES = 1000
MAX_CRITERIA = 20
MAX_DEPTH = 20
MAX_SCENARIOS = 20
MAX_PROPERTY_STRING_CHARS = 3_800
MAX_REPORT_PARTS = 3

ALLOWED_KINDS = {
    "asset", "service", "state", "vuln", "control",
    "resource", "requirement", "threat", "gate",
}
ALLOWED_CONSTRAINTS = {"budget", "months", "personnel", "regulatory"}
# Вузол-стан може оголосити, який саме це стан. Знати ідентичність — не те
# саме, що мати сесію, контекст виконання чи привілей, а шлях, який
# прослизає від одного до іншого без названої дії, — не сценарій.
# Нетипізовані стани зберігають попередню, поблажливу поведінку.
ALLOWED_STATE_TYPES = {
    "known_identity",
    "authenticated_session",
    "execution_context",
    "privilege",
    "security_state",
}
# Увійти в один із цих станів можна лише ребром, що називає дію переходу;
# сама по собі ідентичність доступу не дає.
ACCESS_STATE_TYPES = {"authenticated_session", "execution_context", "privilege"}
# Вузол-гейт виражає кон'юнкцію або диз'юнкцію попередників — простий шлях
# сам по собі цього виразити не може.
ALLOWED_GATES = {"and", "or"}
# Полярність критерію. Критерії-витрати інвертуються перед інтегральною оцінкою,
# щоб J лишалася порівнянною між моделями з різними наборами критеріїв.
ALLOWED_DIRECTIONS = {"benefit", "cost"}
INLINE_NODE_ATTRIBUTES = {
    "cost", "months", "personnel", "external_ids", "evidence_terms", "regulatory",
    "state_type", "gate",
}
TREATMENT_RELATIONS = {"mitigates", "detects", "recovers_from"}
# Атрибути обмежень на вузлах/ребрах; для months у фікстурах даних немає, тож сума
# завжди нуль. Ліміт regulatory >= 1 вимагає схвалених контролів на шляху.
CONSTRAINT_ATTRIBUTE = {
    "budget": "cost",
    "months": "months",
    "personnel": "personnel",
}
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
_TTP_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
_TERM_RE = re.compile(r"[a-z][a-z0-9_-]{3,}")
_EVIDENCE_LABEL_STOPWORDS = {
    "access", "account", "application", "asset", "control", "current",
    "external", "full", "goal", "host", "initial", "local", "service",
    "state", "user",
}


class TwinModelError(ValueError):
    """Документ моделі двійника порушує обмежену схему."""


@dataclass(frozen=True)
class TwinNode:
    id: str
    kind: str
    label: str
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TwinEdge:
    source: str
    target: str
    relation: str
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TwinCriterion:
    name: str
    description: str
    weight: float
    direction: str = "benefit"


@dataclass(frozen=True)
class TwinConstraint:
    criterion: str
    limit: float


@dataclass(frozen=True)
class TwinModel:
    name: str
    nodes: list[TwinNode]
    edges: list[TwinEdge]
    criteria: list[TwinCriterion]
    constraints: list[TwinConstraint]
    current_state_id: str
    goal_state_id: str

    @property
    def node_by_id(self) -> dict[str, TwinNode]:
        return {node.id: node for node in self.nodes}

    def outgoing(self, node_id: str) -> list[TwinEdge]:
        return [edge for edge in self.edges if edge.source == node_id]

    def incoming(self, node_id: str) -> list[TwinEdge]:
        return [edge for edge in self.edges if edge.target == node_id]

    def state_type(self, node_id: str) -> str:
        node = self.node_by_id.get(node_id)
        if node is None or node.kind != "state":
            return ""
        return str(node.attributes.get("state_type") or "")


def _require(condition: object, message: str) -> None:
    if not condition:
        raise TwinModelError(message)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_path_list(value: object) -> list:
    return list(value) if isinstance(value, list) else []


def _scalars(raw: object, context: str) -> dict[str, object]:
    """Приймає скаляри та обмежені списки скалярів (напр. evidence_terms)."""
    if raw is None:
        return {}
    _require(
        isinstance(raw, dict),
        f"{context} attributes must be a mapping; use attributes: {{}} or omit it",
    )
    assert isinstance(raw, dict)
    clean: dict[str, object] = {}
    for key, value in raw.items():
        _require(
            isinstance(key, str) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key)),
            f"{context} attribute name {key!r} is invalid",
        )
        if isinstance(value, list):
            _require(len(value) <= 20, f"{context} attribute {key!r} exceeds 20 items")
            for item in value:
                _require(
                    item is None or isinstance(item, (bool, int, float, str)),
                    f"{context} attribute {key!r} items must be JSON scalars",
                )
            clean[key] = value
            continue
        _require(
            value is None or isinstance(value, (bool, int, float, str)),
            f"{context} attribute {key!r} must be a JSON scalar or bounded list",
        )
        clean[key] = value
    return clean


def parse_twin_model(yaml_text: str) -> TwinModel:
    """Розібрати й перевірити один обмежений документ моделі двійника."""
    try:
        document = yaml.safe_load(yaml_text)
    except yaml.YAMLError as error:
        raise TwinModelError(f"twin model is not valid YAML: {error}") from error
    _require(isinstance(document, dict), "twin model must be a mapping")
    assert isinstance(document, dict)
    raw_nodes = document.get("nodes")
    _require(isinstance(raw_nodes, list) and bool(raw_nodes), "twin model requires nodes")
    assert isinstance(raw_nodes, list)
    _require(len(raw_nodes) <= MAX_NODES, f"twin model exceeds {MAX_NODES} nodes")
    nodes: list[TwinNode] = []
    seen: set[str] = set()
    for raw in raw_nodes:
        _require(isinstance(raw, dict), "each node must be a mapping")
        assert isinstance(raw, dict)
        node_id = str(raw.get("id", "")).strip()
        kind = str(raw.get("kind", "")).strip().lower()
        label = str(raw.get("label", "")).strip()
        _require(bool(node_id) and len(node_id) <= 128, "node id is required (max 128 chars)")
        _require(node_id not in seen, f"duplicate node id {node_id!r}")
        _require(kind in ALLOWED_KINDS, f"node {node_id!r} has unsupported kind {kind!r}")
        _require(bool(label) and len(label) <= 256, f"node {node_id!r} needs a label (max 256)")
        seen.add(node_id)
        context = f"node {node_id}"
        attributes = _scalars(raw.get("attributes"), context)
        unknown_keys = (
            raw.keys()
            - {"id", "kind", "label", "attributes"}
            - INLINE_NODE_ATTRIBUTES
        )
        _require(
            not unknown_keys,
            f"{context} has unknown fields {sorted(unknown_keys)}",
        )
        inline_attributes = _scalars(
            {
                key: raw[key]
                for key in INLINE_NODE_ATTRIBUTES
                if key in raw
            },
            context,
        )
        duplicate_attributes = attributes.keys() & inline_attributes.keys()
        _require(
            not duplicate_attributes,
            f"{context} repeats attributes {sorted(duplicate_attributes)}",
        )
        merged_attributes = {**inline_attributes, **attributes}
        state_type = str(merged_attributes.get("state_type", "")).strip().lower()
        if state_type:
            _require(kind == "state", f"{context}: state_type applies to state nodes only")
            _require(
                state_type in ALLOWED_STATE_TYPES,
                f"{context} has unsupported state_type {state_type!r}",
            )
            merged_attributes["state_type"] = state_type
        gate = str(merged_attributes.get("gate", "")).strip().lower()
        if kind == "gate":
            _require(
                gate in ALLOWED_GATES,
                f"{context} needs gate: {sorted(ALLOWED_GATES)}",
            )
            merged_attributes["gate"] = gate
        else:
            _require(not gate, f"{context}: gate applies to gate nodes only")
        nodes.append(TwinNode(node_id, kind, label, merged_attributes))

    state_type_by_id = {
        node.id: str(node.attributes.get("state_type") or "")
        for node in nodes
        if node.kind == "state"
    }

    raw_edges = document.get("edges")
    _require(isinstance(raw_edges, list), "twin model requires an edges list")
    assert isinstance(raw_edges, list)
    _require(len(raw_edges) <= MAX_EDGES, f"twin model exceeds {MAX_EDGES} edges")
    edges: list[TwinEdge] = []
    for raw in raw_edges:
        _require(isinstance(raw, dict), "each edge must be a mapping")
        assert isinstance(raw, dict)
        source = str(raw.get("source", "")).strip()
        target = str(raw.get("target", "")).strip()
        relation = str(raw.get("relation", "")).strip()
        _require(source in seen, f"edge source {source!r} is not a declared node")
        _require(target in seen, f"edge target {target!r} is not a declared node")
        relation_valid = bool(relation) and bool(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", relation)
        )
        _require(relation_valid, f"edge relation {relation!r} is invalid")
        context = f"edge {source}->{target}"
        attributes = _scalars(raw.get("attributes"), context)
        inline_attributes = _scalars(
            {
                key: value
                for key, value in raw.items()
                if key not in {"source", "target", "relation", "attributes"}
            },
            context,
        )
        duplicate_attributes = attributes.keys() & inline_attributes.keys()
        _require(
            not duplicate_attributes,
            f"{context} repeats attributes {sorted(duplicate_attributes)}",
        )
        attributes = {**inline_attributes, **attributes}
        target_state_type = state_type_by_id.get(target, "")
        if target_state_type in ACCESS_STATE_TYPES:
            _require(
                bool(str(attributes.get("action", "")).strip()),
                f"{context} enters {target_state_type} state {target!r} "
                "without naming an action",
            )
        edges.append(TwinEdge(source, target, relation, attributes))

    current_state_id = str(document.get("current_state", "")).strip()
    goal_state_id = str(document.get("goal_state", "")).strip()
    kinds = {node.id: node.kind for node in nodes}
    _require(current_state_id in kinds, "current_state must reference a declared node")
    _require(goal_state_id in kinds, "goal_state must reference a declared node")
    _require(kinds[current_state_id] == "state", "current_state must be a state node")
    _require(kinds[goal_state_id] == "state", "goal_state must be a state node")
    _require(current_state_id != goal_state_id, "current_state and goal_state must differ")

    raw_criteria = document.get("criteria")
    _require(isinstance(raw_criteria, list) and bool(raw_criteria), "twin model requires criteria")
    assert isinstance(raw_criteria, list)
    _require(len(raw_criteria) <= MAX_CRITERIA, f"twin model exceeds {MAX_CRITERIA} criteria")
    criteria: list[TwinCriterion] = []
    names: set[str] = set()
    for raw in raw_criteria:
        _require(isinstance(raw, dict), "each criterion must be a mapping")
        assert isinstance(raw, dict)
        name = str(raw.get("name", "")).strip()
        _require(bool(name) and len(name) <= 128, "criterion name is required (max 128)")
        _require(name not in names, f"duplicate criterion {name!r}")
        names.add(name)
        weight = raw.get("weight")
        weight_value = _as_float(weight, default=-1.0)
        _require(
            not isinstance(weight, bool) and weight_value > 0,
            f"criterion {name!r} needs a positive numeric weight",
        )
        description = str(raw.get("description", "")).strip()
        _require(bool(description), f"criterion {name!r} needs a description")
        direction = str(raw.get("direction", "benefit")).strip().lower() or "benefit"
        _require(
            direction in ALLOWED_DIRECTIONS,
            f"criterion {name!r} has unsupported direction {direction!r}",
        )
        criteria.append(
            TwinCriterion(name, description[:2000], weight_value, direction)
        )

    raw_constraints = document.get("constraints") or []
    _require(isinstance(raw_constraints, list), "constraints must be a list")
    constraints: list[TwinConstraint] = []
    for raw in raw_constraints:
        _require(isinstance(raw, dict), "each constraint must be a mapping")
        assert isinstance(raw, dict)
        criterion = str(raw.get("criterion", "")).strip()
        _require(criterion in ALLOWED_CONSTRAINTS, f"unsupported constraint {criterion!r}")
        limit = raw.get("limit")
        limit_value = _as_float(limit, default=-1.0)
        _require(
            not isinstance(limit, bool) and limit_value >= 0,
            f"constraint {criterion!r} needs a non-negative numeric limit",
        )
        constraints.append(TwinConstraint(criterion, limit_value))

    return TwinModel(
        name=str(document.get("name", "twin"))[:128],
        nodes=nodes,
        edges=edges,
        criteria=criteria,
        constraints=constraints,
        current_state_id=current_state_id,
        goal_state_id=goal_state_id,
    )


def load_into_lattice(model: TwinModel, lattice) -> dict[str, object]:
    """Зберегти двійника як вузли ScenarioEntity та типізовані орієнтовані ребра.

    Властивості в LatticeDB лише скалярні, тому атрибути-списки серіалізуються
    в JSON-рядки перед записом.
    """
    node_index: dict[str, int] = {}
    for node in model.nodes:
        serialized = {
            key: (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, list)
                else value
            )
            for key, value in node.attributes.items()
        }
        properties = json.dumps(
            {"label": node.label, **serialized}, ensure_ascii=False
        )
        result = json.loads(lattice.store_entity(node.kind, node.id, properties))
        node_index[node.id] = _as_int(result.get("node_id"))
    for edge in model.edges:
        serialized = {
            key: (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, list)
                else value
            )
            for key, value in edge.attributes.items()
        }
        lattice.connect_entities(
            node_index[edge.source],
            node_index[edge.target],
            edge.relation,
            json.dumps(serialized, ensure_ascii=False),
        )
    return {
        "nodes": len(model.nodes),
        "edges": len(model.edges),
        "current_state_id": model.current_state_id,
        "goal_state_id": model.goal_state_id,
        "criteria": [criterion.name for criterion in model.criteria],
        "constraints": {
            constraint.criterion: constraint.limit for constraint in model.constraints
        },
    }


MAX_EDGE_EVIDENCE = 200
EVIDENCE_FRAGMENT_CHARS = 240


def _document_text(document: dict[str, object]) -> str:
    return " ".join(
        str(document.get(key) or "")
        for key in ("title", "summary", "content", "passage")
    ).lower()


def _node_evidence_terms(node: TwinNode) -> set[str]:
    """Рядки в нижньому регістрі, наявність яких у документі вказує на цей вузол."""
    terms: set[str] = set()
    for key in ("evidence_terms", "external_ids"):
        raw = node.attributes.get(key)
        values = raw if isinstance(raw, list) else ([raw] if raw else [])
        terms.update(str(value).lower() for value in values if str(value).strip())
    normalized_label = " ".join(re.findall(r"[a-z0-9]+", node.label.lower()))
    if len(normalized_label) >= 3:
        terms.add(normalized_label)
    terms.update(
        token
        for token in normalized_label.split()
        if len(token) >= 4 and token not in _EVIDENCE_LABEL_STOPWORDS
    )
    terms.update(
        token.lower() for token in re.findall(r"[A-Z][A-Z0-9]{2,}", node.label)
    )
    return {term for term in terms if term}


def _first_hit(text: str, terms: set[str]) -> int:
    positions = [text.find(term) for term in terms]
    hits = [position for position in positions if position >= 0]
    return min(hits) if hits else -1


def _fragment(text: str, left: int, right: int) -> str:
    start = max(0, min(left, right) - EVIDENCE_FRAGMENT_CHARS // 2)
    end = min(len(text), max(left, right) + EVIDENCE_FRAGMENT_CHARS // 2)
    return " ".join(text[start:end].split())[:EVIDENCE_FRAGMENT_CHARS]


def verify_edges(
    model: TwinModel,
    documents: list[dict[str, object]],
    matched_node_ids: list[str] | set[str],
) -> list[dict[str, object]]:
    """Прив'язати ребра до документа, який підтверджує перехід.

    Позначка «вузол згадано» нічого не каже про те, чи підтверджує джерело
    крок між двома вузлами. Ребро вважається підтвердженим лише коли один
    документ згадує обидва кінці; слово-відношення в тому ж документі
    підвищує впевненість, але не є обов'язковим.
    """
    matched = set(matched_node_ids)
    prepared = [
        (
            _document_text(document),
            str(document.get("mongo_id") or "").strip().lower(),
            str(document.get("observed_at") or document.get("published_at") or ""),
        )
        for document in documents
    ]
    terms_by_node = {node.id: _node_evidence_terms(node) for node in model.nodes}
    records: list[dict[str, object]] = []
    for index, edge in enumerate(model.edges):
        if edge.source not in matched or edge.target not in matched:
            continue
        source_terms = terms_by_node.get(edge.source, set())
        target_terms = terms_by_node.get(edge.target, set())
        if not source_terms or not target_terms:
            continue
        relation_word = edge.relation.replace("_", " ").lower()
        for text, mongo_id, observed_at in prepared:
            if not text:
                continue
            left = _first_hit(text, source_terms)
            right = _first_hit(text, target_terms)
            if left < 0 or right < 0:
                continue
            relation_hit = bool(relation_word) and relation_word in text
            records.append(
                {
                    "edge_index": index,
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "mongo_id": mongo_id,
                    "observed_at": observed_at[:64],
                    "fragment": _fragment(text, left, right),
                    "confidence": 0.9 if relation_hit else 0.6,
                }
            )
            break
        if len(records) >= MAX_EDGE_EVIDENCE:
            break
    return records


def actualize(
    model: TwinModel,
    lattice,
    documents: list[dict[str, object]],
    query: str,
) -> dict[str, object]:
    """Позначити вузли двійника, підтверджені знайденими документами й термінами запиту."""
    corpus = " ".join(
        " ".join(
            str(document.get(key) or "")
            for key in ("title", "summary", "content", "passage")
        )
        for document in documents
    ).lower()
    extracted_ids = {value.upper() for value in _CVE_RE.findall(corpus)}
    extracted_ids.update(value.upper() for value in _TTP_RE.findall(corpus))
    extracted_ids.update(value.upper() for value in _CVE_RE.findall(query))
    query_terms = set(_TERM_RE.findall(query.lower()))
    normalized_corpus = " ".join(re.findall(r"[a-z0-9]+", corpus))

    matched: list[str] = []
    for node in model.nodes:
        external_ids = node.attributes.get("external_ids")
        values = external_ids if isinstance(external_ids, list) else (
            [external_ids] if external_ids else []
        )
        external_hit = any(str(value).upper() in extracted_ids for value in values)
        evidence_terms = node.attributes.get("evidence_terms")
        terms = evidence_terms if isinstance(evidence_terms, list) else (
            [evidence_terms] if evidence_terms else []
        )
        term_hit = any(
            str(term).lower() in corpus or str(term).lower() in query.lower()
            for term in terms
        )
        normalized_label = " ".join(re.findall(r"[a-z0-9]+", node.label.lower()))
        label_tokens = set(normalized_label.split())
        distinctive_tokens = {
            token
            for token in label_tokens
            if len(token) >= 4 and token not in _EVIDENCE_LABEL_STOPWORDS
        }
        acronym_tokens = {
            token.lower()
            for token in re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", node.label)
        }
        corpus_tokens = set(normalized_corpus.split())
        label_hit = (
            len(normalized_label) >= 3
            and f" {normalized_label} " in f" {normalized_corpus} "
        ) or bool((distinctive_tokens | acronym_tokens) & corpus_tokens)
        name_hit = node.id.lower() in query_terms
        if external_hit or term_hit or label_hit or name_hit:
            matched.append(node.id)
    if matched:
        with lattice.database.write() as transaction:
            for node_id in matched:
                found = transaction.find_nodes_by_label_property(
                    "ScenarioEntity", "name", node_id, limit=5
                )
                for candidate in found:
                    transaction.set_property(candidate, "evidence_matched", True)
            transaction.commit()
    evidence_ids: list[str] = []
    for document in documents:
        mongo_id = str(document.get("mongo_id") or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{24}", mongo_id) and mongo_id not in evidence_ids:
            evidence_ids.append(mongo_id)
    edge_evidence = verify_edges(model, documents, matched)
    supported_edges = sorted({int(record["edge_index"]) for record in edge_evidence})
    return {
        "matched_node_ids": matched,
        "supported_edge_indexes": supported_edges,
        "edge_evidence": edge_evidence,
        "evidence_ids": evidence_ids[:40],
        "documents_considered": len(documents),
    }


def _bfs(model: TwinModel, start_id: str, max_depth: int, reverse: bool) -> dict[str, list]:
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for index, edge in enumerate(model.edges):
        source, target = (edge.target, edge.source) if reverse else (edge.source, edge.target)
        adjacency.setdefault(source, []).append((target, index))
    visited = [start_id]
    seen = {start_id}
    edge_ids: list[int] = []
    frontier: list[tuple[str, int]] = [(start_id, 0)]
    while frontier:
        node_id, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for target, edge_index in adjacency.get(node_id, []):
            edge_ids.append(edge_index)
            if target not in seen:
                seen.add(target)
                visited.append(target)
                frontier.append((target, depth + 1))
    return {"node_ids": visited, "edge_ids": edge_ids}


def expand_forward(model: TwinModel, start_id: str = "", max_depth: int = MAX_DEPTH) -> dict[str, list]:
    """Пряме розкриття від поточного стану вздовж орієнтованих ребер."""
    start = start_id or model.current_state_id
    _require(start in model.node_by_id, f"unknown forward start node {start!r}")
    _require(model.node_by_id[start].kind == "state", "forward expansion must start at a state")
    _require(1 <= int(max_depth) <= MAX_DEPTH, f"max_depth must be 1..{MAX_DEPTH}")
    return _bfs(model, start, int(max_depth), reverse=False)


def expand_backward(model: TwinModel, goal_id: str = "", max_depth: int = MAX_DEPTH) -> dict[str, list]:
    """Зворотне розкриття від цільового стану по обернених ребрах."""
    goal = goal_id or model.goal_state_id
    _require(goal in model.node_by_id, f"unknown backward start node {goal!r}")
    _require(model.node_by_id[goal].kind == "state", "backward expansion must start at a state")
    _require(1 <= int(max_depth) <= MAX_DEPTH, f"max_depth must be 1..{MAX_DEPTH}")
    return _bfs(model, goal, int(max_depth), reverse=True)


def reconcile(model: TwinModel, forward: dict, backward: dict) -> dict[str, list]:
    """Перетнути обидві підмережі в обмежений контекстний граф G_C."""
    shared = set(forward.get("node_ids") or []) & set(backward.get("node_ids") or [])
    node_set = shared | {model.current_state_id, model.goal_state_id}
    forward_edges = set(forward.get("edge_ids") or [])
    backward_edges = set(backward.get("edge_ids") or [])
    edge_ids = [
        index
        for index, edge in enumerate(model.edges)
        if index in forward_edges
        and index in backward_edges
        and edge.source in node_set
        and edge.target in node_set
    ]
    return {"node_ids": sorted(node_set), "edge_ids": edge_ids}


def _treatment_count(model: TwinModel, path: list[str]) -> int:
    treated: set[str] = set()
    for node_id in path:
        if model.node_by_id[node_id].kind != "control":
            continue
        for edge in model.outgoing(node_id):
            target = model.node_by_id.get(edge.target)
            if edge.relation in TREATMENT_RELATIONS and target and target.kind == "vuln":
                treated.add(edge.target)
    return len(treated)


def _within_constraints(
    model: TwinModel,
    path: list[str],
    path_edge_indexes: list[int],
) -> bool:
    totals = {name: 0.0 for name in CONSTRAINT_ATTRIBUTE}
    regulatory_required = any(
        constraint.criterion == "regulatory" and constraint.limit >= 1
        for constraint in model.constraints
    )
    elements: list[object] = [
        model.node_by_id[node_id]
        for node_id in path
        if model.node_by_id[node_id].kind != "state"
    ]
    elements.extend(model.edges[index] for index in path_edge_indexes)
    for element in elements:
        attributes = getattr(element, "attributes", {})
        for criterion, attribute in CONSTRAINT_ATTRIBUTE.items():
            value = attributes.get(attribute)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[criterion] += float(value)
    for constraint in model.constraints:
        if constraint.criterion in totals and totals[constraint.criterion] > constraint.limit:
            return False
    if regulatory_required:
        for element in elements:
            if getattr(element, "kind", None) == "control" and not getattr(
                element, "attributes", {}
            ).get("regulatory_approved"):
                return False
    return True


def within_constraints(
    model: TwinModel,
    path: list[str],
    path_edge_indexes: list[int],
) -> bool:
    """Публічний синонім `_within_constraints` для викликів ЗВНЕ цього модуля.

    НАВІЩО ОКРЕМА ФУНКЦІЯ. `_within_constraints` лишається приватною й далі
    для внутрішнього виклику з `enumerate_scenarios` — ім'я з підкресленням
    сигналізує "деталь реалізації перебору шляхів атаки". Але той самий
    рахунок бюджету/строків/персоналу/regulatory потрібен і за межами цього
    модуля (`twin_defense.rank_defenses` планує НАБОРИ ЗАСОБІВ ЗАХИСТУ, а не
    шляхи атаки) — а для публічного репозиторію імпорт приватного імені з
    іншого модуля виглядає як випадковість, а не як навмисний контракт.
    Синонім робить цей контракт явним, не дублюючи арифметику."""
    return _within_constraints(model, path, path_edge_indexes)


def _and_gate_requirements(model: TwinModel, edge_ids: list[int]) -> dict[str, set[str]]:
    """Попередники, які потрібні кожному AND-гейту, перш ніж шлях зможе через нього пройти."""
    requirements: dict[str, set[str]] = {}
    for node in model.nodes:
        if node.kind != "gate" or str(node.attributes.get("gate")) != "and":
            continue
        requirements[node.id] = {
            model.edges[index].source
            for index in edge_ids
            if model.edges[index].target == node.id
        }
    return requirements


def enumerate_scenarios(
    model: TwinModel,
    gc: dict,
    matched: set[str],
    max_paths: int = MAX_SCENARIOS,
    supported_edges: set[int] | None = None,
) -> dict[str, object]:
    """Видобути прості допустимі шляхи S0→ST із G_C.

    Простий шлях не може виразити кон'юнкцію, тому AND-гейт застосовується
    тут: шлях може пройти через гейт лише тоді, коли кожен його попередник
    уже є на шляху. Набір обмежень R застосовується після цього, і обидва
    лічильники відкинутих шляхів звітуються — щоб прогін фіксував, чи були
    фільтри активні, а не лишав це на здогади.
    """
    _require(1 <= int(max_paths) <= MAX_SCENARIOS, f"max_paths must be 1..{MAX_SCENARIOS}")
    edge_ids = list(gc.get("edge_ids") or [])
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for index in edge_ids:
        edge = model.edges[index]
        adjacency.setdefault(edge.source, []).append((edge.target, index))
    gate_requirements = _and_gate_requirements(model, edge_ids)

    paths: list[tuple[list[str], list[int]]] = []
    rejected_by_gate = 0

    def dfs(node_id: str, path: list[str], edge_path: list[int]) -> None:
        nonlocal rejected_by_gate
        if len(paths) >= max_paths * 10:
            return
        if node_id == model.goal_state_id:
            paths.append((list(path), list(edge_path)))
            return
        for target, edge_index in adjacency.get(node_id, []):
            if target in path:
                continue
            required = gate_requirements.get(target)
            if required and not required.issubset(set(path)):
                rejected_by_gate += 1
                continue
            path.append(target)
            edge_path.append(edge_index)
            dfs(target, path, edge_path)
            path.pop()
            edge_path.pop()

    dfs(model.current_state_id, [model.current_state_id], [])

    survivors: list[tuple[int, tuple[str, ...], list[str], list[int]]] = []
    rejected_by_constraint = 0
    for path, edge_path in paths:
        if _within_constraints(model, path, edge_path):
            survivors.append((-_treatment_count(model, path), tuple(path), path, edge_path))
        else:
            rejected_by_constraint += 1
    survivors.sort(key=lambda item: (item[0], item[1]))
    total_found = len(survivors)
    kept = survivors[: int(max_paths)]
    scenarios = [
        {
            "path": path,
            # Індекси пройдених ребер: без них зовнішнє ранжування не бачить
            # НАЗВ відношень і не може відрізнити крок, що називає механізм,
            # від простого твердження про досяжність.
            "edge_path": list(edge_path),
            "labels": [model.node_by_id[node_id].label for node_id in path],
            "structural": structural_score(
                model, path, edge_path, matched, supported_edges
            ),
        }
        for _order, _key, path, edge_path in kept
    ]
    return {
        "scenarios": scenarios,
        "total_found": total_found,
        "truncated": total_found > len(kept),
        "rejected_by_gate": rejected_by_gate,
        "rejected_by_constraint": rejected_by_constraint,
        "constraints_applied": [
            {"criterion": constraint.criterion, "limit": constraint.limit}
            for constraint in model.constraints
        ],
        "and_gates": sorted(gate_requirements),
    }


def structural_score(
    model: TwinModel,
    path: list[str],
    edge_path: list[int],
    matched: set[str],
    supported_edges: set[int] | None = None,
) -> dict[str, object]:
    """Структурні характеристики одного шляху-сценарію."""
    weights = [
        _as_float(model.edges[index].attributes.get("weight"))
        for index in edge_path
    ]
    total_weight = sum(weights)
    hops = max(len(path) - 1, 0)
    coverage = (
        sum(1 for node_id in path if node_id in matched) / len(path) if path else 0.0
    )
    supported = set(supported_edges or ())
    edge_coverage = (
        sum(1 for index in edge_path if index in supported) / len(edge_path)
        if edge_path
        else 0.0
    )
    return {
        "mean_weight": round(total_weight / hops, 4) if hops else 0.0,
        "total_weight": round(total_weight, 4),
        "hops": hops,
        "node_kinds": [model.node_by_id[node_id].kind for node_id in path],
        "state_types": [
            model.state_type(node_id) for node_id in path if model.state_type(node_id)
        ],
        "evidence_coverage": round(coverage, 4),
        "edge_evidence_coverage": round(edge_coverage, 4),
    }


def oriented_score(value: float, direction: str) -> float:
    """Оцінка на спільній шкалі користі, щоб критерії лишалися порівнянними."""
    return round(1.0 - value, 6) if direction == "cost" else round(value, 6)


def aggregate(
    scores: list[dict[str, object]],
    criteria: list[TwinCriterion],
) -> tuple[float, float]:
    """Зважена інтегральна оцінка J та розбіжність популяції для одного сценарію.

    Критерії, оголошені як витрати, спершу інвертуються. Без цього модель, що
    формулює критерій як «залишковий ризик», і модель, що формулює його як
    «безпека», дають інтегральні оцінки, які неможливо порівняти.
    """
    weights = {criterion.name: criterion.weight for criterion in criteria}
    directions = {criterion.name: criterion.direction for criterion in criteria}
    total_weight = sum(weights.values()) or 1.0
    integral = 0.0
    values: list[float] = []
    for score in scores:
        name = str(score.get("criterion"))
        value = oriented_score(
            _as_float(score.get("score")), directions.get(name, "benefit")
        )
        values.append(value)
        integral += value * weights.get(name, 0.0)
    disagreement = statistics.pstdev(values) if len(values) > 1 else 0.0
    return round(integral / total_weight, 4), round(disagreement, 4)


def merge_panel_votes(votes: list[dict[str, object]]) -> dict[str, object]:
    """Звести голоси кількох оцінювачів за одним критерієм в одну оцінку.

    Медіана стійка до одного оцінювача-викиду, а обґрунтування береться від
    того голосу, на якому зійшлася панель, а не як суміш текстів, яких ніхто
    з оцінювачів не писав. Стан свідчень навмисно песимістичний: щоб назвати
    його достатнім, потрібна одностайність. Розкид фіксується, бо панель, що
    згодна, і панель, що розділилася, не мають читатися однаково.
    """
    _require(bool(votes), "a panel vote merge needs at least one vote")
    values = [_as_float(vote.get("score")) for vote in votes]
    median = statistics.median(values)
    representative = min(votes, key=lambda vote: abs(_as_float(vote.get("score")) - median))
    states = [str(vote.get("evidence_state") or "") for vote in votes]
    merged = dict(representative)
    merged["score"] = round(median, 4)
    merged["confidence"] = round(
        sum(_as_float(vote.get("confidence")) for vote in votes) / len(votes), 4
    )
    if all(state == "sufficient" for state in states):
        merged["evidence_state"] = "sufficient"
    elif "contradictory" in states:
        merged["evidence_state"] = "contradictory"
    else:
        merged["evidence_state"] = "insufficient"
    merged["panel"] = [
        {
            "evaluator": str(vote.get("evaluator") or ""),
            "score": _as_float(vote.get("score")),
            "confidence": _as_float(vote.get("confidence")),
            "evidence_state": str(vote.get("evidence_state") or ""),
        }
        for vote in votes
    ]
    merged["panel_disagreement"] = (
        round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0
    )
    return merged


def refine_targets(scores: list[dict[str, object]], disagreement: float) -> list[str]:
    """Назви критеріїв, оцінки яких потребують рівно одного раунду уточнення."""
    targets = [
        str(score.get("criterion"))
        for score in scores
        if _as_float(score.get("confidence")) < 0.4
        or score.get("evidence_state") == "contradictory"
    ]
    if disagreement > 0.20:
        targets.extend(str(score.get("criterion")) for score in scores)
    return sorted(set(targets))


def keep_refinement(initial: dict[str, object], candidate: dict[str, object]) -> bool:
    """Прийняти уточнення лише коли воно впевненіше і не слабше."""
    return _as_float(candidate.get("confidence")) > _as_float(initial.get("confidence")) and (
        _as_float(candidate.get("score")) >= _as_float(initial.get("score"))
        or initial.get("evidence_state") != "sufficient"
    )


def rank(evaluations: list[dict[str, object]]) -> list[dict[str, object]]:
    """Детерміноване ранжування: інтегральна оцінка, потім коротший шлях, потім порядок."""
    return sorted(
        evaluations,
        key=lambda item: (
            -_as_float(item.get("J")),
            len(_as_path_list(item.get("path"))),
            tuple(_as_path_list(item.get("path"))),
        ),
    )


MAX_REPAIRS = 40


def repair_model(yaml_text: str) -> tuple[str, list[str]]:
    """Прибрати структурні дефекти, через які документ двійника не розбирається.

    Перезавантаження всієї моделі викидає граф, який здебільшого був справним,
    і платить за другий синтез. Натомість тут — точкові видалення, і лише
    видалення: дубльований id, ребро до вузла, якого не оголошували,
    петля, повторне ребро. Нічого тут не вигадує вузол, ребро чи атрибут,
    тож документ, який так не полагодити, все одно падає.
    Повертає полагоджений документ і по рядку на кожне застосоване виправлення.
    """
    try:
        document = yaml.safe_load(yaml_text)
    except yaml.YAMLError as error:
        raise TwinModelError(f"twin model is not valid YAML: {error}") from error
    if not isinstance(document, dict):
        raise TwinModelError("twin model must be a mapping")

    repairs: list[str] = []
    raw_nodes = document.get("nodes")
    if not isinstance(raw_nodes, list):
        raise TwinModelError("twin model requires nodes")

    nodes: list[dict] = []
    declared: set[str] = set()
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            repairs.append("dropped a node entry that is not a mapping")
            continue
        node_id = str(raw.get("id", "")).strip()
        if not node_id:
            repairs.append("dropped a node without an id")
            continue
        if node_id in declared:
            repairs.append(f"dropped duplicate node {node_id!r}")
            continue
        declared.add(node_id)
        nodes.append(raw)
    if len(nodes) > MAX_NODES:
        repairs.append(f"dropped {len(nodes) - MAX_NODES} nodes over the {MAX_NODES} limit")
        nodes = nodes[:MAX_NODES]
        declared = {str(node.get("id", "")).strip() for node in nodes}

    raw_edges = document.get("edges")
    edges: list[dict] = []
    if isinstance(raw_edges, list):
        seen_edges: set[tuple[str, str, str]] = set()
        for raw in raw_edges:
            if not isinstance(raw, dict):
                repairs.append("dropped an edge entry that is not a mapping")
                continue
            source = str(raw.get("source", "")).strip()
            target = str(raw.get("target", "")).strip()
            relation = str(raw.get("relation", "")).strip()
            if source not in declared or target not in declared:
                repairs.append(f"dropped edge {source!r}->{target!r} with an undeclared endpoint")
                continue
            if source == target:
                repairs.append(f"dropped self-loop on {source!r}")
                continue
            key = (source, target, relation)
            if key in seen_edges:
                repairs.append(f"dropped repeated edge {source!r}->{target!r} ({relation!r})")
                continue
            seen_edges.add(key)
            edges.append(raw)
        if len(edges) > MAX_EDGES:
            repairs.append(f"dropped {len(edges) - MAX_EDGES} edges over the {MAX_EDGES} limit")
            edges = edges[:MAX_EDGES]

    document["nodes"] = nodes
    document["edges"] = edges
    return yaml.safe_dump(document, allow_unicode=True, sort_keys=False), repairs[:MAX_REPAIRS]


def parse_or_repair(yaml_text: str) -> tuple[TwinModel, list[str]]:
    """Розібрати документ двійника, за потреби один раз полагодивши структурні дефекти."""
    try:
        return parse_twin_model(yaml_text), []
    except TwinModelError:
        repaired, repairs = repair_model(yaml_text)
        if not repairs:
            raise
        return parse_twin_model(repaired), repairs


def persist_report(lattice, payload: dict[str, object]) -> dict[str, object]:
    """Зберегти обмежену сутність ScenarioReport; рядкові властивості — до 3800 символів."""
    ranked_json = json.dumps(payload.get("ranked", []), ensure_ascii=False, default=str)
    truncated = len(ranked_json) > MAX_PROPERTY_STRING_CHARS * MAX_REPORT_PARTS
    ranked_json = ranked_json[: MAX_PROPERTY_STRING_CHARS * MAX_REPORT_PARTS]
    properties: dict[str, object] = {
        "status": str(payload.get("status", "ok"))[:64],
        "paths": _as_int(payload.get("paths")),
        "answer_summary": str(payload.get("answer", ""))[:2_000],
    }
    for index in range(MAX_REPORT_PARTS):
        chunk = ranked_json[
            index * MAX_PROPERTY_STRING_CHARS : (index + 1) * MAX_PROPERTY_STRING_CHARS
        ]
        if chunk:
            properties[f"ranked_part_{index}"] = chunk
    lattice.store_entity(
        "ScenarioReport",
        "ranked-scenarios",
        json.dumps({"truncated": truncated, **properties}, ensure_ascii=False),
    )
    return {"persisted": True, "truncated": truncated}


def build_snapshot(
    model: TwinModel,
    matched: set[str],
    forward: dict | None,
    backward: dict | None,
    gc: dict | None,
) -> dict[str, object]:
    """Обмежена серіалізація двійника та накладених шарів його генерації."""
    snapshot: dict[str, object] = {
        "current_state_id": model.current_state_id,
        "goal_state_id": model.goal_state_id,
        "nodes": [
            {
                "id": node.id,
                "label": node.label,
                "kind": node.kind,
                "evidence": node.id in matched,
            }
            for node in model.nodes[:MAX_NODES]
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
            }
            for edge in model.edges[:MAX_EDGES]
        ],
    }
    if forward is not None:
        snapshot["forward"] = {
            "nodes": list(forward.get("node_ids") or [])[:MAX_NODES],
            "edges": list(forward.get("edge_ids") or [])[:MAX_EDGES],
        }
    if backward is not None:
        snapshot["backward"] = {
            "nodes": list(backward.get("node_ids") or [])[:MAX_NODES],
            "edges": list(backward.get("edge_ids") or [])[:MAX_EDGES],
        }
    if gc is not None:
        snapshot["gc"] = {
            "nodes": list(gc.get("node_ids") or [])[:MAX_NODES],
            "edges": list(gc.get("edge_ids") or [])[:MAX_EDGES],
        }
    return snapshot
