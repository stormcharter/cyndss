"""Експеримент «реконструкція ланцюга» на HTB-моделях: тести корпусу та метрики.

Головна гарантія, яку тут стережемо, — ПРИХОВАНИЙ `attack_path` НЕ потрапляє
в корпус, з якого будується семантична мережа. Інакше експеримент вимірює не
реконструкцію, а здатність прочитати відповідь.

Ідуть і під pytest, і напряму: `python tests/test_htb_experiment.py`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.htb_reconstruction.corpus import (
    build_records,
    entry_concepts,
    goal_concept,
    gold_chain,
    scoring_gold,
)
from experiments.htb_reconstruction.run_experiment import summarise
from experiments.htb_reconstruction.metric import (
    ordered_coverage,
    precision_against_gold,
    step_coverage,
)


# --- Синтетична машина: маленька, але з усіма полями, які нас цікавлять -----

MACHINE = {
    "machine": "Testbox",
    "components": [
        {"component_id": "H1", "name": "Testbox VM", "layer_or_type": "Host",
         "role": "OS boundary", "binding_or_path": "10.0.0.1",
         "software_or_version": "Ubuntu 16.04", "runs_as": "root",
         "exposure": "VPN reachable", "analysis_note": "Standalone"},
        {"component_id": "C1", "name": "Apache HTTP", "layer_or_type": "Service",
         "role": "Web front end", "binding_or_path": "0.0.0.0:80/tcp",
         "software_or_version": "Apache 2.4.18", "runs_as": "www-data",
         "exposure": "Public", "analysis_note": "Serves development directory"},
        {"component_id": "C2", "name": "OpenSSL heartbeat", "layer_or_type": "Library",
         "role": "TLS library", "binding_or_path": "HTTPS handshake",
         "software_or_version": "OpenSSL 1.0.1f", "runs_as": "Apache",
         "exposure": "Public TLS", "analysis_note": "Memory disclosure"},
    ],
    "identities": [
        {"identity_id": "I1", "name": "hype", "type": "local user",
         "scope_or_privilege": "shell", "analysis_note": "Owns tmux socket"},
    ],
    "permission_edges": [
        {"subject": "anonymous", "permission_or_action": "read private key",
         "object": "C1", "condition_or_scope": "public web path",
         "security_consequence": "Obtains hype credential ciphertext"},
    ],
    "artifacts": [
        {"artifact_id": "A1", "name": "hype private key", "location_or_path": "/dev",
         "content_or_purpose": "encrypted credential", "analysis_note": "Downloadable"},
    ],
    "vulnerabilities": [
        {"finding_id": "V1", "affected_component": "C2",
         "affected_version": "OpenSSL 1.0.1f", "issue": "Heartbeat memory disclosure",
         "preconditions": "HTTPS reachable",
         "vulnerable_behavior": "Oversized heartbeat leaks process memory",
         "resulting_access": "passphrase disclosure",
         "reference": "CVE-2014-0160", "failed_control": "Vulnerable TLS library"},
    ],
    # Прихована відповідь. `unobtainium` — маркер, якого немає ніде більше:
    # якщо він просочиться в корпус, тест це спіймає.
    "attack_path": [
        {"step": 1, "source_principal": "anonymous", "action": "download unobtainium key",
         "target_component_or_artifact": "A1", "prerequisite": "HTTP",
         "resulting_principal_or_access": "encrypted credential"},
        {"step": 2, "source_principal": "anonymous", "action": "unobtainium heartbleed",
         "target_component_or_artifact": "C2", "prerequisite": "vulnerable TLS",
         "resulting_principal_or_access": "passphrase"},
        {"step": 3, "source_principal": "attacker", "action": "unobtainium ssh",
         "target_component_or_artifact": "C1", "prerequisite": "key and passphrase",
         "resulting_principal_or_access": "root"},
    ],
}


# --- Корпус ----------------------------------------------------------------

def test_corpus_does_not_leak_attack_path():
    """Найважливіший тест файлу: жодне слово з прихованого ланцюга, якого немає
    в описових полях, не має опинитись у корпусі."""
    records = build_records(MACHINE)
    assert records, "порожній корпус зробив би цю перевірку беззмістовною"
    blob = " ".join(
        f"{r.get('title', '')} {r.get('summary', '')}" for r in records
    ).lower()
    assert "unobtainium" not in blob, (
        "attack_path протік у корпус — експеримент вимірював би не реконструкцію, "
        "а читання відповіді"
    )


def test_corpus_is_split_into_many_documents():
    """Одна машина = багато дрібних документів, а не один суцільний текст.
    Інакше всі поняття співзустрічаються з усіма й граф вироджується у клік."""
    records = build_records(MACHINE)
    assert len(records) >= 6, f"очікували документ на кожну сутність, отримали {len(records)}"
    assert all(r.get("mongo_id") for r in records), "кожен запис потребує стабільного id"


def test_corpus_introduces_no_words_of_its_own():
    """Корпус має складатись ЛИШЕ зі значень полів моделі.

    Якщо шаблон додає власні слова-скріпки («runs as», «failed control»,
    «resulting access»), вони стають повноцінними вузлами графа — ще й
    високого ступеня, бо трапляються в КОЖНОМУ документі. Обхід починає
    ходити через них, і експеримент вимірює вплив мого форматування,
    а не структуру інфраструктури."""
    from semantic_reasoner import extract_terms

    source_values = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "attack_path":
                    continue
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif node is not None:
            source_values.append(str(node))

    walk(MACHINE)
    allowed = set(extract_terms(" ".join(source_values)))

    produced = set()
    for record in build_records(MACHINE):
        produced.update(extract_terms(f"{record['title']} {record['summary']}"))

    invented = produced - allowed
    assert not invented, f"шаблон вигадав поняття, яких немає в моделі: {sorted(invented)}"


def test_corpus_ids_are_stable_across_runs():
    a = [r["mongo_id"] for r in build_records(MACHINE)]
    b = [r["mongo_id"] for r in build_records(MACHINE)]
    assert a == b


def test_verified_cve_is_carried_as_anchor():
    records = build_records(MACHINE)
    cves = {c for r in records for c in (r.get("verified_cve_ids") or "").split(",") if c}
    assert "CVE-2014-0160" in cves


# --- Еталонний ланцюг ------------------------------------------------------

def test_gold_chain_keeps_step_order_and_resolves_component_names():
    gold = gold_chain(MACHINE)
    assert len(gold) == 3
    # Крок 2 цілиться в C2 → має розкритись у назву «OpenSSL heartbeat».
    assert "openssl" in gold[1] or "heartbeat" in gold[1]
    # Останній крок дає root.
    assert "root" in gold[2]


def test_gold_chain_excludes_action_wording():
    """Дія — це ребро, а не вузол. Якщо втягнути слова дії в еталон, збіг
    почне зараховуватись за формулювання, а не за структуру."""
    gold = gold_chain(MACHINE)
    assert not any("unobtainium" in tokens for tokens in gold)


def test_goal_concept_is_final_privilege():
    assert goal_concept(MACHINE) == "root"


def test_scoring_gold_drops_the_goal_step():
    """Останній крок еталона — це і є ціль обходу, яку ми ПОДАЛИ на вхід.
    Зараховувати її як «відтворену» = накрутити метрику на дурничку."""
    full = gold_chain(MACHINE)
    scored = scoring_gold(MACHINE)
    assert len(scored) == len(full) - 1
    assert scored == full[:-1]


def test_entry_concepts_come_from_exposure_not_from_attack_path():
    """Стартові поняття для двофронтового пошуку мають виводитись із опису
    (що виставлено назовні), а не з прихованої відповіді."""
    entries = entry_concepts(MACHINE)
    assert entries, "у машини є публічні компоненти — старт не може бути порожнім"
    # Apache і OpenSSL позначені Public / Public TLS; hype — локальний користувач.
    assert "apache" in entries
    assert "hype" not in entries


def test_entry_concepts_ignore_internal_components():
    machine = {
        "machine": "Internal",
        "components": [
            {"component_id": "C1", "name": "Backoffice Service", "exposure": "Internal only"},
        ],
        "attack_path": [],
    }
    assert entry_concepts(machine) == set()


# --- Агрегація -------------------------------------------------------------

def test_summary_excludes_machines_without_scorable_gold():
    """Машина, у якої після відкидання цільового кроку не лишилось жодного
    еталонного кроку (напр. ланцюг з одного кроку), НЕ може бути оцінена.
    Якщо зарахувати їй 0, середнє змішає «не впорався» з «не було чого шукати»
    і занизить результат на рівному місці."""
    def row(scorable_goal, gold_steps, value):
        keys = ["a_coverage", "a_ordered", "a_precision", "a_coverage_topk",
                "b_coverage", "b_ordered", "b_precision",
                "c_coverage", "c_ordered", "c_precision", "c_coverage_topk",
                "d_coverage", "d_ordered", "d_precision", "d_coverage_topk",
                "e_coverage", "e_ordered", "e_precision", "e_coverage_topk",
                "e_precision_nodes", "e_nodes"]
        base = {"goal_in_graph": scorable_goal, "gold_steps": gold_steps,
                "b_found": bool(value)}
        base.update({k: value for k in keys})
        return base

    rows = [
        row(True, 2, 1.0),
        # Нічим оцінювати — має випасти з агрегатів, а не тягнути їх униз.
        row(True, 0, 0.0),
        # Цілі взагалі немає в графі — теж не оцінюється.
        row(False, 3, 0.0),
    ]
    summary = summarise(rows)
    assert summary["machines_total"] == 3
    assert summary["machines_scored"] == 1
    assert summary["unidirectional_from_goal"]["coverage"]["mean"] == 1.0
    # Кожна гілка агрегується за тим самим правилом — нулі непридатних
    # машин не мають просочитись у жодну з них.
    assert summary["bidirectional_single_path"]["coverage"]["mean"] == 1.0
    assert summary["bidirectional_production"]["coverage"]["mean"] == 1.0
    assert summary["unidirectional_from_given_starts"]["coverage"]["mean"] == 1.0


# --- Метрика ---------------------------------------------------------------

def test_step_coverage_counts_only_matched_steps():
    gold = [{"apache"}, {"openssl", "heartbeat"}, {"root"}]
    assert step_coverage(["apache", "root"], gold) == 2 / 3
    assert step_coverage(["apache", "openssl", "root"], gold) == 1.0
    assert step_coverage(["nothing"], gold) == 0.0


def test_step_coverage_of_empty_gold_is_zero_not_crash():
    assert step_coverage(["apache"], []) == 0.0


def test_ordered_coverage_penalises_reversed_chain():
    gold = [{"apache"}, {"openssl"}, {"root"}]
    forward = ["apache", "openssl", "root"]
    reversed_chain = ["root", "openssl", "apache"]
    assert ordered_coverage(forward, gold) == 1.0
    # Зворотний порядок покриває ті самі вузли, але послідовність зламана:
    # найдовша зростаюча підпослідовність — один елемент.
    assert ordered_coverage(reversed_chain, gold) == 1 / 3
    assert ordered_coverage(forward, gold) >= step_coverage(forward, gold) - 1e-9


def test_ordered_coverage_never_exceeds_step_coverage():
    gold = [{"a1"}, {"b2"}, {"c3"}]
    chain = ["c3", "a1"]
    assert ordered_coverage(chain, gold) <= step_coverage(chain, gold)


def test_precision_is_share_of_chain_nodes_present_in_gold():
    gold = [{"apache"}, {"openssl"}, {"root"}]
    assert precision_against_gold(["apache", "root"], gold) == 1.0
    assert precision_against_gold(["apache", "junk"], gold) == 0.5
    assert precision_against_gold([], gold) == 0.0


if __name__ == "__main__":
    # Консоль Windows за замовчуванням cp1252 — український вивід її валить.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover — старі інтерпретатори
        pass
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001 — самописний раннер
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failed")
    sys.exit(1 if failures else 0)


def _row(machine="m", gold_steps=2, **arms):
    """Рядок результату з УСІМА полями всіх гілок.

    Неназвана гілка отримує нулі, а не відсутній ключ: агрегатор має бачити
    повний набір, інакше падіння тесту говорить про неповну фікстуру, а не про
    поведінку коду."""
    row = {"machine": machine, "goal_in_graph": True, "gold_steps": gold_steps,
           "b_found": True, "e_goal_reachable": bool(arms.get("e")),
           "e_nodes": 0, "e_precision_nodes": arms.get("e", 0.0)}
    for arm in ("a", "b", "c", "d", "e"):
        value = arms.get(arm, 0.0)
        for suffix in ("coverage", "ordered", "precision", "coverage_topk"):
            row[f"{arm}_{suffix}"] = value
    return row


# --- Гілка D: контроль, що ізолює внесок обходу ---------------------------
# Навіщо. A і C відрізняються ДВОМА факторами: примітивом обходу і тим, хто
# обирає старти. Тому різниця між ними не каже, який фактор дав ефект. Гілка D
# отримує ТІ САМІ старти, що C, і той самий ранжувальник, а відрізняється лише
# одностороннім обходом. Порівняння C vs D ізолює обхід, D vs A — вибір стартів.

def test_summary_reports_the_control_arm():
    rows = [_row(machine="m1", a=0.1, b=0.2, c=0.5, d=0.4)]
    got = summarise(rows)
    assert "unidirectional_from_given_starts" in got, list(got)
    assert got["unidirectional_from_given_starts"]["coverage"]["mean"] == 0.4


def test_control_arm_is_scored_like_the_others():
    rows = [_row(machine="m1", a=0.0, b=0.0, c=1.0, d=1.0)]
    got = summarise(rows)
    for key in ("coverage", "ordered_coverage", "precision", "coverage_top_k"):
        assert key in got["unidirectional_from_given_starts"], key


def test_control_arm_chains_are_dumped_for_reuse():
    from experiments.htb_reconstruction.run_experiment import FIELDS
    for field in ("d_coverage", "d_ordered", "d_precision", "d_coverage_topk"):
        assert field in FIELDS, field


# --- Гілка E: ОРІЄНТОВАНИЙ ТИПІЗОВАНИЙ граф -------------------------------
# Гілки A-D усі ходять одним і тим самим неорієнтованим графом співзустрічальності
# й відрізняються лише обходом та стартами. E міняє САМУ МОДЕЛЬ: ребро має
# напрям і назву, узяті з полів опису, а не зі статистики. Це єдина гілка, у
# якій «орієнтована семантична мережа» зі статті існує насправді.

def test_summary_reports_the_directed_arm():
    rows = [_row(machine="m1", a=0.1, c=0.5, e=0.7)]
    got = summarise(rows)
    assert "directed_typed_graph" in got, list(got)
    assert got["directed_typed_graph"]["coverage"]["mean"] == 0.7


def test_directed_arm_is_scored_like_the_others():
    got = summarise([_row(machine="m1", a=0.0, c=1.0, e=1.0)])
    for key in ("coverage", "ordered_coverage", "precision", "coverage_top_k"):
        assert key in got["directed_typed_graph"], key


def test_directed_arm_fields_are_dumped():
    from experiments.htb_reconstruction.run_experiment import FIELDS
    for field in ("e_coverage", "e_ordered", "e_precision", "e_coverage_topk"):
        assert field in FIELDS, field


def test_directed_arm_runs_on_a_real_machine():
    """Наскрізний прогін формул (8)-(11) на справжній моделі — граф має бути
    зв'язним настільки, щоб ціль була досяжною хоча б десь."""
    import json
    from pathlib import Path
    from experiments.htb_reconstruction.run_experiment import evaluate
    path = Path(__file__).resolve().parents[1] / "experiments/htb_reconstruction/dataset/demo/grandpa.json"
    machine = json.loads(path.read_text(encoding="utf-8"))
    row = evaluate(machine, k=5, weight_mode="npmi", max_hops=6)
    assert "e_coverage" in row
    # Grandpa має досяжну ціль, тож наскрізний прогін мусить дати ланцюг —
    # інакше це перевірка «не впало», а не перевірка поведінки.
    assert row["e_chains"] > 0
    assert row["e_goal_reachable"] is True


def test_directed_arm_reports_precision_at_entity_granularity():
    """Токенна точність гілок непорівнянна: вузол неорієнтованого графа — це
    ОДИН токен, а вузол орієнтованого — сутність із кількох слів. Порівнювати
    їх «як є» означає карати орієнтовану гілку за спосіб запису, а не за суть.
    Тому для неї окремо рахується частка СУТНОСТЕЙ шляху, дотичних до еталона."""
    from experiments.htb_reconstruction.run_experiment import FIELDS
    assert "e_precision_nodes" in FIELDS
    assert "e_nodes" in FIELDS


def test_entity_precision_counts_nodes_not_tokens():
    from experiments.htb_reconstruction.run_experiment import entity_precision
    gold = [{"webdav"}, {"system"}]
    # два вузли, кожен із кількох токенів; дотичний до еталона лише перший
    node_concepts = [["microsoft", "webdav", "application"], ["grandpa", "boundary"]]
    assert entity_precision(node_concepts, gold) == 0.5


def test_entity_precision_is_zero_for_empty_path():
    from experiments.htb_reconstruction.run_experiment import entity_precision
    assert entity_precision([], [{"webdav"}]) == 0.0
