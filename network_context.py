"""Фізична мережа (XML кіберполігону) як КОНТЕКСТ для побудови семантичної мережі.

Мережа тут не джерело структури, а опис світу, у якому розв'язується проблема:
які є вузли, у яких сегментах, які засоби захисту ввімкнено, вимкнено чи
працюють частково. Цей опис іде в промпти мовної моделі, щоб поняття
семантичної мережі були про ЦЮ мережу, а не загальні поради.

Розділ `<attacks>` і цілі атакуючого свідомо ігноруються: задача — не шукати
атаки, а будувати шлях від проблеми до розв'язку.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

#: Скільки проблемних засобів захисту перелічувати в описі поіменно. Решта
#: підсумовується числом — інакше контекст великої мережі з'їдає вікно моделі.
MAX_LISTED_GAPS = 12


@dataclass
class Network:
    id: str
    profile: str = ""
    topology: str = ""
    subnets: list[dict] = field(default_factory=list)
    assets: dict[str, dict] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)


def parse_network_xml(xml_text: str) -> Network:
    """XML кіберполігону -> `Network`. Кидає ValueError із зрозумілою причиною."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Це не коректний XML: {exc}") from exc
    if root.tag != "network":
        raise ValueError("Очікується кореневий елемент <network>.")
    net = Network(id=root.get("id", "net"), profile=root.get("profile", ""),
                  topology=root.get("topology", ""))
    net.subnets = [dict(s.attrib) for s in root.iterfind("subnets/subnet")]
    for asset in root.iterfind("assets/asset"):
        os_el = asset.find("os")
        net.assets[asset.get("id")] = {
            "id": asset.get("id"),
            "role": asset.get("role", ""),
            "subnet": asset.get("subnet", ""),
            "criticality": asset.get("criticality", ""),
            "os": " ".join(filter(None, [os_el.get("product"), os_el.get("version")]))
            if os_el is not None else "",
            "defenses": {d.get("kind"): d.get("state") for d in asset.iter("defense")},
            "vulns": sum(1 for _ in asset.iter("vulnerability")),
            "reused_accounts": sum(1 for _ in asset.iter("reusedOn")),
        }
    net.constraints = [c.text.strip() for c in root.iterfind("task/constraints/constraint")
                       if c.text and c.text.strip()]
    if not net.assets:
        raise ValueError("У мережі немає жодного вузла <asset>.")
    return net


def defense_gaps(net: Network) -> list[tuple[str, str, str]]:
    """(вузол, засіб, стан) для засобів, що не в стані `enabled`."""
    return [(a["id"], kind, state) for a in net.assets.values()
            for kind, state in a["defenses"].items() if state != "enabled"]


def unprotected(net: Network) -> list[str]:
    """Критичні й важливі вузли без жодного засобу захисту."""
    return [a["id"] for a in net.assets.values()
            if not a["defenses"] and a["criticality"] in {"critical", "high"}]


def network_brief(net: Network) -> str:
    """Опис мережі для промпту: склад, сегменти, прогалини в захисті, обмеження."""
    roles = Counter(a["role"] for a in net.assets.values())
    lines = [
        f"Мережа {net.id} (профіль {net.profile or '—'}, топологія {net.topology or '—'}): "
        f"{len(net.assets)} вузлів — " + ", ".join(f"{r} ×{n}" for r, n in roles.most_common()) + ".",
    ]
    if net.subnets:
        lines.append("Сегменти: " + ", ".join(
            f"{s.get('id')} ({s.get('zone')}, довіра {s.get('trust')})" for s in net.subnets) + ".")
    gaps = defense_gaps(net)
    if gaps:
        listed = "; ".join(f"{node}: {kind} — {state}" for node, kind, state in gaps[:MAX_LISTED_GAPS])
        more = f" і ще {len(gaps) - MAX_LISTED_GAPS}" if len(gaps) > MAX_LISTED_GAPS else ""
        lines.append(f"Засоби захисту не в робочому стані ({len(gaps)}): {listed}{more}.")
    bare = unprotected(net)
    if bare:
        lines.append("Важливі вузли без жодного засобу захисту: " + ", ".join(bare) + ".")
    vulns = sum(a["vulns"] for a in net.assets.values())
    if vulns:
        worst = sorted(net.assets.values(), key=lambda a: -a["vulns"])[:4]
        lines.append(f"Відомих вразливостей: {vulns}; найбільше на "
                     + ", ".join(f"{a['id']} ({a['vulns']})" for a in worst if a["vulns"]) + ".")
    reused = [a["id"] for a in net.assets.values() if a["reused_accounts"]]
    if reused:
        lines.append("Облікові записи з повторними паролями на: " + ", ".join(reused) + ".")
    if net.constraints:
        lines.append("Обмеження: " + "; ".join(net.constraints) + ".")
    return "\n".join(lines)


def default_problem(net: Network) -> str:
    gaps = defense_gaps(net)
    bare = unprotected(net)
    vulns = sum(a["vulns"] for a in net.assets.values())
    text = (f"Слабка захищеність мережі {net.id}: {vulns} відомих вразливостей, "
            f"{len(gaps)} засобів захисту вимкнено або працюють частково")
    if bare:
        text += f", {len(bare)} важливих вузлів без захисту ({', '.join(bare[:4])})"
    return text + "."


def default_solution(net: Network) -> str:
    return (f"Мережа {net.id} у задовільному стані кібербезпеки: засоби захисту працюють "
            "на всіх важливих вузлах, вразливості під контролем, вимоги обмежень виконано.")
