"""
Коммит-скрипт: генерує канонічний словник {external_id: name} для технік
MITRE ATT&CK (enterprise-attack) і зберігає його у mitre_names.json у корені репо.

Логіка вибору external_id/name СВІДОМО повторює shipper/verifier.py
(`MitreAttackKB._add_technique`): source_name == "mitre-attack", формат
T\\d{4}(?:\\.\\d{3})?, виключаємо revoked / x_mitre_deprecated обʼєкти.

Запуск:
    python tools/gen_mitre_names.py
"""

import json
import os
import re
import sys
import urllib.request

MITRE_STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)
MITRE_FORMAT_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mitre_names.json"
)


def extract_technique_names(bundle: dict) -> dict[str, str]:
    """Плоский словник {external_id: name} з STIX-бандла enterprise-attack.

    Правила ідентичні shipper/verifier.py::MitreAttackKB._add_technique:
    - беремо лише obj["type"] == "attack-pattern";
    - пропускаємо revoked is True або x_mitre_deprecated is True;
    - шукаємо external_reference з source_name == "mitre-attack",
      external_id має відповідати T####（.###)?, беремо перший такий ref.
    """
    names: dict[str, str] = {}
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True:
            continue
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tid = ref.get("external_id")
                if tid and MITRE_FORMAT_RE.match(tid):
                    names[tid] = obj.get("name")
                break
    return names


def fetch_bundle(url: str = MITRE_STIX_URL, timeout: float = 120.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    print(f"Скачую MITRE ATT&CK bundle з {MITRE_STIX_URL} ...")
    bundle = fetch_bundle()
    names = extract_technique_names(bundle)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Записано {len(names)} технік у {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main() or 0)
