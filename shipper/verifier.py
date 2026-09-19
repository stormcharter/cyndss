"""
Нейро-символічний шар: валідація видобутих LLM сутностей проти
авторитетних джерел.

Адаптовано під плоску структуру списків рядків ExtractedEntities.
"""

import asyncio
import ipaddress
import re
from datetime import datetime
from typing import Optional

import httpx
import structlog

from .config import settings
from .models import (
    ExtractedEntities,
    VerificationStatus,
    VerifiedCVE,
    VerifiedRelation,
    VerifiedTTP,
)

logger = structlog.get_logger(__name__)

CVE_FORMAT_RE = re.compile(r"^CVE-\d{4}-\d{4,}$")
MITRE_FORMAT_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


# ---------- MITRE ATT&CK ----------

class MitreAttackKB:
    """Локальний кеш STIX-бандла MITRE ATT&CK."""

    def __init__(self):
        self.techniques: dict[str, dict] = {}  # T1190 -> {name, tactic, desc, ...}
        self.relationships: set[tuple[str, str]] = set()

    async def load(self, client: httpx.AsyncClient) -> None:
        logger.info("loading_mitre_kb", url=settings.mitre_stix_url)
        response = await client.get(settings.mitre_stix_url, timeout=120.0)
        response.raise_for_status()
        bundle = response.json()

        for obj in bundle.get("objects", []):
            obj_type = obj.get("type")
            if obj_type == "attack-pattern":
                self._add_technique(obj)
            elif obj_type == "relationship":
                src = obj.get("source_ref", "")
                tgt = obj.get("target_ref", "")
                self.relationships.add((src, tgt))

        logger.info("mitre_kb_loaded", techniques=len(self.techniques))

    def _add_technique(self, obj: dict) -> None:
        # Символьний шар не має довіряти відкликаним (revoked) чи deprecated
        # технікам MITRE ATT&CK — інакше вони "верифікують" галлюцинації LLM.
        # Порівнюємо саме `is True` (не truthy), бо в STIX ці поля або відсутні,
        # або строго bool.
        if obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True:
            return
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tid = ref.get("external_id")
                if tid and MITRE_FORMAT_RE.match(tid):
                    tactic = None
                    phases = obj.get("kill_chain_phases", [])
                    if phases:
                        tactic = phases[0].get("phase_name")
                    self.techniques[tid] = {
                        "name": obj.get("name"),
                        "tactic": tactic,
                        "description": obj.get("description", "")[:1000],
                    }
                break

    def lookup(self, technique_id: str) -> Optional[dict]:
        return self.techniques.get(technique_id)


# ---------- NVD ----------

class NVDClient:
    """Клієнт до NVD API з кешуванням і семафором на кількість одночасних запитів."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.headers = {}
        if settings.nvd_api_key:
            self.headers["apiKey"] = settings.nvd_api_key
        self._sem = asyncio.Semaphore(5 if not settings.nvd_api_key else 20)
        self._cache: dict[str, VerifiedCVE] = {}

    async def lookup(self, cve_id: str) -> VerifiedCVE:
        if not CVE_FORMAT_RE.match(cve_id):
            return VerifiedCVE(cve_id=cve_id, status=VerificationStatus.INVALID_FORMAT)

        if cve_id in self._cache:
            return self._cache[cve_id]

        async with self._sem:
            try:
                response = await self.client.get(
                    settings.nvd_base_url,
                    params={"cveId": cve_id},
                    headers=self.headers,
                    timeout=30.0,
                )
                if response.status_code == 404:
                    result = VerifiedCVE(cve_id=cve_id, status=VerificationStatus.NOT_FOUND)
                    self._cache[cve_id] = result
                    return result
                response.raise_for_status()
                data = response.json()
                result = self._parse_nvd_response(cve_id, data)
                self._cache[cve_id] = result
                return result
            except httpx.HTTPError as e:
                logger.warning("nvd_unreachable", cve_id=cve_id, error=str(e))
                return VerifiedCVE(cve_id=cve_id, status=VerificationStatus.UNREACHABLE)

    @staticmethod
    def _parse_nvd_response(cve_id: str, data: dict) -> VerifiedCVE:
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return VerifiedCVE(cve_id=cve_id, status=VerificationStatus.NOT_FOUND)

        cve = vulns[0].get("cve", {})
        description = None
        for d in cve.get("descriptions", []):
            if d.get("lang") == "en":
                description = d.get("value")
                break

        cvss = None
        metrics = cve.get("metrics", {})
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics and metrics[key]:
                cvss = metrics[key][0].get("cvssData", {}).get("baseScore")
                break

        cwes = []
        for weakness in cve.get("weaknesses", []):
            for desc in weakness.get("description", []):
                if desc.get("lang") == "en":
                    val = desc.get("value", "")
                    if val.startswith("CWE-"):
                        cwes.append(val)

        published = None
        if cve.get("published"):
            try:
                published = datetime.fromisoformat(cve["published"].replace("Z", "+00:00"))
            except ValueError:
                pass

        return VerifiedCVE(
            cve_id=cve_id,
            status=VerificationStatus.VERIFIED,
            cvss_score=cvss,
            description=description,
            published_date=published,
            cwe_ids=cwes,
        )


# ---------- Головний валідатор ----------

class EntityVerifier:
    """Координатор нейро-символічної верифікації."""

    def __init__(self, mitre_kb: MitreAttackKB, nvd_client: NVDClient):
        self.mitre = mitre_kb
        self.nvd = nvd_client

    async def verify(self, extracted: ExtractedEntities) -> tuple[
        list[VerifiedCVE], list[VerifiedTTP], list, list
    ]:
        # 1. CVE — обробляємо плоский список рядків
        cve_tasks = [self.nvd.lookup(cve_id.upper()) for cve_id in extracted.extracted_cve_ids]
        verified_cves = await asyncio.gather(*cve_tasks) if cve_tasks else []

        # 2. TTP — обробляємо плоский список рядків
        verified_ttps = [self._verify_ttp(tech_id) for tech_id in extracted.mitre_tactics_techniques]

        # 3. IoC і зв'язки поки що повертають порожні списки (під нову схему UI)
        verified_iocs = []
        verified_relations = []

        return verified_cves, verified_ttps, verified_iocs, verified_relations

    def _verify_ttp(self, technique_id: str) -> VerifiedTTP:
        tid = technique_id.upper().strip()
        if not MITRE_FORMAT_RE.match(tid):
            return VerifiedTTP(technique_id=tid, status=VerificationStatus.INVALID_FORMAT)
        record = self.mitre.lookup(tid)
        if not record:
            return VerifiedTTP(technique_id=tid, status=VerificationStatus.NOT_FOUND)
        return VerifiedTTP(
            technique_id=tid,
            status=VerificationStatus.VERIFIED,
            name=record["name"],
            tactic=record["tactic"],
            description=record["description"],
        )