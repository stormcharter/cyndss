"""
Видобування сутностей із тексту через LLM (OpenAI-сумісний API: OpenRouter / Hermes 3).

Строгий JSON просимо прямо в промпті. На випадок, якщо модель усе ж загорне відповідь
у текст ("Sure, here are the CVEs: ..."), нижче є запасний варіант — видобування підрядка
між першою { та останньою }.
"""

import asyncio
import json

import httpx
import structlog

from prompt_guard import ISOLATION_INSTRUCTION_EN, wrap_untrusted
from .config import settings
from .models import ExtractedEntities

logger = structlog.get_logger(__name__)


# Промпт без .format() — проста конкатенація. Це позбавляє потреби
# екранувати фігурні дужки JSON-схеми всередині шаблону.
#
# ВАЖЛИВО: у прикладі відповіді НЕ МАЄ бути конкретних ідентифікаторів. Раніше
# тут стояли "CVE-2026-3055" і "T1190" як зразки — модель їх просто
# списувала, і на бойовому корпусі T1190 опинився на 375 мітках із 380. Місце
# ідентифікатора займає опис типу, а не значення (регресійний тест —
# tests/test_extractor_contract.py).
EXTRACTION_PROMPT_PREFIX = """You are an expert Cyber Threat Intelligence (CTI) analyst. Your task is to analyze the provided raw security feed text from an OSINT source and extract structured threat data.

You MUST respond with a valid JSON object strictly adhering to the following schema. Do not include any conversational text, markdown formatting, or explanations.

JSON Schema:
{
  "title": "A short, concise title of the threat event in English (max 10 words).",
  "summary": "A comprehensive summary and analytical description of the threat event in English.",
  "extracted_cve_ids": [
    {"id": "<CVE identifier>", "quote": "<exact fragment of the text that mentions it>"}
  ],
  "mitre_tactics_techniques": [
    {"id": "<ATT&CK technique identifier>", "quote": "<exact fragment of the text that supports it>"}
  ],
  "detected_language": "<ISO 639-1 code of the source text>"
}

Strict Rules:
1. 'title' and 'summary' MUST be strictly in English.
2. EVIDENCE RULE. Attach a CVE or an ATT&CK technique ONLY if this specific text
   supports it. Every entry MUST carry a "quote" copied VERBATIM from the text
   below — do not translate it, do not paraphrase it, do not invent it. If you
   cannot quote the supporting fragment, omit the entry.
3. Most security posts mention no CVE and no technique at all. That is the normal
   case: return empty arrays []. An empty array is a CORRECT answer; a guess is not.
4. Never output an identifier merely because it is common, plausible or typical
   for this kind of story. No evidence in the text — no entry.
5. Ensure JSON is valid.

""" + ISOLATION_INSTRUCTION_EN + """

Text to analyse (untrusted, delimited):
"""


# Скільки символів вихідного тексту віддаємо моделі (як і раніше).
MAX_PROMPT_TEXT_LEN = 4000


def build_extraction_prompt(text):
    """Збирає промпт екстрактора: довірена частина + НЕДОВІРЕНИЙ текст,
    загорнутий у делімітери (захист від непрямої ін'єкції промпту)."""
    return EXTRACTION_PROMPT_PREFIX + wrap_untrusted(text, max_len=MAX_PROMPT_TEXT_LEN)


class OllamaExtractor:
    """Клієнт до LLM для видобування сутностей."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(timeout=None)
        # LLM через OpenAI-сумісний API (OpenRouter, Hermes 3 у хмарі).
        self.url = f"{settings.llm_base_url}/chat/completions"
        self.model = settings.llm_model
        self.headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            self.headers["Authorization"] = f"Bearer {settings.llm_api_key}"
            self.headers["X-Title"] = "ONYX-CTI"

    async def _post_chat(self, prompt: str) -> httpx.Response:
        """POST у chat/completions з ретраями на 429/503 (rate-limit OpenRouter).
        Поважає Retry-After; експоненційний бекоф. Кількість спроб обмежена, щоб при
        вичерпаному денному ліміті не зависнути назавжди — далі віддаємо останню
        відповідь, і .raise_for_status() деградує до порожнього обогащення."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
        }
        delay = 2.0
        response = None
        for attempt in range(4):
            response = await self.client.post(self.url, headers=self.headers, json=payload)
            if response.status_code not in (429, 503):
                return response
            retry_after = response.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else delay
            logger.warning(
                "llm_rate_limited", status=response.status_code, wait=wait, attempt=attempt + 1
            )
            await asyncio.sleep(wait)
            delay = min(delay * 2, 30.0)
        return response

    async def extract(self, text: str) -> ExtractedEntities:
        try:
            # Формування промпту загорнуто в try — на випадок неочікуваних
            # помилок під час роботи з рядком (наприклад, якщо text виявиться не str).
            prompt = build_extraction_prompt(text)

            response = await self._post_chat(prompt)
            response.raise_for_status()
            data = response.json()
            # Формат OpenAI: choices[0].message.content
            choices = data.get("choices") or [{}]
            raw_content = (choices[0].get("message") or {}).get("content", "")

            if not raw_content:
                logger.warning("extractor_empty_response")
                return ExtractedEntities(summary=text[:200])

            # При format=json Ollama повертає чистий JSON, але про всяк випадок
            # лишаємо видобування між першою { та останньою } — захист від
            # моделей, які інколи додають пояснення навколо.
            start, end = raw_content.find("{"), raw_content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(raw_content[start : end + 1])
                    return self._to_pydantic(parsed)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "extractor_json_decode_failed",
                        error=str(e),
                        raw_preview=raw_content[:200],
                    )

            # Повертаємо порожній результат, якщо розбір не вдався
            return ExtractedEntities(summary=text[:200])

        except Exception as e:
            import traceback
            tb_lines = traceback.format_exc()
            logger.error("extractor_critical_fail", error_msg=str(e), traceback=tb_lines)
            return ExtractedEntities(summary=str(text)[:200])

    # Ключі, під якими модель може віддати ідентифікатор і цитату. Схему ми
    # диктуємо ("id"/"quote"), але моделі дрейфують і подеколи повертають стару
    # плоску форму або називають поля по-своєму — розбираємо терпимо.
    _ID_KEYS = ("id", "cve_id", "technique_id", "value")
    _QUOTE_KEYS = ("quote", "claimed_context", "context", "evidence")

    @classmethod
    def _ids_with_contexts(cls, value) -> tuple[list[str], dict[str, str]]:
        """Розбирає масив міток у (список ID, {ID: цитата}).

        Приймає і об'єктну форму {"id": ..., "quote": ...}, і застарілу форму
        з простих рядків (тоді цитати просто немає — мітка зможе пройти
        заземлення лише дослівною згадкою ID у тексті)."""
        ids: list[str] = []
        contexts: dict[str, str] = {}
        if not isinstance(value, list):
            return ids, contexts

        for item in value:
            ident = quote = None
            if isinstance(item, str):
                ident = item
            elif isinstance(item, dict):
                ident = next((item[k] for k in cls._ID_KEYS
                              if isinstance(item.get(k), str)), None)
                quote = next((item[k] for k in cls._QUOTE_KEYS
                              if isinstance(item.get(k), str)), None)
            if not ident or not ident.strip():
                continue
            ident = ident.strip().upper()
            if ident in ids:
                continue
            ids.append(ident)
            if quote and quote.strip():
                contexts[ident] = quote.strip()
        return ids, contexts

    @classmethod
    def _to_pydantic(cls, parsed: dict) -> ExtractedEntities:
        """Перетворює dict від LLM на Pydantic-модель."""
        cve_ids, cve_ctx = cls._ids_with_contexts(parsed.get("extracted_cve_ids"))
        ttp_ids, ttp_ctx = cls._ids_with_contexts(parsed.get("mitre_tactics_techniques"))

        return ExtractedEntities(
            title=parsed.get("title", "Unparsed Threat Event"),
            summary=parsed.get("summary") if isinstance(parsed.get("summary"), str) else None,
            extracted_cve_ids=cve_ids,
            mitre_tactics_techniques=ttp_ids,
            cve_contexts=cve_ctx,
            ttp_contexts=ttp_ctx,
            detected_language=parsed.get("detected_language", "unknown"),
        )

    async def aclose(self):
        await self.client.aclose()
