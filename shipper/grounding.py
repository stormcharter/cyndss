"""
Заземлення (grounding) міток CVE/MITRE у тексті документа.

НАВІЩО. Екстрактор дає LLM лише сирий текст і схему відповіді — модель сама
вирішує, які ідентифікатори навісити. Верифікація (`shipper/verifier.py`)
підтверджує ЛИШЕ те, що такий ідентифікатор існує в NVD/MITRE, але не те, що
документ справді про нього. На бойовому корпусі це виродилось: 375 із 380
міток — T1190, і ЖОДНА техніка не трапляється в тексті поста дослівно.

ПРАВИЛО. Мітка приймається, лише якщо вона заземлена в документі:
  * LITERAL — ідентифікатор є в тексті дослівно;
  * QUOTED  — модель навела фрагмент (`claimed_context`), і цей фрагмент
              реально міститься в документі;
  * UNSUPPORTED — ані того, ані того: відкидаємо ще ДО звернення до NVD/MITRE
              (заодно економимо зовнішні виклики).

Цитата — слабший доказ, ніж дослівний ідентифікатор: модель може процитувати
речення й усе одно навісити не ту техніку. Але вимога цитувати змушує модель
показати, НА ЩО вона спирається, і різко піднімає ціну вигадки. Режим можна
затиснути до `literal` (див. `settings.grounding_mode`).

Модуль свідомо на чистій stdlib і без pydantic — його імпортують і шиппер,
і офлайнові інструменти чистки в `tools/`, що ганяються host-пітоном сервера.
"""

from __future__ import annotations

import re
import unicodedata
from enum import Enum


class Grounding(str, Enum):
    LITERAL = "literal"
    QUOTED = "quoted"
    UNSUPPORTED = "unsupported"


#: Коротший фрагмент доказом не вважаємо — «шлюзу» є майже в будь-якому тексті.
MIN_CONTEXT_CHARS = 12
#: Довший — це вже не цитата, а дамп документа; доказової сили не має.
MAX_CONTEXT_CHARS = 400

#: Що вважати достатнім доказом. Назви явні (без «default»), бо дефолт у проді —
#: саме ACCEPT_LITERAL: виміряно, що модель охоче підкріплює вигадану техніку
#: справжньою, але не по темі цитатою (див. tests/test_grounding.py, блок
#: «ВІДОМА СЛАБКІСТЬ»). ACCEPT_QUOTED лишається як свідомо мʼякший режим.
ACCEPT_LITERAL = (Grounding.LITERAL,)
ACCEPT_QUOTED = (Grounding.LITERAL, Grounding.QUOTED)

# Усі юнікодні варіанти дефіса, які трапляються в постах (редактори Telegram
# люблять підміняти ASCII-дефіс на non-breaking hyphen чи en-dash).
_DASHES = "‐‑‒–—―−"
_DASH_RE = re.compile(f"[{_DASHES}]")
_WS_RE = re.compile(r"\s+")

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)
_TTP_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)


def normalize(text: str) -> str:
    """Приводить текст до форми, в якій має сенс порівняння: NFKC, єдиний
    дефіс, згорнуті пробіли, casefold. Ідентифікатор і документ нормалізуються
    ОДНАКОВО, тож «CVE‑2026–3055» і «cve-2026-3055» стають одним рядком."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _DASH_RE.sub("-", text)
    text = _WS_RE.sub(" ", text)
    # «CVE - 2026 - 3055» → «cve-2026-3055»: пробіли навколо дефіса зайві.
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip().casefold()


def _literal_pattern(identifier: str) -> re.Pattern | None:
    """Регулярка для пошуку ідентифікатора в нормалізованому тексті.

    Межі виставлені так, щоб «T11901» і «T1190bis» не рахувались згадкою T1190
    (це інші токени), але щоб «T1078.004» рахувалось згадкою батьківської T1078
    (вона там справді є). Зворотне не діє: сама лише T1078 не підтверджує
    підтехніку .004. Крапка цифрою й літерою не є, тому межу не рве."""
    ident = normalize(identifier)
    if not ident:
        return None
    if not (_CVE_RE.match(ident) or _TTP_RE.match(ident)):
        return None
    return re.compile(rf"(?<![a-z0-9]){re.escape(ident)}(?![0-9a-z])")


def id_is_literal(identifier: str, text: str) -> bool:
    """Чи згадано ідентифікатор у тексті дослівно."""
    pattern = _literal_pattern(identifier)
    if pattern is None:
        return False
    return bool(pattern.search(normalize(text)))


def context_is_quoted(context: str | None, text: str) -> bool:
    """Чи є наведений моделлю фрагмент справжньою цитатою з документа."""
    if not context:
        return False
    stripped = str(context).strip()
    if not (MIN_CONTEXT_CHARS <= len(stripped) <= MAX_CONTEXT_CHARS):
        return False
    needle = normalize(stripped)
    if not needle:
        return False
    return needle in normalize(text)


def grounding_of(identifier: str, context: str | None, text: str) -> Grounding:
    """Найсильніше заземлення, яке має ця мітка в цьому документі."""
    if id_is_literal(identifier, text):
        return Grounding.LITERAL
    if context_is_quoted(context, text):
        return Grounding.QUOTED
    return Grounding.UNSUPPORTED


def filter_grounded(
    identifiers,
    contexts: dict | None,
    text: str,
    accept=ACCEPT_QUOTED,
) -> tuple[list[str], list[tuple[str, Grounding]]]:
    """Ділить мітки на прийняті й відкинуті.

    `contexts` — {ідентифікатор: цитата} від моделі (регістр не важливий).
    Повертає (kept, dropped), де dropped несе ще й ДОСЯГНУТЕ заземлення —
    видно, мітку відкинуто як цілком безпідставну чи лише через режим."""
    ctx = {str(k).strip().upper(): v for k, v in (contexts or {}).items()}
    accept = tuple(accept)
    kept: list[str] = []
    dropped: list[tuple[str, Grounding]] = []
    seen: set[str] = set()

    for raw in identifiers or []:
        ident = str(raw).strip().upper()
        if not ident or ident in seen:
            continue
        seen.add(ident)
        level = grounding_of(ident, ctx.get(ident), text)
        if level in accept:
            kept.append(ident)
        else:
            dropped.append((ident, level))
    return kept, dropped


# =====================================================================
#  Символьний збір: те, що стоїть у тексті, LLM питати не треба
# =====================================================================
# Гейт вище вміє лише ВІДКИДАТИ назване моделлю. Але на веб-корпусі проблема
# зворотна: 16 статей із 45 містять CVE дослівно (одна — 95 штук у щотижневому
# огляді), а модель називає одиниці. Для дослівного ідентифікатора модель і не
# потрібна: регулярка знаходить його детерміновано, NVD/MITRE підтверджують
# існування і дають авторитетний опис. Галюцинація тут неможлива за побудовою.

#: Стеля на документ. Без неї щотижневий огляд із 95 CVE перетворюється на 95
#: зовнішніх викликів у NVD, а недоброзичливий пост — на самообстріл конвеєра.
DEFAULT_HARVEST_LIMIT = 20

_CVE_SCAN = re.compile(r"(?<![a-z0-9])cve-\d{4}-\d{4,}(?![0-9a-z])")
_TTP_SCAN = re.compile(r"(?<![a-z0-9])t\d{4}(?:\.\d{3})?(?![0-9a-z])")


def _scan(pattern, haystack, limit):
    found, seen = [], set()
    for match in pattern.finditer(haystack):
        ident = match.group(0).upper()
        if ident in seen:
            continue
        seen.add(ident)
        found.append(ident)
        if len(found) >= limit:
            break
    return found


def harvest_literal_ids(text, limit: int = DEFAULT_HARVEST_LIMIT):
    """Дістає з тексту ідентифікатори CVE і технік, які стоять там дослівно.

    Повертає (cve_ids, technique_ids) у порядку появи, без дублікатів, кожен
    список окремо обмежений `limit`. Правдивість ідентифікатора не перевіряє —
    це робота верифікатора (NVD/MITRE); тут гарантується лише те, що рядок
    справді є в документі."""
    haystack = normalize(text)
    if not haystack:
        return [], []
    return _scan(_CVE_SCAN, haystack, limit), _scan(_TTP_SCAN, haystack, limit)
