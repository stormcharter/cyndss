"""
Безпечна побудова HTML-плашок верифікованих CVE/TTP для ui.py.

НАВІЩО ОКРЕМИЙ МОДУЛЬ. Ідентифікатори приходять з Manticore, тобто з ланцюга
«допис у Telegram → LLM-екстракція → верифікація NVD/MITRE», і далі йшли в
`st.markdown(..., unsafe_allow_html=True)` конкатенацією f-рядків без жодного
екранування. Сьогодні туди потрапляють лише підтверджені id, тож дірки немає —
але це безпека через збіг обставин, а не за побудовою.

Захист у три шари, від найнадійнішого до останнього рубежу:
  1. ФІЛЬТР ЗА ФОРМОЮ: усе, що не схоже на CVE-2024-1234 чи T1059.003, просто
     не показуємо. Дозволений список завжди сильніший за заборонений.
  2. ЕКРАНУВАННЯ: те, що пройшло фільтр, усе одно проходить html.escape.
  3. ОБМЕЖЕННЯ КІЛЬКОСТІ: сотні плашок ламають верстку — це теж вплив контенту
     на інтерфейс.

Чому це важливо саме тут: cookie довіреного пристрою в Streamlit неможливо
позначити HttpOnly (див. auth/streamlit_gate.py), тож будь-який XSS в UI
означає крадіжку токена.

Модуль чистий: без streamlit, тестується ізольовано.
"""

import html
import re

# Форма ідентифікаторів. CVE — рік і послідовний номер; MITRE — техніка з
# необовʼязковою під-технікою (T1059.003).
_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,7}$")
_TTP_RE = re.compile(r"^T\d{4}(\.\d{3})?$")

# Скільки плашок максимум показуємо в одному рядку результатів.
MAX_TAGS = 16

_CVE_STYLE = ("background-color:#ff4b4b; color:white; padding:3px 8px; "
              "margin-right:5px; border-radius:3px; font-weight:bold;")
_TTP_STYLE = ("background-color:#0068c9; color:white; padding:3px 8px; "
              "margin-right:5px; border-radius:3px; font-weight:bold;")


def split_ids(raw) -> list:
    """Розбирає рядок «A,B,C» у список без порожніх елементів і пробілів."""
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _keep_valid(raw, pattern, limit) -> list:
    kept = []
    for value in split_ids(raw):
        candidate = value.upper()
        if pattern.match(candidate):
            kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept


def _tag(text: str, style: str) -> str:
    return '<span style="%s">%s</span>' % (style, html.escape(text))


def build_verified_tags_html(cve_ids, ttp_ids) -> str:
    """HTML-плашки верифікованих CVE та MITRE-технік ("" якщо показувати нема що)."""
    cves = _keep_valid(cve_ids, _CVE_RE, MAX_TAGS)
    ttps = _keep_valid(ttp_ids, _TTP_RE, MAX_TAGS - len(cves))
    parts = [_tag(cve, _CVE_STYLE) for cve in cves]
    parts += [_tag("MITRE " + ttp, _TTP_STYLE) for ttp in ttps]
    return "".join(parts)
