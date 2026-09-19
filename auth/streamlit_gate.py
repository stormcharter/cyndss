"""
Сторінка входу ONYX і ворота перед усім рештою UI.

ЯК ЦЕ ПРАЦЮЄ. `require_login()` викликається на самому початку ui.py — ДО
будь-якого запиту в Mongo/Manticore. Якщо користувач не впізнаний, малюється
форма входу і виконання скрипта зупиняється (`st.stop()`), тож жоден рядок з
даними не встигає навіть завантажитись.

Порядок впізнавання:
  1. `st.session_state` — активна вкладка (живе до перезавантаження сторінки);
  2. cookie довіреного пристрою — «запамʼятати це устройство»; токен при
     кожному використанні ротується, старий одразу мертвий;
  3. форма логін/пароль.

ЧЕСНЕ ОБМЕЖЕННЯ. Cookie ставиться з JavaScript (Streamlit не дає доступу до
заголовків відповіді), тому позначку HttpOnly видати неможливо — токен
пристрою в принципі читається скриптом на сторінці. Отже XSS в UI = крадіжка
токена. Саме тому: (а) токен одноразовий і ротується, (б) пристрій видно й
можна відкликати, (в) місця з unsafe_allow_html в ui.py треба тримати під
контролем. Для рівня «диплом на спільному боксі» це прийнятний компроміс;
у справжньому проді сесію тримав би реверс-проксі, а не сам застосунок.
"""

import os
from datetime import datetime, timedelta, timezone

import extra_streamlit_components as stx
import pymongo
import streamlit as st
from pymongo.errors import PyMongoError

from .store import DEVICE_TTL_DAYS, AuthStore

COOKIE_NAME = "onyx_device"

# Cookie з позначкою Secure не ставиться по звичайному HTTP. Зовнішній вхід у
# ONYX — тільки HTTPS, тож за замовчуванням True; для налагодження по локальній
# мережі (http://<LAN-IP>) можна вимкнути через ONYX_COOKIE_SECURE=0.
COOKIE_SECURE = os.getenv("ONYX_COOKIE_SECURE", "1") not in ("0", "false", "False")

_REASON_MESSAGES = {
    # Однакове формулювання для «немає такого логіна» і «невірний пароль» —
    # інакше форма підказує, які облікові записи існують.
    "invalid": "Невірний логін або пароль",
    "locked": "Забагато невдалих спроб. Спробуй за %s",
    "disabled": "Обліковий запис вимкнено",
}


# Скільки чекати на Mongo. Дефолтні 30 с pymongo тут неприйнятні: якщо база
# лежить, сторінка входу просто «висить» півхвилини, і людина не розуміє, чому.
SERVER_SELECTION_TIMEOUT_MS = 5000

# Як часто вже відкрита вкладка перепитує базу, чи обліковий запис іще діє.
SESSION_RECHECK_SECONDS = 300


@st.cache_resource
def get_auth_store() -> AuthStore:
    """Один AuthStore на процес (pymongo сам тримає пул зʼєднань)."""
    uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/?directConnection=true")
    client = pymongo.MongoClient(
        uri,
        serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
        connectTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
    )
    db = client[os.getenv("MONGO_DB", "threat_intel")]
    store = AuthStore(db["users"], db["devices"])
    try:
        store.ensure_indexes()
    except Exception:
        # Індекси могли бути створені раніше, бракувати прав або база ще не
        # піднялась — вхід від цього не залежить, тож не валимо застосунок.
        pass
    return store


def _human_delay(seconds: int) -> str:
    if seconds >= 60:
        return "%d хв" % ((seconds + 59) // 60)
    return "%d с" % seconds


def _device_label() -> str:
    """Мітка пристрою для списку — з User-Agent, якщо він доступний."""
    try:
        agent = st.context.headers.get("User-Agent", "")
    except Exception:
        agent = ""
    for name in ("Firefox", "Edg", "Chrome", "Safari"):
        if name in agent:
            system = "Windows" if "Windows" in agent else (
                "Android" if "Android" in agent else (
                    "iOS" if "iPhone" in agent or "iPad" in agent else (
                        "macOS" if "Mac OS" in agent else "Linux")))
            return "%s / %s" % (name.replace("Edg", "Edge"), system)
    return "невідомий браузер"


def _set_device_cookie(cookies: stx.CookieManager, token: str) -> None:
    cookies.set(
        COOKIE_NAME,
        token,
        key="onyx_set_device",
        expires_at=datetime.now(timezone.utc) + timedelta(days=DEVICE_TTL_DAYS),
        secure=COOKIE_SECURE,
        same_site="strict",
    )


def _clear_device_cookie(cookies: stx.CookieManager) -> None:
    try:
        cookies.delete(COOKIE_NAME, key="onyx_del_device")
    except Exception:
        pass


def _session_still_valid(store: AuthStore, user: dict) -> bool:
    """Раз на SESSION_RECHECK_SECONDS перепитує базу, чи запис іще діє.

    Без цього `disable` в CLI нічого не давав би вже відкритій вкладці: вона
    тримає користувача в session_state і в базу більше не зазирає. Не на
    кожному прогоні — інакше кожен рух повзунка ходив би в Mongo.
    """
    now = datetime.now(timezone.utc)
    checked_at = st.session_state.get("auth_user_checked_at")
    if isinstance(checked_at, datetime) and \
            (now - checked_at).total_seconds() < SESSION_RECHECK_SECONDS:
        return True
    try:
        alive = store.is_active(user.get("username"))
    except PyMongoError:
        return True   # база лежить — не викидаємо людину з робочої сесії
    if alive:
        st.session_state["auth_user_checked_at"] = now
    return alive


def _try_device_login(store: AuthStore, cookies: stx.CookieManager):
    """Спроба входу за cookie. Повертає користувача або None."""
    if st.session_state.get("auth_device_checked"):
        return None
    token = cookies.get(COOKIE_NAME)
    if not token:
        # Компонент cookie віддає значення не на першому проході — не
        # позначаємо спробу як вичерпану, щоб автологін спрацював після rerun.
        return None
    st.session_state["auth_device_checked"] = True
    try:
        result = store.authenticate_device(token)
    except PyMongoError:
        return None   # база лежить — покажемо звичайну форму входу
    if not result.ok:
        _clear_device_cookie(cookies)
        return None
    _set_device_cookie(cookies, result.token)   # ротація: стара cookie вже недійсна
    return result.user


def _render_login_form(store: AuthStore, cookies: stx.CookieManager) -> None:
    st.markdown("## 🛡️ ONYX — вхід")
    st.caption("Платформа кіберрозвідки. Доступ лише для зареєстрованих аналітиків.")

    with st.form("onyx_login", clear_on_submit=False):
        username = st.text_input("Логін", autocomplete="username")
        password = st.text_input("Пароль", type="password",
                                 autocomplete="current-password")
        remember = st.checkbox(
            "Запамʼятати цей пристрій",
            value=False,
            help="Не вмикай на спільному компʼютері: вхід зберігається %d днів."
                 % DEVICE_TTL_DAYS,
        )
        submitted = st.form_submit_button("Увійти", type="primary")

    if not submitted:
        return

    try:
        result = store.authenticate(username, password)
    except PyMongoError:
        # Червоний трейсбек на сторінці входу — це і паніка користувача, і
        # безкоштовна розвідка для стороннього (видно шляхи й версії).
        st.error("База недоступна. Спробуй пізніше або скажи адміністратору.")
        return

    if not result.ok:
        if result.reason == "locked":
            st.error(_REASON_MESSAGES["locked"] % _human_delay(result.retry_after))
        else:
            st.error(_REASON_MESSAGES.get(result.reason, _REASON_MESSAGES["invalid"]))
        return

    st.session_state["auth_user"] = result.user
    st.session_state["auth_user_checked_at"] = datetime.now(timezone.utc)
    if remember:
        token = store.remember_device(result.user["username"], label=_device_label())
        _set_device_cookie(cookies, token)
    st.rerun()


def _render_sidebar(store: AuthStore, user: dict, cookies: stx.CookieManager) -> None:
    with st.sidebar:
        st.markdown("---")
        st.caption("Ви увійшли як **%s**" % user.get("username", "?"))

        try:
            devices = store.list_devices(user["username"])
        except PyMongoError:
            devices = []
        if devices:
            with st.expander("Довірені пристрої (%d)" % len(devices)):
                for device in devices:
                    # st.write екранує — мітку формує User-Agent, тобто ввід ззовні.
                    st.write("**%s**" % (device.get("label") or "пристрій"))
                    last_seen = device.get("last_seen_at")
                    if isinstance(last_seen, datetime):
                        st.caption("востаннє: %s" % last_seen.strftime("%d.%m.%Y %H:%M"))
                    if st.button("Відкликати", key="revoke_%s" % device["_id"]):
                        store.revoke_device(user["username"], device["_id"])
                        st.rerun()

        if st.button("Вийти", key="onyx_logout"):
            _clear_device_cookie(cookies)
            for key in ("auth_user", "auth_user_checked_at", "auth_device_checked"):
                st.session_state.pop(key, None)
            st.rerun()


def require_login() -> dict:
    """Пускає далі лише впізнаного користувача. Інакше малює форму і зупиняє
    скрипт — жоден запит до даних не встигає виконатись."""
    store = get_auth_store()
    # Один екземпляр на прогін скрипта: CookieManager — це віджет, тож його не
    # можна ані кешувати (@st.cache_resource виконає його лише при промаху й
    # cookie перестане перечитуватись), ані створювати двічі за прогін
    # (конфлікт ключів віджета).
    cookies = stx.CookieManager(key="onyx_cookies")

    user = st.session_state.get("auth_user")
    if user is not None and not _session_still_valid(store, user):
        # Доступ відкликали, доки вкладка була відкрита.
        for key in ("auth_user", "auth_user_checked_at", "auth_device_checked"):
            st.session_state.pop(key, None)
        user = None
        st.warning("Доступ до облікового запису відкликано. Увійдіть знову.")

    if user is None:
        user = _try_device_login(store, cookies)
        if user is not None:
            st.session_state["auth_user"] = user
            st.session_state["auth_user_checked_at"] = datetime.now(timezone.utc)

    if user is None:
        _render_login_form(store, cookies)
        st.stop()

    _render_sidebar(store, user, cookies)
    return user
