"""
Юніт-тести для collectors/website.py — SSRF у collect_feed() повз net_guard.

Мотивація (security): feedparser.parse(url) якає стрічку САМ (власний
_FeedURLHandler на базі urllib.request.HTTPRedirectHandler) — робить власний
HTTP-запит і сам слідує редіректам, повністю обходячи is_safe_url()/safe_get()
з net_guard (які перевіряють КОЖЕН хоп редіректу окремо). Раніше перевірявся
лише початковий feed_url; підроблений/скомпрометований фід міг зарезолвитись
у публічну IP (проходить is_safe_url), а сервер віддати 302 на внутрішній
сервіс (127.0.0.1:9999 Manticore, 169.254.169.254 cloud-metadata) — і
feedparser сам туди сходить, обійшовши захист.

Фікс: якаємо XML стрічки самі через safe_get() (він валідує кожен хоп), і
віддаємо feedparser уже ЗАВАНТАЖЕНИЙ вміст (bytes), а не сирий URL.

Тести не ходять у мережу: httpx.get і feedparser.parse інжектяться/мавпляться.
Хости — літеральні IP (як PUBLIC_IP у test_net_guard.py), щоб не залежати
від DNS-резолвера.

Запуск:
    pytest tests/test_website.py -v
або:
    python tests/test_website.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feedparser  # noqa: E402

import collectors.website as website  # noqa: E402

PUBLIC_IP = "93.184.216.34"
FEED_URL = "http://" + PUBLIC_IP + "/feed"


class FakeResponse(object):
    """Мінімальна імітація httpx.Response для тестів safe_get()."""

    def __init__(self, status_code, headers=None, content=b"", text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content
        self.text = text or content.decode("utf-8", "replace")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP " + str(self.status_code))


def fake_httpx_get(script):
    """script: url -> FakeResponse. Пише відвідані URL у .visited (як у
    fake_client з tests/test_net_guard.py)."""
    visited = []

    def _get(url, **kwargs):
        visited.append(url)
        assert kwargs.get("follow_redirects") is False, \
            "safe_get не має віддавати редіректи на відкуп httpx"
        return script[url]

    _get.visited = visited
    return _get


class FakeWriter(object):
    """Замінник RawEventWriter — не ходить у Mongo, лише записує виклики."""

    def __init__(self):
        self.calls = []

    def write(self, **kwargs):
        self.calls.append(kwargs)
        return True


# --- вміст фіда йде через net_guard, а не напряму в feedparser -------------
def test_collect_feed_passes_downloaded_content_not_raw_url(monkeypatch):
    """feedparser.parse() має отримати ЗАВАНТАЖЕНИЙ вміст (bytes), а не сирий
    URL-рядок — інакше feedparser сам робить запит повз net_guard."""
    body = b"<rss><channel><title>T</title></channel></rss>"
    get_fn = fake_httpx_get({FEED_URL: FakeResponse(200, content=body)})
    monkeypatch.setattr(website.httpx, "get", get_fn)

    captured = {}
    real_parse = feedparser.parse

    def spy_parse(data, *a, **kw):
        captured["arg"] = data
        # Порожній фід — цикл по entries нічого не зробить, writer не чіпаємо.
        return real_parse(b"<rss><channel></channel></rss>")

    monkeypatch.setattr(website.feedparser, "parse", spy_parse)

    writer = FakeWriter()
    website.collect_feed(writer, FEED_URL)

    assert "arg" in captured, "feedparser.parse не викликався"
    assert captured["arg"] != FEED_URL, \
        "feedparser.parse отримав сирий URL — сам піде в мережу повз net_guard"
    assert captured["arg"] == body, captured["arg"]
    assert get_fn.visited == [FEED_URL], get_fn.visited


# --- редірект у ланцюжку фіда на внутрішній хост блокується ----------------
def test_collect_feed_blocks_redirect_to_internal_host(monkeypatch):
    """ГОЛОВНИЙ SSRF-кейс: feed_url публічний і проходить is_safe_url, але
    сервер віддає 302 на внутрішній сервіс (тут — 127.0.0.1:9999,
    Manticore). collect_feed НЕ має туди сходити і НЕ має викликати
    feedparser.parse узагалі."""
    get_fn = fake_httpx_get({
        FEED_URL: FakeResponse(
            302, headers={"location": "http://127.0.0.1:9999/"}),
    })
    monkeypatch.setattr(website.httpx, "get", get_fn)

    parse_calls = []
    real_parse = website.feedparser.parse

    def spy_parse(data, *a, **kw):
        parse_calls.append(data)
        # Якщо тест все ж дійде сюди (бага не пофіксили) — повертаємо
        # порожній розібраний фід, щоб collect_feed не впав з AttributeError
        # і ми побачили саме наш assert, а не побічний ефект бага.
        return real_parse(b"<rss><channel></channel></rss>")

    monkeypatch.setattr(website.feedparser, "parse", spy_parse)

    writer = FakeWriter()
    added = website.collect_feed(writer, FEED_URL)

    assert added == 0
    assert parse_calls == [], "feedparser.parse не мав викликатись — редірект на internal host"
    assert writer.calls == []
    # safe_get мав спробувати лише початковий URL і зупинитись на редіректі,
    # НЕ пішовши на 127.0.0.1.
    assert get_fn.visited == [FEED_URL], get_fn.visited


# --- ручний запуск без pytest (monkeypatch руками, без фікстури pytest) ----
class _ManualMonkeypatch(object):
    """Мінімальна заміна pytest-фікстури monkeypatch для запуску як скрипт."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)


if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        mp = _ManualMonkeypatch()
        try:
            fn(mp)
            print("PASS " + name)
            passed += 1
        except Exception:
            print("FAIL " + name)
            traceback.print_exc()
            failed += 1
        finally:
            mp.undo()
    print("\n%d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)
