"""
Юніт-тести для collectors/net_guard.py — захист веб-колектора від SSRF.

Мотивація (security): collect_feed() бере `link` ПРЯМО зі стрічки RSS —
тобто з-під контролю того, хто цю стрічку віддає. Далі httpx.get(link,
follow_redirects=True) виконувався без жодної перевірки схеми чи хоста.
Колектор бігає на тому ж хості, що й внутрішні сервіси (Mongo 27018,
Manticore 9999/9998, Ollama 11434), тож підроблений фід міг змусити його
ходити на 127.0.0.1 / приватні мережі / cloud-metadata 169.254.169.254 —
і складати відповіді в raw_events.

Тести не ходять у мережу: DNS-резолвер і HTTP-клієнт інжектяться.

Запуск:
    pytest tests/test_net_guard.py -v
або:
    python tests/test_net_guard.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.net_guard import is_safe_url, safe_get  # noqa: E402


PUBLIC_IP = "93.184.216.34"


def resolver_stub(mapping, default=PUBLIC_IP):
    """Фейковий DNS: hostname -> список IP."""
    def _resolve(host):
        if host in mapping:
            return mapping[host]
        if default is None:
            raise OSError("NXDOMAIN: " + host)
        return [default]
    return _resolve


ALLOW_ALL = resolver_stub({})


# --- схеми -----------------------------------------------------------------
def test_only_http_and_https_allowed():
    assert is_safe_url("http://example.com/a", resolver=ALLOW_ALL) is True
    assert is_safe_url("https://example.com/a", resolver=ALLOW_ALL) is True


def test_dangerous_schemes_blocked():
    for url in [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com:9999/_",
        "javascript:alert(1)",
        "data:text/html,hi",
        "mongodb://127.0.0.1:27018/",
        "//example.com/protocol-relative",
        "example.com/no-scheme",
        "",
    ]:
        assert is_safe_url(url, resolver=ALLOW_ALL) is False, url


def test_none_and_non_string_blocked():
    assert is_safe_url(None, resolver=ALLOW_ALL) is False
    assert is_safe_url(12345, resolver=ALLOW_ALL) is False


def test_url_without_host_blocked():
    assert is_safe_url("http:///path", resolver=ALLOW_ALL) is False


# --- loopback --------------------------------------------------------------
def test_loopback_blocked():
    for url in [
        "http://127.0.0.1/",
        "http://127.0.0.1:9999/",
        "http://127.1.2.3:9998/",
        "http://localhost:11434/api/tags",
        "http://LOCALHOST/",
        "http://[::1]:27018/",
    ]:
        assert is_safe_url(url, resolver=ALLOW_ALL) is False, url


# --- приватні мережі RFC1918 ------------------------------------------------
def test_rfc1918_blocked():
    for url in [
        "http://10.0.0.5/",
        "http://10.255.255.254/x",
        "http://172.16.0.1/",
        "http://172.31.255.254/",
        "http://192.168.1.1/",
        "http://192.168.0.254:8501/",
    ]:
        assert is_safe_url(url, resolver=ALLOW_ALL) is False, url


def test_public_ip_in_172_range_still_allowed():
    """172.32.0.0 вже НЕ приватна (RFC1918 закінчується на 172.31) — не ріжемо зайве."""
    assert is_safe_url("http://172.32.0.1/", resolver=ALLOW_ALL) is True


# --- link-local / cloud metadata --------------------------------------------
def test_link_local_and_cloud_metadata_blocked():
    for url in [
        "http://169.254.0.1/",
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://[fe80::1]/",
    ]:
        assert is_safe_url(url, resolver=ALLOW_ALL) is False, url


# --- інші спецдіапазони -----------------------------------------------------
def test_unspecified_and_reserved_blocked():
    for url in [
        "http://0.0.0.0/",
        "http://0.0.0.0:9999/",
        "http://[::]/",
        "http://[fc00::1]/",
        "http://255.255.255.255/",
        "http://224.0.0.1/",
    ]:
        assert is_safe_url(url, resolver=ALLOW_ALL) is False, url


# --- DNS: імена, що резолвляться у внутрішні адреси -------------------------
def test_hostname_resolving_to_loopback_blocked():
    """DNS rebinding у лоб: публічне ім'я -> 127.0.0.1."""
    resolver = resolver_stub({"evil.example.com": ["127.0.0.1"]})
    assert is_safe_url("http://evil.example.com/x", resolver=resolver) is False


def test_hostname_resolving_to_private_blocked():
    resolver = resolver_stub({"intranet.example.com": ["192.168.10.10"]})
    assert is_safe_url("http://intranet.example.com/", resolver=resolver) is False


def test_hostname_with_any_internal_answer_blocked():
    """Якщо ХОЧА Б одна A-запис внутрішня — блокуємо (обхід round-robin)."""
    resolver = resolver_stub({"mixed.example.com": [PUBLIC_IP, "10.1.2.3"]})
    assert is_safe_url("http://mixed.example.com/", resolver=resolver) is False


def test_unresolvable_hostname_blocked():
    resolver = resolver_stub({}, default=None)
    assert is_safe_url("http://nowhere.invalid/", resolver=resolver) is False


def test_normal_public_site_allowed():
    resolver = resolver_stub({"thehackernews.com": ["104.21.0.1"]})
    assert is_safe_url("https://thehackernews.com/2024/01/post.html",
                       resolver=resolver) is True


# --- safe_get: редіректи ----------------------------------------------------
class FakeResponse(object):
    def __init__(self, status_code, headers=None, text="OK"):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP " + str(self.status_code))


def fake_client(script):
    """script: url -> FakeResponse. Пише відвідані URL у .visited."""
    visited = []

    def _get(url, **kwargs):
        visited.append(url)
        assert kwargs.get("follow_redirects") is False, \
            "safe_get не має віддавати редіректи на відкуп httpx"
        return script[url]

    _get.visited = visited
    return _get


def test_safe_get_returns_response_for_public_url():
    get_fn = fake_client({"http://example.com/a": FakeResponse(200, text="BODY")})
    resp = safe_get("http://example.com/a", get_fn=get_fn, resolver=ALLOW_ALL)
    assert resp.text == "BODY"
    assert get_fn.visited == ["http://example.com/a"]


def test_safe_get_rejects_unsafe_initial_url():
    get_fn = fake_client({})
    try:
        safe_get("http://127.0.0.1:9999/", get_fn=get_fn, resolver=ALLOW_ALL)
    except ValueError:
        assert get_fn.visited == [], "запит не мав відбутись узагалі"
        return
    raise AssertionError("небезпечний URL мав бути відхилений")


def test_safe_get_blocks_redirect_to_internal_host():
    """ГОЛОВНИЙ кейс: зовні URL публічний, але 302 веде на внутрішній сервіс."""
    get_fn = fake_client({
        "http://example.com/a": FakeResponse(
            302, headers={"location": "http://127.0.0.1:11434/api/tags"}),
    })
    try:
        safe_get("http://example.com/a", get_fn=get_fn, resolver=ALLOW_ALL)
    except ValueError:
        assert get_fn.visited == ["http://example.com/a"], get_fn.visited
        return
    raise AssertionError("редірект на loopback мав бути відхилений")


def test_safe_get_blocks_redirect_to_cloud_metadata():
    get_fn = fake_client({
        "http://example.com/a": FakeResponse(
            302, headers={"location": "http://169.254.169.254/latest/meta-data/"}),
    })
    try:
        safe_get("http://example.com/a", get_fn=get_fn, resolver=ALLOW_ALL)
    except ValueError:
        return
    raise AssertionError("редірект на cloud-metadata мав бути відхилений")


def test_safe_get_follows_safe_redirect():
    get_fn = fake_client({
        "http://example.com/a": FakeResponse(
            301, headers={"location": "https://example.com/b"}),
        "https://example.com/b": FakeResponse(200, text="FINAL"),
    })
    resp = safe_get("http://example.com/a", get_fn=get_fn, resolver=ALLOW_ALL)
    assert resp.text == "FINAL"
    assert get_fn.visited == ["http://example.com/a", "https://example.com/b"]


def test_safe_get_resolves_relative_redirect():
    get_fn = fake_client({
        "http://example.com/a/b": FakeResponse(302, headers={"location": "/c"}),
        "http://example.com/c": FakeResponse(200, text="REL"),
    })
    resp = safe_get("http://example.com/a/b", get_fn=get_fn, resolver=ALLOW_ALL)
    assert resp.text == "REL"


def test_safe_get_stops_on_redirect_loop():
    get_fn = fake_client({
        "http://example.com/a": FakeResponse(302, headers={"location": "http://example.com/b"}),
        "http://example.com/b": FakeResponse(302, headers={"location": "http://example.com/a"}),
    })
    try:
        safe_get("http://example.com/a", get_fn=get_fn, resolver=ALLOW_ALL,
                 max_redirects=3)
    except ValueError:
        assert len(get_fn.visited) <= 4, get_fn.visited
        return
    raise AssertionError("нескінченний редірект мав бути обірваний")


# --- ручний запуск без pytest ----------------------------------------------
if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS " + name)
            passed += 1
        except Exception:
            print("FAIL " + name)
            traceback.print_exc()
            failed += 1
    print("\n%d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)
