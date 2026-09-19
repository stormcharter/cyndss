"""
Захист вихідних HTTP-запитів колектора від SSRF.

ЗАГРОЗА. `link` у collect_feed() приходить із RSS-стрічки, тобто його диктує
власник стрічки (або той, хто її підмінив). Колектор працює на тому ж хості,
де слухають внутрішні сервіси без автентифікації — Mongo (27018), Manticore
(9999/9998), Ollama (11434), — а також, у хмарі, метадані інстансу
(169.254.169.254). Запит без перевірки схеми/адреси перетворює колектор на
проксі в довірену мережу, ще й із записом відповіді в raw_events.

ЩО РОБИМО.
  * whitelist схем: лише http/https;
  * резолвимо хост ПЕРЕД запитом і блокуємо, якщо ХОЧА Б одна відповідь DNS
    веде у loopback / приватну / link-local / зарезервовану мережу;
  * редіректи НЕ віддаємо http-клієнту (follow_redirects=False) — кожен хоп
    перевіряємо самі, інакше 302 на 127.0.0.1 обходить усю перевірку.

Лишається вузьке вікно TOCTOU (DNS може віддати іншу адресу на самому запиті) —
повністю закрити його можна лише піном IP у конекторі; для колектора OSINT це
надмірно, а планка «випадковий фід не дотягнеться до внутрішніх портів» —
закрита.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

ALLOWED_SCHEMES = ("http", "https")
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
DEFAULT_MAX_REDIRECTS = 5

# Імена, що завжди вказують на локальну машину (навіть якщо DNS бреше).
BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


def _default_resolver(host):
    """Реальний DNS: усі адреси, які віддає getaddrinfo (A + AAAA)."""
    infos = socket.getaddrinfo(host, None)
    return [info[4][0] for info in infos]


def _is_blocked_ip(ip_str):
    """Чи веде адреса всередину (loopback/приватна/link-local/зарезервована)."""
    try:
        ip = ipaddress.ip_address(str(ip_str).split("%")[0])   # %scope для IPv6
    except ValueError:
        return True          # не змогли розібрати — вважаємо небезпечним
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped  # ::ffff:127.0.0.1 — той самий loopback
    return bool(
        ip.is_private            # 10/8, 172.16/12, 192.168/16, 127/8, 169.254/16, fc00::/7
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_safe_url(url, resolver=None):
    """True, якщо за URL можна безпечно піти назовні.

    resolver — функція hostname -> список IP (інжектиться в тестах)."""
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        return False

    try:
        host = parts.hostname
    except ValueError:      # напр. зіпсований IPv6-літерал у дужках
        return False
    if not host:
        return False
    host = host.rstrip(".").lower()
    if not host or host in BLOCKED_HOSTNAMES:
        return False

    # Літеральна IP-адреса — перевіряємо без DNS.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return not _is_blocked_ip(host)

    resolve = resolver or _default_resolver
    try:
        addresses = list(resolve(host))
    except Exception:
        return False        # не резолвиться — не йдемо
    if not addresses:
        return False
    return not any(_is_blocked_ip(a) for a in addresses)


def safe_get(url, get_fn, headers=None, timeout=30.0,
             max_redirects=DEFAULT_MAX_REDIRECTS, resolver=None):
    """GET із перевіркою КОЖНОГО хопа. Кидає ValueError на небезпечному URL.

    get_fn — httpx.get (або сумісний); викликається з follow_redirects=False,
    бо ланцюг редіректів ми ведемо самі."""
    current = url
    for _ in range(max_redirects + 1):
        if not is_safe_url(current, resolver=resolver):
            raise ValueError("SSRF-захист: небезпечний або нерезолвний URL: %r" % (current,))
        response = get_fn(current, headers=headers, timeout=timeout,
                          follow_redirects=False)
        if response.status_code not in REDIRECT_STATUSES:
            return response
        location = (response.headers or {}).get("location") or (response.headers or {}).get("Location")
        if not location:
            return response
        current = urljoin(current, location)
    raise ValueError("SSRF-захист: забагато редіректів для %r" % (url,))
