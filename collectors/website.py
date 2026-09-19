"""
Колектор веб-сайтів.

Підтримує два типи джерел у channels.json -> "websites":

    RSS/Atom-стрічка   https://feeds.example.com/rss   (перебираємо записи)
    звичайна сторінка  https://example.com/article     (витягуємо основний текст)

Тип визначається автоматично: за розширенням/шляхом (.xml, /feed, /rss) або
за вмістом перших байтів (<rss / <feed). Основний текст статей дістаємо через
trafilatura (відсікає меню, рекламу, футер), стрічки парсимо через feedparser.

Усе складається в raw_events із source_type="website", далі цим займається
shipper (ембеддинг, дедуплікація, витяг CVE/MITRE).

Запуск:  python collect.py websites   (або  python -m collectors.website)
"""

import datetime

import feedparser
import httpx
import trafilatura

from .base import RawEventWriter, load_sources
from .net_guard import is_safe_url, safe_get

HEADERS = {"User-Agent": "Mozilla/5.0 (CyberIntel collector)"}

# Скільки записів максимум брати з однієї стрічки за запуск.
MAX_ENTRIES_PER_FEED = 30


def _domain(url: str) -> str:
    return url.split("/")[2] if "://" in url else url


def fetch_html(url: str) -> str | None:
    """SECURITY: URL сюди приходить зі стрічки (недовірене джерело), а колектор
    бачить внутрішні сервіси на 127.0.0.1 — тому йдемо через safe_get, який
    перевіряє схему/адресу і КОЖЕН редірект окремо (див. net_guard)."""
    try:
        resp = safe_get(url, get_fn=httpx.get, headers=HEADERS, timeout=30.0)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"    [!] Не вдалося завантажити {url}: {e}")
        return None


def looks_like_feed(url: str, html: str | None) -> bool:
    if url.endswith((".xml", ".rss", ".atom")) or "/feed" in url or "/rss" in url:
        return True
    if html:
        head = html[:600].lower()
        if "<rss" in head or "<feed" in head or "application/rss" in head:
            return True
    return False


def extract_article(url: str, html: str | None = None) -> tuple[str | None, str | None]:
    """Повертає (text, title) основного контенту сторінки."""
    if html is None:
        html = fetch_html(url)
    if not html:
        return None, None
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    title = None
    try:
        meta = trafilatura.extract_metadata(html)
        if meta:
            title = meta.title
    except Exception:
        pass
    return text, title


def collect_feed(writer: RawEventWriter, feed_url: str) -> int:
    # SECURITY: feedparser.parse(url) якає стрічку САМ (власний urllib-хендлер,
    # власні редіректи) — це повністю обходить net_guard, бо перевіряється
    # лише початковий feed_url, а не кожен хоп редіректу. Тому якаємо самі
    # через safe_get (валідує КОЖЕН хоп) і віддаємо feedparser уже завантажений
    # вміст, а не сирий URL.
    if not is_safe_url(feed_url):
        print(f"    [!] Небезпечна або нерезолвна адреса стрічки, пропускаю: {feed_url}")
        return 0
    try:
        resp = safe_get(feed_url, get_fn=httpx.get, headers=HEADERS, timeout=30.0)
        resp.raise_for_status()
    except Exception as e:
        print(f"    [!] Не вдалося завантажити стрічку {feed_url}: {e}")
        return 0
    parsed = feedparser.parse(resp.content)
    site_title = parsed.feed.get("title") if getattr(parsed, "feed", None) else None
    added = 0

    for entry in parsed.entries[:MAX_ENTRIES_PER_FEED]:
        link = entry.get("link")
        if not link:
            continue
        # Посилання диктує власник стрічки — SSRF-фільтр перед будь-яким запитом.
        if not is_safe_url(link):
            print(f"    [!] Пропускаю небезпечне посилання зі стрічки: {link}")
            continue

        # Намагаємось дістати повний текст статті; якщо ні — беремо опис зі стрічки.
        text, _ = extract_article(link)
        if not text:
            text = entry.get("summary") or entry.get("description") or ""

        published = None
        if entry.get("published_parsed"):
            published = datetime.datetime(*entry["published_parsed"][:6])

        ok = writer.write(
            source_type="website",
            text=text,
            channel_name=site_title or _domain(link),
            url=link,
            collected_at=published,
            raw_payload={"feed": feed_url, "entry_title": entry.get("title")},
        )
        if ok:
            added += 1
    return added


def collect_page(writer: RawEventWriter, url: str) -> int:
    if not is_safe_url(url):
        print(f"    [!] Небезпечна або нерезолвна адреса сторінки, пропускаю: {url}")
        return 0
    html = fetch_html(url)
    if looks_like_feed(url, html):
        return collect_feed(writer, url)

    text, title = extract_article(url, html)
    if not text:
        print(f"    [!] Не вдалося витягти текст зі сторінки: {url}")
        return 0

    ok = writer.write(
        source_type="website",
        text=text,
        channel_name=title or _domain(url),
        url=url,
        raw_payload={"page_title": title},
    )
    return 1 if ok else 0


def collect() -> None:
    sources = load_sources("websites")
    if not sources:
        print("[Web] Список 'websites' у channels.json порожній — нічого збирати.")
        return

    writer = RawEventWriter()
    added = 0
    try:
        for src in sources:
            print(f"[Web] Обробка джерела: {src}")
            if looks_like_feed(src, None):
                added += collect_feed(writer, src)
            else:
                added += collect_page(writer, src)
    finally:
        writer.close()

    print(f"[Web] Готово. Додано нових матеріалів: {added}")


if __name__ == "__main__":
    collect()
