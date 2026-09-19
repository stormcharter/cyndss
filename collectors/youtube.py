"""
Колектор YouTube.

Витягує текстові транскрипти (субтитри) відео та складає їх у raw_events.
Підтримує два формати джерел у channels.json -> "youtube":

    пряме відео     https://www.youtube.com/watch?v=ID  /  https://youtu.be/ID  /  ID
    канал/плейлист  https://www.youtube.com/@name  /  ...&list=PL...

Для прямого відео транскрипт тягнеться напряму (youtube-transcript-api).
Для каналу/плейлиста спершу через yt-dlp дістаємо список відео, потім по
кожному — транскрипт. yt-dlp потрібен ЛИШЕ для перебору каналів; якщо його
немає, прямі посилання на відео все одно працюють.

Запуск:  python collect.py youtube   (або  python -m collectors.youtube)
"""

import re

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
)

from .base import RawEventWriter, load_sources

# Бажані мови субтитрів за пріоритетом (українська, російська, англійська).
PREFERRED_LANGS = ["uk", "ru", "en"]

# Скільки відео максимум брати з одного каналу/плейлиста за запуск.
MAX_VIDEOS_PER_CHANNEL = 50


def extract_video_id(url_or_id: str) -> str | None:
    """Дістає 11-символьний id відео з різних форматів посилань."""
    if re.fullmatch(r"[\w-]{11}", url_or_id):
        return url_or_id
    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/)([\w-]{11})", url_or_id)
    return m.group(1) if m else None


def is_channel_or_playlist(url: str) -> bool:
    markers = ("/channel/", "/@", "/c/", "/user/", "list=", "/playlist")
    return any(m in url for m in markers)


def list_channel_videos(url: str) -> list[tuple[str, str]]:
    """Через yt-dlp повертає [(video_id, title), ...] для каналу/плейлиста.

    yt-dlp — опціональна залежність. Якщо не встановлено — повертаємо [].
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        print(
            "    [!] yt-dlp не встановлено — пропускаю канал/плейлист.\n"
            "        Встанови: pip install yt-dlp  (або вкажи прямі посилання на відео)."
        )
        return []

    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,          # не лізти в кожне відео, лише список
        "playlistend": MAX_VIDEOS_PER_CHANNEL,
    }
    videos: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _walk(entries) -> None:
        # Для каналу yt-dlp може повернути вкладки (Videos/Shorts/Live) як вкладені
        # плейлисти. Спускаємось на рівень нижче і беремо лише валідні id відео.
        for entry in (entries or []):
            if len(videos) >= MAX_VIDEOS_PER_CHANNEL:
                return
            if entry.get("entries"):
                _walk(entry["entries"])
                continue
            vid = entry.get("id")
            if vid and re.fullmatch(r"[\w-]{11}", vid) and vid not in seen:
                seen.add(vid)
                videos.append((vid, entry.get("title") or vid))

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        _walk(info.get("entries"))
    except Exception as e:
        print(f"    [!] yt-dlp не зміг розібрати {url}: {e}")
    return videos[:MAX_VIDEOS_PER_CHANNEL]


def fetch_transcript(video_id: str) -> str | None:
    """Повертає суцільний текст субтитрів відео або None, якщо їх немає."""
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = listing.find_transcript(PREFERRED_LANGS)
        except NoTranscriptFound:
            # будь-які доступні (зокрема авто-згенеровані) субтитри
            available = [t.language_code for t in listing]
            transcript = listing.find_generated_transcript(available)
        segments = transcript.fetch()
        return " ".join(seg["text"] for seg in segments if seg.get("text")).strip()
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        print(f"    [!] Не вдалося отримати транскрипт {video_id}: {e}")
        return None


def collect() -> None:
    sources = load_sources("youtube")
    if not sources:
        print("[YouTube] Список 'youtube' у channels.json порожній — нічого збирати.")
        return

    writer = RawEventWriter()
    added = 0
    try:
        for src in sources:
            print(f"[YouTube] Обробка джерела: {src}")

            if is_channel_or_playlist(src):
                videos = list_channel_videos(src)
            else:
                vid = extract_video_id(src)
                if not vid:
                    print(f"    [!] Не розпізнав id відео в посиланні: {src}")
                    continue
                videos = [(vid, vid)]

            for vid, title in videos:
                text = fetch_transcript(vid)
                if not text:
                    continue
                ok = writer.write(
                    source_type="youtube",
                    text=text,
                    channel_name=title,
                    url=f"https://www.youtube.com/watch?v={vid}",
                    message_id=vid,
                    raw_payload={"title": title},
                )
                if ok:
                    added += 1
    finally:
        writer.close()

    print(f"[YouTube] Готово. Додано нових відео: {added}")


if __name__ == "__main__":
    collect()
