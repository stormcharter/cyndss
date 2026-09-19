"""
Юніт-тести для стоп-листа `semantic_reasoner.extract_terms()`.

Проблема: панель «Гіпотези / предикт» будувала ланцюги через сміттєві
проміжні вузли на кшталт `found`, `files` — це generic-слова, не CTI-концепти
(джерело — пост, ніяк не повʼязаний із загрозами, проходив через шум).

Ці тести фіксують:
  1. Розширений клас generic-шуму (found/files/video/anomalies тощо) НЕ
     потрапляє в `extract_terms()`.
  2. Реальні CTI-терміни (ransomware, exploit, backdoor тощо) виживають —
     стоп-лист не повинен зачіпати домен.

Запуск:
    pytest tests/test_extract_terms_stopwords.py -v
або як звичайний скрипт (без pytest):
    python tests/test_extract_terms_stopwords.py
"""

import os
import sys

# Дозволяємо запуск і як `pytest`, і як `python tests/...` з кореня репо.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_reasoner import extract_terms  # noqa: E402


# Слова, які реально спричинили сміттєвий ланцюг у прод-даних
# (MITRE T1494 → found → ransomware, джерело — пост про "тіні від радіації").
GENERIC_NOISE_WORDS = [
    "found", "files", "video", "anomalies", "mysterious", "unexplained",
]

# Ширший набір generic-шуму, доданого в стоп-лист цим патчем.
EXTRA_GENERIC_NOISE_WORDS = [
    "find", "finds", "file", "videos", "photo", "photos", "image", "images",
    "anomaly", "story", "stories", "news", "today", "yesterday", "tomorrow",
    "data", "info", "information", "thing", "things", "people", "world",
    "place", "places", "point", "points", "case", "cases", "part", "parts",
    "week", "weeks", "month", "months", "year", "years", "hour", "hours",
    "time", "times", "first", "last", "next", "good", "better", "best",
    "real", "main", "full", "open", "close", "high", "long", "small",
    "large", "back", "kind", "type", "need", "needs", "want", "wants",
    "know", "knows", "look", "looks", "show", "shows", "shown", "seen",
    "tell", "tells", "told", "call", "calls", "called", "help", "helps",
    "keep", "kept", "turn", "turns", "move", "moves", "work", "works",
    "worked", "working", "went", "gone", "took", "taken", "give", "gives",
    "given", "way", "ways", "lot", "lots",
]

# CTI-терміни, які НІ В ЯКОМУ РАЗІ не мають потрапити в стоп-лист.
CTI_TERMS = [
    "ransomware", "malware", "phishing", "exploit", "exploits", "payload",
    "botnet", "trojan", "backdoor", "credential", "credentials",
    "exfiltration", "vulnerability", "vulnerabilities", "breach", "attack",
    "attacks", "campaign", "threat", "threats", "actor", "actors", "victim",
    "encrypt", "decrypt", "spyware", "keylogger", "dropper", "loader",
    "stealer", "beacon", "spearphishing", "worm", "rootkit",
]


def test_generic_noise_words_are_filtered_out():
    """Реальний сміттєвий кейс: found/files/video/anomalies тощо не мають
    попадати у витягнуті терміни навіть у "чужому" контексті."""
    text = (
        "Researchers found strange video files showing anomalies that remain "
        "mysterious and unexplained near the reactor shadow."
    )
    terms = set(extract_terms(text))
    for word in GENERIC_NOISE_WORDS:
        assert word not in terms, f"generic noise '{word}' не відфільтровано"


def test_extra_generic_noise_words_are_filtered_out():
    """Розширений клас generic-шуму (частини мови-звʼязки/побутова лексика)
    теж має бути відфільтрований."""
    text = " ".join(EXTRA_GENERIC_NOISE_WORDS)
    terms = set(extract_terms(text))
    for word in EXTRA_GENERIC_NOISE_WORDS:
        assert word not in terms, f"generic noise '{word}' не відфільтровано"


def test_cti_terms_survive_expanded_stopwords():
    """Розширення стоп-листа НЕ має зачіпати реальні CTI-терміни — вони
    мають виживати й далі витягуватись."""
    text = (
        "The ransomware campaign used malware and a phishing lure to deliver "
        "an exploit and several exploits as payload via a botnet trojan "
        "backdoor, stealing a credential and several credentials "
        "through exfiltration of data tied to a known vulnerability and "
        "vulnerabilities that enabled the breach; the attack and follow-on "
        "attacks were part of a broader campaign against the threat and "
        "threats faced by the actor and actors of the victim. "
        "The loader would decrypt and encrypt files using spyware, a keylogger, "
        "a dropper, a stealer and a beacon, spread via spearphishing worm "
        "rootkit techniques."
    )
    terms = set(extract_terms(text))
    for word in CTI_TERMS:
        assert word in terms, f"CTI-термін '{word}' помилково відфільтровано"


if __name__ == "__main__":
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures.append(name)
                print(f"FAIL {name}: {e}")
    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        sys.exit(1)
    print("\nAll tests passed.")
