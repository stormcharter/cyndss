"""Тести нормалізації collector_dedup.content_hash."""

from collector_dedup import content_hash


def test_basic_normalization() -> None:
    h1 = content_hash("Hello World")
    h2 = content_hash("  Hello   World  ")
    assert h1 == h2, "whitespace collapse mismatch"


def test_case_insensitive() -> None:
    h1 = content_hash("Critical CVE in Linux")
    h2 = content_hash("critical cve in linux")
    assert h1 == h2, "case normalization mismatch"


def test_empty_text() -> None:
    h = content_hash("")
    assert isinstance(h, str) and len(h) == 64


def test_none_coerced_to_empty() -> None:
    h = content_hash(None)
    assert isinstance(h, str) and len(h) == 64


def test_different_texts_differ() -> None:
    h1 = content_hash("exploit CVE-2024-3094 in xz")
    h2 = content_hash("new vulnerability in OpenSSL")
    assert h1 != h2, "different texts should produce different hashes"
