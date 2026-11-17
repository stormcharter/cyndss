"""Валідація Settings шиппера: fail-fast на кривих URL/лімітах + свідомий
опт-аут для plaintext-ключа всередині приватної мережі (docker-бридж).

Тести йдуть і під pytest, і напряму (`python tests/test_shipper_config.py`).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shipper.config import Settings


def _base(**over):
    """Мінімально валідний набір полів; .env НЕ читаємо, щоб тест не залежав
    від машини розробника."""
    kwargs = dict(
        _env_file=None,
        mongo_uri="mongodb://localhost:27017/?replicaSet=rs0",
        manticore_url="http://localhost:9308",
        ollama_url="http://localhost:11434",
        llm_base_url="http://localhost:11434/v1",
        llm_model="hermes3:8b",
    )
    kwargs.update(over)
    return kwargs


def test_plaintext_key_to_loopback_is_allowed():
    s = Settings(**_base(llm_api_key="sk-test"))
    assert s.llm_api_key == "sk-test"


def test_plaintext_key_to_remote_host_is_rejected():
    """Bearer по відкритому HTTP на неloopback-хост — це витік ключа."""
    try:
        Settings(**_base(llm_base_url="http://ollama:11434/v1", llm_api_key="sk-test"))
    except Exception as exc:
        assert "HTTPS" in str(exc), f"очікували скаргу на HTTPS, отримали: {exc}"
    else:
        raise AssertionError("plaintext Bearer на віддалений хост мав бути відхилений")


def test_plaintext_key_allowed_with_explicit_optout():
    """Docker-бридж (`http://ollama:11434/v1`) — не публічна мережа. Дозволяємо,
    але ЛИШЕ через явний прапорець, щоб це було свідомим рішенням, а не
    мовчазним послабленням перевірки."""
    s = Settings(**_base(
        llm_base_url="http://ollama:11434/v1",
        llm_api_key="sk-test",
        allow_plaintext_llm_key=True,
    ))
    assert s.llm_api_key == "sk-test"
    assert s.allow_plaintext_llm_key is True


def test_optout_does_not_leak_to_remote_embeddings():
    """Опт-аут стосується лише LLM-ендпойнта; віддалений ембеддер має лишитись
    під тим самим правилом."""
    try:
        Settings(**_base(
            remote_embed_url="http://embed.example.com",
            remote_embed_api_key="sk-embed",
            allow_plaintext_llm_key=True,
        ))
    except Exception as exc:
        assert "HTTPS" in str(exc), f"очікували скаргу на HTTPS, отримали: {exc}"
    else:
        raise AssertionError("опт-аут не мав поширюватись на REMOTE_EMBED_URL")


def test_bad_url_scheme_is_rejected():
    try:
        Settings(**_base(manticore_url="localhost:9308"))
    except Exception as exc:
        assert "MANTICORE_URL" in str(exc)
    else:
        raise AssertionError("відносний URL мав бути відхилений")


def test_default_llm_endpoint_stays_openrouter():
    """Дефолти не мають тихо переїжджати на локальну Ollama: у проєкті
    за замовчуванням LLM ходить у хмару (див. .env.example)."""
    s = Settings(_env_file=None)
    assert "openrouter.ai" in s.llm_base_url
    assert s.llm_model.startswith("nousresearch/")


def test_embeddings_stay_local_nomic_by_default():
    s = Settings(_env_file=None)
    assert s.embed_provider == "ollama"
    assert s.ollama_embed_model == "nomic-embed-text"
    assert s.embed_dims == 768


def test_semantic_dedup_is_off_by_default():
    """Виміряно на живому стеку (1486 подій): сирі вектори nomic давали cosine >= 0.92
    між РІЗНИМИ постами, і ~30% корпусу мовчки відкидалось як «дублікати». Вибірка
    із 10 відкинутих при 0.92-0.95 була 10/10 різними постами. Тому за
    замовчуванням семантичний дедуп вимкнено; точний дедуп (message_id +
    content_hash) працює на рівні БД незалежно від цього."""
    s = Settings(_env_file=None)
    assert s.semantic_dedup_enabled is False
    # Поріг лишається для випадку, коли дедуп вмикають свідомо.
    assert s.dedup_similarity_threshold == 0.92


def test_semantic_dedup_can_be_enabled_from_env():
    old = os.environ.get("SEMANTIC_DEDUP_ENABLED")
    os.environ["SEMANTIC_DEDUP_ENABLED"] = "true"
    try:
        assert Settings(_env_file=None).semantic_dedup_enabled is True
    finally:
        if old is None:
            del os.environ["SEMANTIC_DEDUP_ENABLED"]
        else:
            os.environ["SEMANTIC_DEDUP_ENABLED"] = old


def test_build_deduplicator_follows_the_switch():
    """`None` = дедуп вимкнено (Pipeline пропускає find_duplicate/remember_indexed)."""
    from shipper.config import settings
    from shipper.dedup import SemanticDeduplicator, build_deduplicator

    old = settings.semantic_dedup_enabled
    try:
        settings.semantic_dedup_enabled = False
        assert build_deduplicator() is None
        settings.semantic_dedup_enabled = True
        assert isinstance(build_deduplicator(), SemanticDeduplicator)
    finally:
        settings.semantic_dedup_enabled = old


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001 — самописний раннер
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failed")
    sys.exit(1 if failures else 0)
