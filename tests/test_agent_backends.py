"""Тести для llm_backends.py — незалежні бекенди (base_url/ключ/модель) для
різних ШІ-агентів ONYX, щоб «незалежність» агентів була реальною, а не лише
різними системними промптами.

Жодних мережевих викликів — перевіряємо тільки формування Backend/URL/
заголовків/моделі з оточення.

Тести йдуть і під pytest, і напряму (`python tests/test_agent_backends.py`) —
зразок такого запуску: tests/test_shipper_config.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_backends import Backend, available_backends, resolve_backend  # noqa: E402


# Повний список env-змінних, які чіпають бекенди — щоб чистити оточення між тестами
# і не залежати від того, що реально стоїть на машині розробника.
_BACKEND_ENV_VARS = [
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
    "GROQ_API_KEY", "GROQ_BASE_URL", "GROQ_MODEL",
    "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_MODEL",
    "OLLAMA_LLM_MODEL", "OLLAMA_LLM_URL",
]


def _clean_env(**overrides):
    """Повертає контекстний менеджер: очищає всі relevant env-змінні, ставить overrides,
    після виходу — відновлює початковий стан оточення."""

    class _Ctx:
        def __enter__(self):
            self._saved = {k: os.environ.get(k) for k in _BACKEND_ENV_VARS}
            for k in _BACKEND_ENV_VARS:
                os.environ.pop(k, None)
            for k, v in overrides.items():
                os.environ[k] = v
            return self

        def __exit__(self, *exc):
            for k, v in self._saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _Ctx()


def test_default_backend_always_present_with_defaults():
    """Без жодного env — "default" присутній із дефолтами OpenRouter/Hermes (поточна поведінка)."""
    with _clean_env():
        backends = available_backends()
        assert "default" in backends
        b = backends["default"]
        assert isinstance(b, Backend)
        assert b.base_url == "https://openrouter.ai/api/v1"
        assert b.model == "nousresearch/hermes-3-llama-3.1-405b:free"
        assert b.api_key == ""


def test_default_backend_reads_llm_env():
    """"default" бере значення з LLM_BASE_URL/LLM_API_KEY/LLM_MODEL, якщо задані."""
    with _clean_env(LLM_BASE_URL="http://127.0.0.1:11434/v1",
                     LLM_API_KEY="sk-local",
                     LLM_MODEL="hermes3"):
        b = available_backends()["default"]
        assert b.base_url == "http://127.0.0.1:11434/v1"
        assert b.api_key == "sk-local"
        assert b.model == "hermes3"


def test_groq_backend_absent_without_key():
    """Без GROQ_API_KEY бекенд groq НЕ зʼявляється у словнику доступних."""
    with _clean_env():
        assert "groq" not in available_backends()


def test_groq_backend_present_with_key_and_defaults():
    with _clean_env(GROQ_API_KEY="gsk-test"):
        backends = available_backends()
        assert "groq" in backends
        b = backends["groq"]
        assert b.api_key == "gsk-test"
        assert b.base_url == "https://api.groq.com/openai/v1"
        assert b.model == "openai/gpt-oss-120b"


def test_groq_backend_respects_overrides():
    with _clean_env(GROQ_API_KEY="gsk-test",
                     GROQ_BASE_URL="https://groq.example.com/v1",
                     GROQ_MODEL="mixtral-8x7b"):
        b = available_backends()["groq"]
        assert b.base_url == "https://groq.example.com/v1"
        assert b.model == "mixtral-8x7b"


def test_openrouter_backend_absent_without_key():
    """Окремий бекенд "openrouter" (не плутати з "default") — вимагає власного ключа."""
    with _clean_env():
        assert "openrouter" not in available_backends()


def test_openrouter_backend_present_with_key_and_defaults():
    with _clean_env(OPENROUTER_API_KEY="sk-or-test"):
        backends = available_backends()
        assert "openrouter" in backends
        b = backends["openrouter"]
        assert b.api_key == "sk-or-test"
        assert b.base_url == "https://openrouter.ai/api/v1"
        assert b.model == "deepseek/deepseek-chat-v3-0324:free"


def test_ollama_backend_absent_without_model():
    """Без OLLAMA_LLM_MODEL локальний бекенд не показуємо (не всі мають Ollama з LLM)."""
    with _clean_env():
        assert "ollama" not in available_backends()


def test_ollama_backend_present_with_model_and_no_key_required():
    with _clean_env(OLLAMA_LLM_MODEL="hermes3"):
        backends = available_backends()
        assert "ollama" in backends
        b = backends["ollama"]
        assert b.model == "hermes3"
        assert b.base_url == "http://127.0.0.1:11434/v1"
        assert b.api_key == ""  # ключ для локальної Ollama не потрібен


def test_resolve_backend_returns_named_backend_when_available():
    with _clean_env(GROQ_API_KEY="gsk-test"):
        b = resolve_backend("groq")
        assert b.name == "groq"
        assert b.api_key == "gsk-test"


def test_resolve_backend_falls_back_to_default_when_missing():
    """Обраний агентом бекенд зник з оточення (наприклад, прибрали ключ) —
    UI не повинен падати, тихо повертаємось на "default"."""
    with _clean_env():
        b = resolve_backend("groq")
        assert b.name == "default"


def test_resolve_backend_none_returns_default():
    with _clean_env():
        b = resolve_backend(None)
        assert b.name == "default"


def test_backend_is_frozen_dataclass():
    """Backend — незмінний dataclass, щоб конфіг агента не можна було випадково
    підмінити після резолву."""
    with _clean_env():
        b = resolve_backend(None)
        try:
            b.model = "other"
        except Exception:
            pass
        else:
            raise AssertionError("Backend мав бути frozen — присвоєння мало кинути виняток")


def test_backend_label_is_human_readable():
    """label використовується у st.selectbox — має містити провайдера й модель."""
    with _clean_env(GROQ_API_KEY="gsk-test"):
        b = available_backends()["groq"]
        assert "openai/gpt-oss-120b" in b.label


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
