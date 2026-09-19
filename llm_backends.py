"""Незалежні LLM-бекенди для агентів ONYX.

Раніше ВСІ ШІ-агенти ходили в одну й ту саму модель (глобальні LLM_BASE_URL /
LLM_API_KEY / LLM_MODEL з ui.py) — заявлена «незалежність» багатоагентного
аналізу зводилась лише до різних системних промптів. Цей модуль дозволяє
кожному агенту вказати ВЛАСНИЙ бекенд (провайдера + модель), щоб оцінки різних
агентів реально йшли з різних моделей різних провайдерів.

Env читаємо ПРИ КОЖНОМУ виклику (не на імпорті модуля) — щоб тести могли
підміняти оточення, а Streamlit бачив зміну без рестарту процесу.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    """Опис одного LLM-бекенду: куди стукати (OpenAI-сумісний chat/completions),
    яким ключем і якою моделлю."""
    name: str
    base_url: str
    api_key: str
    model: str
    label: str


def available_backends() -> dict:
    """Будує словник доступних бекендів з оточення.

    Бекенд потрапляє у словник ЛИШЕ якщо його обовʼязкові env задані (щоб у
    селекторі агента не світились провайдери, яких насправді немає чим
    викликати). "default" — виняток, він присутній ЗАВЖДИ (поточна поведінка
    ONYX — хмарний OpenRouter/Hermes 3 за замовчуванням)."""
    backends = {}

    # "default" — те саме, що й раніше було зашито у ui.py напряму.
    default_base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    default_model = os.environ.get("LLM_MODEL", "nousresearch/hermes-3-llama-3.1-405b:free")
    default_api_key = os.environ.get("LLM_API_KEY", "")
    backends["default"] = Backend(
        name="default", base_url=default_base_url, api_key=default_api_key,
        model=default_model, label=f"За замовчуванням · {default_model}",
    )

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        backends["groq"] = Backend(
            name="groq",
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=groq_key, model=groq_model,
            label=f"Groq · {groq_model}",
        )

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        openrouter_model = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")
        backends["openrouter"] = Backend(
            name="openrouter",
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=openrouter_key, model=openrouter_model,
            label=f"OpenRouter · {openrouter_model}",
        )

    ollama_model = os.environ.get("OLLAMA_LLM_MODEL")
    if ollama_model:
        backends["ollama"] = Backend(
            name="ollama",
            base_url=os.environ.get("OLLAMA_LLM_URL", "http://127.0.0.1:11434/v1"),
            api_key="",  # локальна Ollama ключа не потребує
            model=ollama_model,
            label=f"Ollama (локально) · {ollama_model}",
        )

    return backends


def resolve_backend(name) -> Backend:
    """Повертає названий бекенд; якщо його немає серед доступних (напр. агент
    обрав groq, а GROQ_API_KEY прибрали з оточення) — тихо падає назад на
    "default", щоб UI не ламався через зміну оточення."""
    backends = available_backends()
    if name and name in backends:
        return backends[name]
    return backends["default"]
