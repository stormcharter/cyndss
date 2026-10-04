"""Локальний стенд методу «двостороння семантична мережа: від проблеми до розв'язку».

НАВІЩО ОКРЕМА СТОРІНКА. `ui.py` вимагає логіну через Mongo і живого Manticore —
локально без Docker він не стартує. Стенд показує той самий екран
(`solution_ui.render_solution_mode`), що й режим «🧭 Від проблеми до
розв'язку» в ONYX, але без жодної інфраструктури: потрібна лише OpenAI-сумісна
LLM і ембедер (LM Studio, Ollama). Корпус для RAG — необов'язковий файл.

Запуск (PowerShell, з кореня репо):
    .venv\\Scripts\\streamlit run scenario_lab.py
LLM за замовчуванням — LM Studio на http://127.0.0.1:1234/v1; змінити можна
змінними LAB_LLM_BASE_URL / LAB_LLM_MODEL / LAB_EMBED_MODEL / LAB_LLM_API_KEY
або в бічній панелі.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import streamlit as st

from semantic_bridge import cosine
from solution_ui import SolutionEnv, render_solution_mode

ROOT = Path(__file__).resolve().parent
# Набір експерименту в репо; локальна тека data/networks — запасний варіант.
NETWORKS_DIR = next((d for d in (ROOT / "experiments" / "corporate_networks" / "dataset",
                                 ROOT / "data" / "networks") if d.exists()),
                    ROOT / "data" / "networks")

st.set_page_config(page_title="ONYX · Від проблеми до розв'язку", layout="wide")


def make_generate(base_url: str, model: str, api_key: str, no_reasoning: bool = True,
                  temperature: float | None = None):
    """Виклик моделі у формі, яку чекають модулі методу.

    `no_reasoning`: «думаючі» моделі (gemma-4, qwen3) інакше витрачають увесь
    ліміт на reasoning_content і повертають порожній content."""
    def generate(system: str, user: str, tag: str) -> str:
        # Розкриття фронтів — з різноманіттям; перевірка мостів і оцінки — без
        # нього, щоб той самий граф давав ті самі мости й бали.
        temp = temperature if temperature is not None else (
            0.0 if tag in {"bridges", "score"} else 0.3)
        payload = {"model": model, "temperature": temp,
                   "max_tokens": 1200 if tag == "synth" else 900, "stream": False,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}]}
        if no_reasoning:
            payload["reasoning_effort"] = "none"
        response = httpx.post(f"{base_url.rstrip('/')}/chat/completions",
                              headers={"Authorization": f"Bearer {api_key or 'local'}"},
                              json=payload, timeout=httpx.Timeout(600.0, connect=5.0))
        response.raise_for_status()
        choices = response.json().get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""
    return generate


def make_embed(base_url: str, model: str, api_key: str):
    def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(f"{base_url.rstrip('/')}/embeddings",
                              headers={"Authorization": f"Bearer {api_key or 'local'}"},
                              json={"model": model, "input": texts},
                              timeout=httpx.Timeout(120.0, connect=5.0))
        response.raise_for_status()
        data = sorted(response.json().get("data", []), key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]
    return embed


def llm_status(base_url: str) -> str:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=3.0)
        response.raise_for_status()
        names = [m.get("id") for m in response.json().get("data", [])]
        return "✅ доступно: " + ", ".join(n for n in names if n)[:300]
    except Exception as exc:  # noqa: BLE001 — показуємо будь-яку причину як є
        return f"❌ недоступно: {exc}"


def make_retrieve(documents: list[dict], embed):
    """RAG по завантаженому корпусу: два найближчі за змістом документи до поняття."""
    if not documents:
        return None
    texts = [" ".join(str(d.get(f) or "") for f in ("title", "summary", "content"))[:1500]
             for d in documents]
    vectors = embed([f"search_document: {t}" for t in texts])

    def retrieve(query: str) -> list[str]:
        q = embed([f"search_query: {query}"])[0]
        best = sorted(range(len(texts)), key=lambda i: -cosine(q, vectors[i]))[:2]
        return [texts[i][:500] for i in best]
    return retrieve


with st.sidebar:
    st.header("⚙️ Модель ШІ")
    base_url = st.text_input("Base URL (OpenAI-сумісний)",
                             os.getenv("LAB_LLM_BASE_URL", "http://127.0.0.1:1234/v1"))
    model_name = st.text_input("Модель", os.getenv("LAB_LLM_MODEL", "google/gemma-4-12b-qat"))
    embed_model = st.text_input("Ембедер",
                                os.getenv("LAB_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"))
    api_key = st.text_input("API key (для локальної не потрібен)",
                            os.getenv("LAB_LLM_API_KEY", ""), type="password")
    no_reasoning = st.checkbox("Вимкнути міркування моделі (reasoning_effort=none)", value=True)
    if st.button("Перевірити з'єднання"):
        st.caption(llm_status(base_url))
    extra = {}
    for n in (2, 3):
        with st.expander(f"Додаткова модель {n} (для ШІ-експертів)"):
            url = st.text_input("Base URL", key=f"extra_url_{n}")
            model = st.text_input("Модель", key=f"extra_model_{n}")
            key = st.text_input("API key", type="password", key=f"extra_key_{n}")
            reasoning_off = st.checkbox("reasoning_effort=none", key=f"extra_nr_{n}")
            if url and model:
                extra[f"extra{n}"] = (url, model, key, reasoning_off)
    st.divider()
    st.header("📚 Корпус (RAG), необов'язково")
    corpus_file = st.file_uploader("JSON/JSONL документів (title, summary, content)",
                                   type=["json", "jsonl"])

documents: list[dict] = []
if corpus_file is not None:
    raw = corpus_file.read().decode("utf-8", "replace").strip()
    try:
        documents = json.loads(raw) if raw.startswith("[") else \
            [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        st.sidebar.error(f"Корпус не прочитано: {exc}")

backends = {"main": f"Основна · {model_name}"}
backends.update({name: f"{model} · {url}" for name, (url, model, _k, _r) in extra.items()})


def generate_for(backend: str | None, temperature: float | None):
    if backend in extra:
        url, model, key, reasoning_off = extra[backend]
        return make_generate(url, model, key, reasoning_off, temperature)
    return make_generate(base_url, model_name, api_key, no_reasoning, temperature)


embed = make_embed(base_url, embed_model, api_key)
render_solution_mode(SolutionEnv(
    make_generate=generate_for,
    backends=backends,
    embed=embed,
    networks_dir=NETWORKS_DIR,
    data_dir=ROOT / "data",
    model_tag=model_name,
    retrieve=make_retrieve(documents, embed) if documents else None,
    retrieve_label="завантажений файл" if documents else "",
))
