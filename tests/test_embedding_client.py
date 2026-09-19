"""Офлайн-тести вибору провайдера ембедингів: Ollama або віддалений API."""

import asyncio

import numpy as np
import pytest

from shipper.config import settings
from shipper.dedup import EmbeddingClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


@pytest.fixture(autouse=True)
def restore_embedding_settings(monkeypatch):
    monkeypatch.setattr(settings, "embed_dims", 3)
    monkeypatch.setattr(settings, "ollama_url", "http://ollama:11434")
    monkeypatch.setattr(settings, "ollama_embed_model", "nomic-embed-text")
    monkeypatch.setattr(settings, "remote_embed_url", "https://provider.example/v1")
    monkeypatch.setattr(settings, "remote_embed_model", "qwen-embed")
    monkeypatch.setattr(settings, "remote_embed_api_key", "")


def test_ollama_embedding_request_and_normalization(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "ollama")
    client = FakeClient([{"embeddings": [[3.0, 4.0, 0.0]]}])

    vector = asyncio.run(EmbeddingClient(client=client).embed("hello"))

    assert client.calls[0][0] == "http://ollama:11434/api/embed"
    assert client.calls[0][1]["json"] == {
        "model": "nomic-embed-text",
        "input": "hello",
    }
    assert np.allclose(vector, [0.6, 0.8, 0.0])


def test_remote_embedding_request(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "remote")
    monkeypatch.setattr(settings, "remote_embed_api_key", "embed-secret")
    client = FakeClient([{"data": [{"embedding": [1.0, 0.0, 0.0]}]}])

    vector = asyncio.run(EmbeddingClient(client=client).embed("hello"))

    assert client.calls[0][0] == "https://provider.example/v1/embeddings"
    assert client.calls[0][1]["headers"]["Authorization"] == "Bearer embed-secret"
    assert client.calls[0][1]["json"]["dimensions"] == 3
    assert np.allclose(vector, [1.0, 0.0, 0.0])


def test_auto_falls_back_from_remote_to_ollama(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "auto")
    client = FakeClient([
        RuntimeError("remote unavailable"),
        {"embeddings": [[1.0, 0.0, 0.0]]},
    ])

    vector = asyncio.run(EmbeddingClient(client=client).embed("hello"))

    assert [call[0] for call in client.calls] == [
        "https://provider.example/v1/embeddings",
        "http://ollama:11434/api/embed",
    ]
    assert np.allclose(vector, [1.0, 0.0, 0.0])


def test_dimension_mismatch_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "ollama")
    client = FakeClient([{"embeddings": [[1.0, 2.0]]}])

    with pytest.raises(ValueError, match="expected 3, got 2"):
        asyncio.run(EmbeddingClient(client=client).embed("hello"))


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "embed_provider", "unknown")

    with pytest.raises(ValueError, match="EMBED_PROVIDER"):
        EmbeddingClient(client=FakeClient([]))
