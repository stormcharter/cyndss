"""
Семантична дедуплікація.

Простий підхід: тримаємо в RAM вікно з останніх N ембедингів з їхніми mongo_id і
текстом. Для кожного нового документа рахуємо косинусну схожість з усім вікном.
Якщо макс. схожість >= threshold — вважаємо дублікатом і зливаємо (беремо довший
текст як канонічний, решта йде в siblings).

Чому ковзне вікно, а не запит у Manticore: дублікати в кіберрозвідці майже
завжди приходять купкою в один час (одна CVE — п'ять каналів протягом години).
Довга пам'ять не потрібна, а перевірка в RAM займає мілісекунди.

Для продакшену з мільйонами документів це треба замінити на HNSW у самому
Manticore — але для MVP ковзного вікна достатньо, і це додатковий матеріал
для розділу про оптимізацію.
"""

from collections import deque
from dataclasses import dataclass

import httpx
import numpy as np
import structlog

from .config import settings

logger = structlog.get_logger(__name__)


@dataclass
class DedupRecord:
    """Запис у вікні дедуплікації."""
    mongo_id: str
    text: str
    embedding: np.ndarray  # shape (768,), нормалізований


class EmbeddingClient:
    """Клієнт ембедингів із вибором провайдера: Ollama або віддалений OpenAI-сумісний API."""

    _VALID_PROVIDERS = {"ollama", "remote", "auto"}

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=60.0)
        self.provider = settings.embed_provider.strip().lower()
        if self.provider not in self._VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported EMBED_PROVIDER={settings.embed_provider!r}; "
                "expected ollama, remote, or auto"
            )

    async def _embed_ollama(self, text: str) -> list[float]:
        response = await self.client.post(
            f"{settings.ollama_url.rstrip('/')}/api/embed",
            json={"model": settings.ollama_embed_model, "input": text[:8000]},
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]

    async def _embed_remote(self, text: str) -> list[float]:
        if not settings.remote_embed_url:
            raise ValueError("REMOTE_EMBED_URL is required for the remote provider")
        headers = {}
        if settings.remote_embed_api_key and settings.remote_embed_api_key not in {
            "unused",
            "not-required",
        }:
            headers["Authorization"] = f"Bearer {settings.remote_embed_api_key}"
        response = await self.client.post(
            f"{settings.remote_embed_url.rstrip('/')}/embeddings",
            headers=headers,
            json={
                "model": settings.remote_embed_model,
                "input": text[:8000],
                "dimensions": settings.embed_dims,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    async def embed(self, text: str) -> np.ndarray:
        """Повертає L2-нормалізований вектор налаштованої розмірності."""
        providers = [self.provider] if self.provider != "auto" else ["remote", "ollama"]
        errors: list[str] = []
        values = None
        for provider in providers:
            try:
                values = await (
                    self._embed_ollama(text) if provider == "ollama" else self._embed_remote(text)
                )
                break
            except Exception as exc:
                errors.append(f"{provider}: {exc}")
                logger.warning("embedding_provider_failed", provider=provider, error=str(exc))
        if values is None:
            raise RuntimeError("All embedding providers failed: " + "; ".join(errors))

        vec = np.asarray(values, dtype=np.float32)
        if vec.ndim != 1 or vec.size != settings.embed_dims:
            raise ValueError(
                f"Embedding dimension mismatch: expected {settings.embed_dims}, got {vec.size}"
            )
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    async def aclose(self):
        if self._owns_client:
            await self.client.aclose()


@dataclass
class DedupResult:
    """Результат перевірки на дублікат."""
    is_duplicate: bool
    similarity: float = 0.0
    duplicate_of_mongo_id: str | None = None


class SemanticDeduplicator:
    """Ковзне вікно, у якому лише надійно проіндексовані канонічні документи."""

    def __init__(self):
        self.window: deque[DedupRecord] = deque(maxlen=settings.dedup_window_size)
        self.threshold = settings.dedup_similarity_threshold

    def find_duplicate(self, embedding: np.ndarray) -> DedupResult:
        """Повертає збіг із надійно записаним канонічним документом, не змінюючи вікно."""
        if not self.window:
            return DedupResult(is_duplicate=False)

        matrix = np.stack([record.embedding for record in self.window])
        similarities = matrix @ embedding
        max_index = int(np.argmax(similarities))
        max_similarity = float(similarities[max_index])

        if max_similarity >= self.threshold:
            canonical = self.window[max_index]
            return DedupResult(
                is_duplicate=True,
                similarity=max_similarity,
                duplicate_of_mongo_id=canonical.mongo_id,
            )
        return DedupResult(is_duplicate=False, similarity=max_similarity)

    def remember_indexed(
        self,
        mongo_id: str,
        text: str,
        embedding: np.ndarray,
    ) -> None:
        """Запам'ятовує документ лише після того, як сінк підтвердив його запис."""
        self.window.append(
            DedupRecord(mongo_id=mongo_id, text=text, embedding=embedding)
        )
