"""OpenAI-compatible Qwen embedding adapter; no local/E5 fallback is allowed."""

from __future__ import annotations

import time
from typing import Iterable

import requests


class EmbeddingError(RuntimeError):
    pass


class DeepInfraEmbeddingAdapter:
    def __init__(self, model: str, token: str, base_url: str, timeout: float = 20.0, batch_size: int = 32):
        if not token:
            raise EmbeddingError("DEEPINFRA_TOKEN is not configured")
        self.model = model
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.last_latency_ms = 0.0

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
                json={"model": self.model, "input": inputs, "encoding_format": "float"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in data]
            if len(vectors) != len(inputs) or not vectors or not all(isinstance(v, list) and v for v in vectors):
                raise EmbeddingError("Embedding provider returned an invalid vector payload")
            return vectors
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("Embedding provider request failed") from exc
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(values), self.batch_size):
            vectors.extend(self._embed(values[start:start + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
