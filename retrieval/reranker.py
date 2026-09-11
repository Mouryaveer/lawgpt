"""Configurable query-passage reranking adapter."""

from __future__ import annotations

import requests


class RerankerError(RuntimeError):
    pass


class CohereReranker:
    def __init__(self, token: str, model: str, url: str, timeout: float = 15.0):
        if not token:
            raise RerankerError("COHERE_API_KEY is not configured")
        self.token = token
        self.model = model
        self.url = url
        self.timeout = timeout

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        try:
            response = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
                json={"model": self.model, "query": query, "documents": [str(c.get("text", "")) for c in candidates], "top_n": min(top_k, len(candidates)), "return_documents": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            ranked = []
            for result in results:
                index = int(result.get("index", -1))
                if 0 <= index < len(candidates):
                    item = dict(candidates[index])
                    item["rerank_score"] = float(result.get("relevance_score", 0.0))
                    ranked.append(item)
            return ranked[:top_k]
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            raise RerankerError("Reranker request failed") from exc
