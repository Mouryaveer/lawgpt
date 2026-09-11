"""Pinecone adapter for the versioned Qwen embedding index."""

from __future__ import annotations

from typing import Any


class PineconeError(RuntimeError):
    pass


class PineconeRetriever:
    def __init__(self, api_key: str, index_name: str, namespace: str):
        if not api_key:
            raise PineconeError("PINECONE_API_KEY is not configured")
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise PineconeError("The pinecone package is not installed") from exc
        self.index_name = index_name
        self.namespace = namespace
        try:
            self.client = Pinecone(api_key=api_key)
            self.index = self.client.Index(index_name)
        except Exception as exc:
            raise PineconeError("Pinecone initialization failed") from exc

    def search(self, vector: list[float], limit: int, metadata_filter: dict | None = None) -> list[dict]:
        kwargs: dict[str, Any] = {"vector": vector, "top_k": limit, "include_metadata": True, "namespace": self.namespace}
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        try:
            response = self.index.query(**kwargs)
            matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
            results = []
            for match in matches:
                metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {}) or {}
                record_id = match.get("id") if isinstance(match, dict) else getattr(match, "id", "")
                score = match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)
                results.append({"id": record_id, "text": metadata.get("text", ""), "metadata": metadata, "score": float(score or 0.0)})
            return results
        except Exception as exc:
            raise PineconeError("Pinecone query failed") from exc
