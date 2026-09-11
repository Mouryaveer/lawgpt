"""Hybrid semantic + lexical retrieval with deterministic reciprocal-rank fusion."""

from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25Index


@dataclass
class RetrievalCandidate:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0

    def as_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "metadata": self.metadata, "semantic_score": self.semantic_score, "lexical_score": self.lexical_score, "fusion_score": self.fusion_score, "rerank_score": self.rerank_score}


def _metadata_filter(document_types: tuple[str, ...]) -> dict | None:
    if len(document_types) == 1 and document_types[0] in {"act", "constitution", "case"}:
        return {"document_type": {"$eq": document_types[0]}}
    return None


class HybridRetriever:
    def __init__(self, embeddings, pinecone, bm25: BM25Index | None, candidate_k: int = 50):
        self.embeddings = embeddings
        self.pinecone = pinecone
        self.bm25 = bm25
        self.candidate_k = candidate_k

    def retrieve(self, queries: list[str], document_types: tuple[str, ...]) -> list[RetrievalCandidate]:
        merged: dict[str, RetrievalCandidate] = {}
        metadata_filter = _metadata_filter(document_types)
        rrf_k = 60.0
        for query in queries:
            semantic = self.pinecone.search(self.embeddings.embed_query(query), self.candidate_k, metadata_filter)
            lexical = self.bm25.search(query, self.candidate_k) if self.bm25 else []
            for rank, item in enumerate(semantic, start=1):
                candidate = merged.setdefault(item["id"], RetrievalCandidate(item["id"], item.get("text", ""), item.get("metadata", {})))
                candidate.semantic_score = max(candidate.semantic_score, float(item.get("score", 0.0)))
                candidate.fusion_score += 1.0 / (rrf_k + rank)
            for rank, item in enumerate(lexical, start=1):
                candidate = merged.setdefault(item["id"], RetrievalCandidate(item["id"], item.get("text", ""), item.get("metadata", {})))
                candidate.lexical_score = max(candidate.lexical_score, float(item.get("score", 0.0)))
                candidate.fusion_score += 1.0 / (rrf_k + rank)
        return sorted(merged.values(), key=lambda c: (c.fusion_score, c.semantic_score, c.lexical_score), reverse=True)[: self.candidate_k]
