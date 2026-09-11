"""Production RAG orchestration. Infrastructure failures are explicit and fail closed."""

from __future__ import annotations

import logging
import time
import uuid

from config import Settings
from citations.verify import verify_answer
from evidence.pack import build_evidence_pack
from generation.groq import GroqGenerator
from query.expansion import expand_query
from query.understanding import analyze_query
from retrieval.bm25 import BM25Index
from retrieval.embeddings import DeepInfraEmbeddingAdapter
from retrieval.hybrid import HybridRetriever
from retrieval.pinecone import PineconeRetriever
from retrieval.reranker import CohereReranker

logger = logging.getLogger(__name__)


class RAGError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._retriever = None
        self._reranker = None
        self._generator = None
        self._bm25 = None

    def diagnostics(self) -> dict:
        result = self.settings.public_diagnostics()
        result["runtime_initialized"] = self._retriever is not None
        return result

    def _initialize(self) -> None:
        if self._retriever is not None:
            return
        missing = self.settings.missing_runtime_secrets()
        if missing:
            raise RAGError("configuration_error", "Required production RAG configuration is missing")
        if self.settings.rag_mode != "rag":
            raise RAGError("configuration_error", "RAG_MODE must be 'rag' in production")
        try:
            embeddings = DeepInfraEmbeddingAdapter(self.settings.embedding_model, self.settings.deepinfra_token, self.settings.embedding_base_url, self.settings.retrieval_timeout)
            pinecone = PineconeRetriever(self.settings.pinecone_api_key, self.settings.index_name, self.settings.namespace)
            if self.settings.bm25_enabled:
                self._bm25 = BM25Index(self.settings.bm25_database_path)
            self._retriever = HybridRetriever(embeddings, pinecone, self._bm25, self.settings.retrieval_candidate_k)
            self._reranker = CohereReranker(self.settings.cohere_api_key, self.settings.reranker_model, self.settings.reranker_url, self.settings.retrieval_timeout)
            self._generator = GroqGenerator(self.settings.groq_api_key, self.settings.generation_model, self.settings.groq_base_url, self.settings.generation_timeout)
        except RAGError:
            raise
        except Exception as exc:
            logger.exception("RAG initialization failed")
            raise RAGError("configuration_error", "Production RAG initialization failed") from exc

    def query(self, question: str) -> dict:
        if not question or not question.strip():
            raise RAGError("validation_error", "Query must not be empty")
        if len(question) > self.settings.max_query_length:
            raise RAGError("validation_error", "Query exceeds the maximum length")
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        self._initialize()
        analysis = analyze_query(question)
        expanded = expand_query(analysis)
        try:
            candidates = self._retriever.retrieve(expanded, analysis.document_types)
        except Exception as exc:
            logger.exception("Retrieval failed request_id=%s", request_id)
            raise RAGError("retrieval_error", "Verified legal retrieval is unavailable") from exc
        if not candidates:
            raise RAGError("insufficient_evidence", "No verified legal evidence matched this query")
        try:
            reranked = self._reranker.rerank(question, [item.as_dict() for item in candidates], self.settings.reranker_top_k)
        except Exception as exc:
            logger.exception("Reranking failed request_id=%s", request_id)
            raise RAGError("reranker_error", "Evidence reranking is unavailable") from exc
        if not reranked:
            raise RAGError("insufficient_evidence", "No verified legal evidence remained after reranking")
        evidence = build_evidence_pack(reranked, self.settings.reranker_top_k)
        try:
            draft = self._generator.generate(question, evidence)
        except Exception as exc:
            logger.exception("Generation failed request_id=%s", request_id)
            raise RAGError("generation_error", "Grounded answer generation is unavailable") from exc
        verified = verify_answer(draft, evidence) if self.settings.citation_verification_enabled else {**draft, "grounded": True, "unsupported_claim_count": 0, "insufficient_evidence": False}
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("rag_request request_id=%s total_latency_ms=%.1f semantic_candidates=%s lexical_enabled=%s reranked=%s citations=%s unsupported_claims=%s model=%s embedding_model=%s index=%s namespace=%s", request_id, elapsed_ms, len(candidates), self.settings.bm25_enabled, len(evidence), len(verified.get("claims", [])), verified.get("unsupported_claim_count", 0), self.settings.generation_model, self.settings.embedding_model, self.settings.index_name, self.settings.namespace)
        return {"response": verified["answer"], "model_used": self.settings.generation_model, "status": "insufficient_evidence" if verified.get("insufficient_evidence") else "grounded", "grounded": verified.get("grounded", False), "insufficient_evidence": verified.get("insufficient_evidence", False), "confidence": verified.get("confidence", "low"), "claims": verified.get("claims", []), "sources": evidence, "retrieval": {"semantic_candidates": len(candidates), "lexical_candidates": len(candidates) if self.settings.bm25_enabled else 0, "reranked": len(evidence)}, "request_id": request_id}
