"""Single source of truth for production LawGPT configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv()


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Required model and retrieval identities. These defaults are intentional.
    generation_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL_NAME", "qwen/qwen3-32b"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-4B"))
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "deepinfra"))
    reranker_provider: str = field(default_factory=lambda: os.getenv("RERANKER_PROVIDER", "cohere"))
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL_NAME", "rerank-v3.5"))

    index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "lawgpt-qwen3-prod"))
    namespace: str = field(default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "qwen3-embedding-4b-v1"))
    pinecone_cloud: str = field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"))
    vector_dimension: int = field(default_factory=lambda: _int("PINECONE_VECTOR_DIMENSION", 0))
    vector_metric: str = field(default_factory=lambda: os.getenv("PINECONE_METRIC", "cosine"))

    rag_mode: str = field(default_factory=lambda: os.getenv("RAG_MODE", "rag").lower())
    retrieval_candidate_k: int = field(default_factory=lambda: _int("RETRIEVAL_CANDIDATE_K", 50))
    reranker_top_k: int = field(default_factory=lambda: _int("RERANKER_TOP_K", 10))
    bm25_database_path: str = field(default_factory=lambda: os.getenv("BM25_DATABASE_PATH", str(ROOT / "storage" / "bm25_legal_v2.sqlite3")))
    corpus_path: str = field(default_factory=lambda: os.getenv("CORPUS_PATH", str(ROOT / "new_data_chunked_documents.jsonl")))
    bm25_enabled: bool = field(default_factory=lambda: _bool("BM25_ENABLED", True))
    reranker_required: bool = field(default_factory=lambda: _bool("RERANKER_REQUIRED", True))
    citation_verification_enabled: bool = field(default_factory=lambda: _bool("CITATION_VERIFICATION_ENABLED", True))

    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    deepinfra_token: str = field(default_factory=lambda: os.getenv("DEEPINFRA_TOKEN", ""))
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    cohere_api_key: str = field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    embedding_base_url: str = field(default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", "https://api.deepinfra.com/v1/openai"))
    reranker_url: str = field(default_factory=lambda: os.getenv("RERANKER_URL", "https://api.cohere.com/v2/rerank"))
    groq_base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))

    generation_timeout: float = field(default_factory=lambda: _float("GROQ_REQUEST_TIMEOUT_SECONDS", 45.0))
    retrieval_timeout: float = field(default_factory=lambda: _float("RETRIEVAL_TIMEOUT_SECONDS", 15.0))
    verification_timeout: float = field(default_factory=lambda: _float("VERIFICATION_TIMEOUT_SECONDS", 10.0))
    max_concurrent_requests: int = field(default_factory=lambda: _int("MAX_CONCURRENT_REQUESTS", 2))
    max_query_length: int = field(default_factory=lambda: _int("MAX_QUERY_LENGTH", 1000))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    version: str = field(default_factory=lambda: os.getenv("LAWGPT_VERSION", "rag-v2"))
    allowed_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "https://turn2law-tan.vercel.app").split(",") if origin.strip()
    ))

    def missing_runtime_secrets(self) -> list[str]:
        missing = []
        for name, value in (("GROQ_API_KEY", self.groq_api_key), ("DEEPINFRA_TOKEN", self.deepinfra_token), ("PINECONE_API_KEY", self.pinecone_api_key)):
            if not value:
                missing.append(name)
        if self.reranker_required and self.reranker_provider == "cohere" and not self.cohere_api_key:
            missing.append("COHERE_API_KEY")
        return missing

    def public_diagnostics(self) -> dict:
        missing = self.missing_runtime_secrets()
        return {
            "rag_enabled": self.rag_mode == "rag",
            "generation_model": self.generation_model,
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "vector_db": "pinecone",
            "bm25_enabled": self.bm25_enabled,
            "reranker_enabled": self.reranker_required,
            "reranker_model": self.reranker_model,
            "citation_verification_enabled": self.citation_verification_enabled,
            "index": self.index_name,
            "namespace": self.namespace,
            "version": self.version,
            "configuration_ready": not missing and self.rag_mode == "rag",
            "missing_configuration": missing,
        }


def get_settings() -> Settings:
    return Settings()
