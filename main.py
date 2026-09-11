"""Backward-compatible facade for callers that historically imported main.ragu."""

from __future__ import annotations

from config import get_settings
from service import RAGService

settings = get_settings()
service = RAGService(settings)
groq_model_name = settings.generation_model
embedding_model_name = settings.embedding_model
embedding_provider = settings.embedding_provider
index_name = settings.index_name
pinecone_namespace = settings.namespace


def ragu(query: str) -> str:
    return service.query(query)["response"]


def query_result(query: str) -> dict:
    return service.query(query)


def get_rag_chain():
    service._initialize()
    return service
