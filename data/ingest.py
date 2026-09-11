"""Resumable ingestion CLI for the versioned Qwen legal corpus.

Examples:
  python -m data.ingest build-bm25
  python -m data.ingest create-index
  python -m data.ingest upload
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config import get_settings
from data.normalize import normalize_record
from retrieval.bm25 import BM25Index
from retrieval.embeddings import DeepInfraEmbeddingAdapter

logger = logging.getLogger(__name__)


def iter_records(path: str):
    with open(path, "r", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if line.strip():
                yield normalize_record(json.loads(line), index)


def build_bm25(settings) -> int:
    return BM25Index.build(settings.corpus_path, settings.bm25_database_path)


def create_index(settings) -> dict:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required to create the versioned index")
    embeddings = DeepInfraEmbeddingAdapter(settings.embedding_model, settings.deepinfra_token, settings.embedding_base_url)
    vector = embeddings.embed_query("Indian legal research index dimension probe")
    dimension = len(vector)
    try:
        from pinecone import Pinecone, ServerlessSpec
        client = Pinecone(api_key=settings.pinecone_api_key)
        existing = {item["name"] if isinstance(item, dict) else item.name for item in client.list_indexes()}
        if settings.index_name not in existing:
            client.create_index(name=settings.index_name, dimension=dimension, metric=settings.vector_metric, spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region))
        return {"index": settings.index_name, "namespace": settings.namespace, "dimension": dimension, "created": settings.index_name not in existing}
    except ImportError as exc:
        raise RuntimeError("The pinecone package is required for index creation") from exc


def upload(settings, batch_size: int = 64) -> int:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required to upload the versioned corpus")
    from pinecone import Pinecone
    embeddings = DeepInfraEmbeddingAdapter(settings.embedding_model, settings.deepinfra_token, settings.embedding_base_url)
    index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.index_name)
    batch_ids, batch_texts, batch_meta = [], [], []
    uploaded = 0
    for record in iter_records(settings.corpus_path):
        batch_ids.append(record["id"])
        batch_texts.append(record["text"])
        batch_meta.append({**record["metadata"], "text": record["text"]})
        if len(batch_ids) >= batch_size:
            vectors = embeddings.embed_documents(batch_texts)
            index.upsert(vectors=[{"id": rid, "values": vector, "metadata": meta} for rid, vector, meta in zip(batch_ids, vectors, batch_meta)], namespace=settings.namespace)
            uploaded += len(batch_ids)
            batch_ids, batch_texts, batch_meta = [], [], []
            if uploaded % 1000 == 0:
                logger.info("Uploaded vectors=%s", uploaded)
    if batch_ids:
        vectors = embeddings.embed_documents(batch_texts)
        index.upsert(vectors=[{"id": rid, "values": vector, "metadata": meta} for rid, vector, meta in zip(batch_ids, vectors, batch_meta)], namespace=settings.namespace)
        uploaded += len(batch_ids)
    return uploaded


def main():
    parser = argparse.ArgumentParser(description="LawGPT legal corpus ingestion")
    parser.add_argument("command", choices=["build-bm25", "create-index", "upload"])
    args = parser.parse_args()
    settings = get_settings()
    if args.command == "build-bm25":
        print(json.dumps({"records": build_bm25(settings), "database": settings.bm25_database_path}))
    elif args.command == "create-index":
        print(json.dumps(create_index(settings)))
    else:
        print(json.dumps({"uploaded": upload(settings), "index": settings.index_name, "namespace": settings.namespace}))


if __name__ == "__main__":
    main()
