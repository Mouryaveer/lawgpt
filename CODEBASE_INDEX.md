# LawGPT backend index

The production backend is an evidence-first RAG service.

## Runtime path

`app.py` exposes `/api/query`, `/api/health`, and `/api/queue-status`. `service.py` orchestrates deterministic query understanding, bounded expansion, Qwen3 embeddings, Pinecone, SQLite FTS5 BM25, reciprocal-rank fusion, Cohere reranking, authority-aware evidence packs, Groq Qwen3-32B generation, and claim-level citation verification.

## Configuration

`config/settings.py` is the single configuration source. Production defaults are:

- generation model: `qwen/qwen3-32b`
- embedding model: `Qwen/Qwen3-Embedding-4B`
- Pinecone index: `lawgpt-qwen3-prod`
- namespace: `qwen3-embedding-4b-v1`
- BM25: SQLite FTS5 persisted index
- reranker: Cohere `rerank-v3.5`

The service fails closed when required RAG configuration is absent. There is no llm-only fallback.

## Data path

`data/normalize.py` canonicalizes existing records and preserves legal hierarchy/source metadata. `data/ingest.py` provides `build-bm25`, `create-index`, and `upload` commands. Existing legal source files and the old Pinecone index are not overwritten.

## Verification

Run `python -m compileall -q .`, `python -m pytest -q`, and `python -m data.ingest build-bm25`. External model/provider and Pinecone checks require production credentials and the new versioned index.
