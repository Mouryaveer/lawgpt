# LawGPT / Turn2Law production RAG

LawGPT is an evidence-first Indian legal research API. The model is not the source of legal facts: retrieved corpus evidence is merged, reranked, packed, cited, and verified before a response is returned.

## Architecture

`POST /api/query` runs:

1. deterministic legal query understanding and bounded expansion;
2. Qwen3-Embedding-4B semantic search in a versioned Pinecone namespace;
3. persistent SQLite FTS5 BM25 lexical search for provisions, articles, rules, and case names;
4. reciprocal-rank fusion, deduplication, and configurable Cohere query-passage reranking;
5. authority-aware evidence packing and source metadata preservation;
6. Qwen3-32B grounded JSON generation through Groq;
7. claim-level citation verification and suppression of unsupported propositions.

If a required dependency is unavailable, the API returns an explicit unavailable/insufficient-evidence result. It never silently switches to an LLM-only answer, E5 vectors, or a different production model.

## Models and data version

- Generation: `qwen/qwen3-32b` through Groq (`GROQ_MODEL_NAME`).
- Embeddings: `Qwen/Qwen3-Embedding-4B` through the configured OpenAI-compatible embedding provider.
- Pinecone: `lawgpt-qwen3-prod`, namespace `qwen3-embedding-4b-v1` by default.
- Reranker: Cohere `rerank-v3.5` by default; the adapter is configurable.
- Corpus: existing `new_data_chunked_documents.jsonl`, normalized without deleting the source data.

The old Pinecone index is not overwritten. Create and validate the new index before switching the production service.

## Environment

Copy `.env.example` to `.env` and provide real values locally. Never commit `.env` or credentials. Required runtime values are `GROQ_API_KEY`, `DEEPINFRA_TOKEN` (or another compatible embedding provider configuration), `PINECONE_API_KEY`, and `COHERE_API_KEY` when the default reranker is enabled.

The complete production template is in `render.yaml`. `RAG_MODE=rag` is mandatory. Model, index, namespace, candidate counts, timeouts, and CORS are centralized in `config/settings.py`.

## Build the lexical index

```bash
python -m data.ingest build-bm25
```

The command creates a resumable-friendly SQLite FTS5 artifact at `BM25_DATABASE_PATH` and indexes normalized metadata alongside legal text. Generated SQLite files are ignored by Git.

## Create and upload the new Pinecone namespace

Use production credentials only after confirming the Qwen embedding provider/model is available:

```bash
python -m data.ingest create-index
python -m data.ingest upload
```

`create-index` probes the embedding dimension and creates the new index only if it does not exist. It never overwrites an existing index. Upload IDs are deterministic, so rerunning upserts the same records instead of duplicating them.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8002
```

Check:

```bash
curl http://localhost:8002/api/health
curl -X POST http://localhost:8002/api/query -H "Content-Type: application/json" -d '{"query":"What does Article 21 protect?"}'
```

`GET /api/health` exposes safe model/index/provider diagnostics and reports `not_ready` when required credentials or infrastructure are missing. Secrets are never returned.

## Tests

```bash
python -m compileall -q .
python -m pytest -q
```

The offline suite covers query understanding, bounded expansion, BM25 exact lookup, hybrid fusion/deduplication, citation verification, health diagnostics, and the no-LLM-only-fallback contract.

## Deployment

Render uses `render.yaml`, installs the small API/runtime dependency set, and starts `uvicorn app:app`. The BM25 SQLite artifact must be provisioned through the selected Render storage strategy; do not commit the generated index. Before production traffic, validate Pinecone dimension, namespace, vector count, metadata queries, sample retrieval, reranker access, and the exact Groq model availability.

The requested Qwen3-32B model is configured exactly and is not silently replaced if the provider has retired or restricted it. If the provider does not make that model available, update provider access or explicitly choose a new approved model and update the configuration/documentation together.

## Security

No raw model HTML is trusted by this backend. Public errors are generic, detailed diagnostics stay in server logs, CORS is limited to `ALLOWED_ORIGINS`, and tracked experimental credential files are excluded/removed. Rotate any Groq or Pinecone credentials that were previously exposed in repository history.
