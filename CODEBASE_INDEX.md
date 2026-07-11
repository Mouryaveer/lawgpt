# Codebase Index: Turn2Law-LawGPT

Last refreshed: 2026-04-23

## Repository purpose

Turn2Law-LawGPT is a legal Retrieval-Augmented Generation (RAG) system focused on Indian law documents. It includes:

- A scraper pipeline for India Code act PDFs.
- PDF extraction and cleanup utilities.
- Token-based chunk generation for embeddings.
- Pinecone-backed retrieval with Groq-powered response generation.
- A FastAPI service for frontend/API integration.

## End-to-end data flow

1. Scrape acts by year from India Code.
   - Script: `scraper3.py`
   - Input: `years_data.json`
   - Output: `scraped_acts/year_XXXX.json` (and `temp_years/` for empty years)

2. Extract and clean PDF text.
   - Utilities: `utils/pdf_utils.py`
   - Called from scraper while processing each PDF.

3. Chunk legal text for vector storage.
   - Script: `Chunk_maker.py`
   - Output: `chunked_documents.jsonl`

4. Build/query vector store.
   - Ingestion notebook: `embedding_to_vectorDB.ipynb`
   - Runtime retriever: `main.py`
   - Vector DB: Pinecone index (default `caus-legal-vdb`)

5. Serve answers over HTTP.
   - API server: `app.py`
   - Primary endpoint: `POST /api/query`

## Top-level source index

### Runtime and API

- `app.py`
  - FastAPI entrypoint.
  - Adds CORS middleware and queue/concurrency controls using `asyncio.Semaphore`.
  - Offloads blocking RAG call (`ragu`) to threadpool via `run_in_executor`.
  - Endpoints:
    - `GET|HEAD /`
    - `POST /api/query`
    - `GET|HEAD /api/health`
    - `GET /api/queue-status`
    - `GET|POST /query` (deprecated guidance endpoint)

- `main.py`
  - Assembles retrieval chain and LLM pipeline.
  - Loads environment variables, initializes Pinecone vector store.
  - Pulls prompt from LangChain Hub.
  - Exposes `ragu(query: str) -> str` with retry/backoff decorator.
  - Embedding strategy:
    - Preferred: DeepInfra API (`DEEPINFRA_TOKEN` present)
    - Fallback: local HuggingFace embeddings

- `deepinfra_embeddings.py`
  - Standalone LangChain-compatible DeepInfra embeddings class.
  - Not currently imported by `main.py` (which defines a local variant inline).

### Data acquisition and preparation

- `scraper3.py`
  - Playwright-based year and act scraper.
  - Downloads PDF bytes via browser session fetch.
  - Extracts text with `extract_text_from_pdf`.
  - Uses resumable per-year JSON writes.
  - Includes retry/backoff and periodic browser restart.

- `Chunk_maker.py`
  - Token-based document chunking using `tiktoken` (`cl100k_base`).
  - Preserves rich metadata (year, act, section, urls, source).
  - Writes one JSON object per line to `chunked_documents.jsonl`.

- `utils/pdf_utils.py`
  - PDF extraction using `pdfminer.six`.
  - Text cleanup optimized for embedding quality:
    - encoding normalization
    - reading-order repair
    - header/footer deduplication
    - hyphenation and spacing repair
    - structure-aware paragraph handling

- `utils/save_utils.py`
  - Atomic JSON writes via temp-file replacement.
  - Helpers for loading prior year data and existing scraped URLs.

### Frontend and local testing

- `frontend_example.html`
  - Minimal browser client for posting to `/api/query`.

- `test_FE.py`
  - Streamlit UI that calls `main.ragu` directly.

### Deployment and configuration

- `requirements.txt`
  - Core dependencies for scraping, embeddings, LangChain, Pinecone, FastAPI, and Streamlit.

- `render.yaml`
  - Render service configuration.
  - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
  - Python version pin via env var: `3.11.9`.

- `runtime.txt`
  - Runtime Python pin (used by some deployment targets).

- `README.md`
  - Setup and API usage guide.

## Data assets index

- `years_data.json`
  - Source list of target years and browse URLs.

- `scraped_acts/year_*.json`
  - Scraped per-year act payloads.
  - Current count: 163 files.

- `temp_years/year_*.json`
  - Years with empty/deferred outputs.
  - Current count: 5 files.

- `chunked_documents.jsonl`
  - Final embedding-ready corpus with `id`, `text`, and `metadata`.

- `new_data/acts_sections (2).json`
- `new_data/constitution_parts_enriched.json`
- `new_data/final_with_content.json`
  - Additional curated or transformed legal datasets.

## API contract summary

- `POST /api/query`
  - Request:
    - `{ "query": "...", "model": null }`
  - Response:
    - `{ "response": "...", "model_used": "openai/gpt-oss-20b" }`

- `GET|HEAD /api/health`
  - Includes model and queue/concurrency status.

- `GET /api/queue-status`
  - Queue depth and success-rate counters.

## Environment variables

Required:

- `GROQ_API_KEY`
- `PINECONE_API_KEY`

Common runtime/config:

- `PINECONE_INDEX_NAME` (default `caus-legal-vdb`)
- `GROQ_MODEL_NAME` (default `openai/gpt-oss-20b`)
- `DEEPINFRA_TOKEN` (enables API embeddings path)
- `EMBEDDING_MODEL_NAME` (default `Qwen/Qwen3-Embedding-0.6B`)
- `EMBEDDING_BATCH_SIZE` (default `8`)
- `MAX_CONCURRENT_REQUESTS` (default `2`)
- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_API_KEY`
- `USER_AGENT`

## Useful commands

Install dependencies:

- `pip install -r requirements.txt`

Run API locally:

- `python app.py`
  or
- `uvicorn app:app --host 0.0.0.0 --port 8001`

Run scraper:

- `python scraper3.py`

Generate chunk corpus:

- `python Chunk_maker.py`

## Notes and caveats

- `tempCodeRunnerFile.py` appears to be a local experimental file and currently contains hard-coded API secrets. Treat it as non-production and rotate/revoke exposed credentials.
- `main.py` duplicates DeepInfra embedding logic instead of importing `deepinfra_embeddings.py`.
- Large data artifacts are present; targeted search by path is recommended for performance.
