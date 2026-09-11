"""FastAPI surface for the production, evidence-first LawGPT service."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from config import get_settings
from service import RAGError, RAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()
rag_service = RAGService(settings)
request_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
request_queue = deque()
queue_lock = asyncio.Lock()


class QueueStats:
    total_requests = 0
    completed_requests = 0
    failed_requests = 0
    current_queue_size = 0


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=settings.max_query_length)
    model: Optional[str] = Field(None, description="Compatibility field; server configuration is authoritative")


class QueryResponse(BaseModel):
    response: str
    model_used: str
    status: str = "grounded"
    grounded: bool = False
    insufficient_evidence: bool = False
    confidence: str = "low"
    claims: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    retrieval: dict = Field(default_factory=dict)
    request_id: Optional[str] = None


app = FastAPI(title="Turn2Law API", version=settings.version, description="Evidence-first Indian legal research API")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins), allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])


async def _enqueue(query: str):
    async with queue_lock:
        request_queue.append({"query": query[:50], "timestamp": datetime.now(timezone.utc).isoformat(), "status": "queued"})
        QueueStats.current_queue_size = len(request_queue)


async def _dequeue():
    async with queue_lock:
        if request_queue:
            request_queue.popleft()
        QueueStats.current_queue_size = len(request_queue)


async def _process(query: str, timeout: int):
    await _enqueue(query)
    try:
        async with asyncio.timeout(timeout):
            async with request_semaphore:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, rag_service.query, query)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="The verified legal research request timed out.") from exc
    finally:
        await _dequeue()


@app.on_event("startup")
async def startup():
    if settings.rag_mode != "rag":
        logger.error("Production startup rejected: RAG_MODE must be rag")
    else:
        logger.info("RAG service configured; dependencies initialize on first request")


@app.get("/")
@app.head("/")
async def root():
    return {"message": "Turn2Law API is running", "status": "healthy"}


@app.get("/favicon.ico")
async def favicon():
    return Response(content=b"", media_type="image/x-icon")


@app.post("/query")
@app.get("/query")
async def query_redirect():
    raise HTTPException(status_code=404, detail="Use /api/query")


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    QueueStats.total_requests += 1
    try:
        result = await _process(request.query, int(settings.generation_timeout + settings.retrieval_timeout + 15))
        QueueStats.completed_requests += 1
        return QueryResponse(**result)
    except RAGError as exc:
        QueueStats.failed_requests += 1
        if exc.category == "validation_error":
            raise HTTPException(status_code=400, detail="Invalid legal query.") from exc
        if exc.category == "insufficient_evidence":
            return QueryResponse(response="The available verified legal sources do not establish this proposition confidently.", model_used=settings.generation_model, status="insufficient_evidence", insufficient_evidence=True)
        logger.error("RAG request failed category=%s", exc.category)
        raise HTTPException(status_code=503, detail="Verified legal research is temporarily unavailable.") from exc
    except HTTPException:
        QueueStats.failed_requests += 1
        raise
    except Exception as exc:
        QueueStats.failed_requests += 1
        logger.exception("Unhandled API failure")
        raise HTTPException(status_code=503, detail="Verified legal research is temporarily unavailable.") from exc


@app.get("/api/health")
@app.head("/api/health")
async def health_check():
    diagnostics = rag_service.diagnostics()
    diagnostics.update({"status": "ready" if diagnostics["configuration_ready"] else "not_ready", "queue_size": QueueStats.current_queue_size, "max_concurrent_requests": settings.max_concurrent_requests, "total_requests": QueueStats.total_requests, "completed_requests": QueueStats.completed_requests, "failed_requests": QueueStats.failed_requests})
    return diagnostics


@app.get("/api/queue-status")
async def queue_status():
    total = QueueStats.total_requests
    return {"queue_size": QueueStats.current_queue_size, "max_concurrent_requests": settings.max_concurrent_requests, "total_requests_received": total, "completed_requests": QueueStats.completed_requests, "failed_requests": QueueStats.failed_requests, "success_rate": QueueStats.completed_requests / total * 100 if total else 0, "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
