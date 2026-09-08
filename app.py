from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import logging
import threading
from dotenv import load_dotenv
import asyncio
from collections import deque
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timeout compatibility for Python < 3.11
if hasattr(asyncio, "timeout"):
    asyncio_timeout = asyncio.timeout
else:
    import async_timeout
    asyncio_timeout = async_timeout.timeout

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
load_dotenv()

# Import runtime and config from main.py
from main import (
    ragu,
    get_rag_chain,
    groq_model_name,
    index_name,
    pinecone_namespace,
    embedding_model_name,
    embedding_provider,
)

# ===== CONCURRENCY & QUEUE MANAGEMENT =====
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
request_queue = deque()
queue_lock = asyncio.Lock()

class QueueStats:
    total_requests = 0
    completed_requests = 0
    failed_requests = 0
    current_queue_size = 0

    @classmethod
    def increment_total(cls): cls.total_requests += 1
    @classmethod
    def increment_completed(cls): cls.completed_requests += 1
    @classmethod
    def increment_failed(cls): cls.failed_requests += 1
    @classmethod
    def set_queue_size(cls, size): cls.current_queue_size = size


async def add_to_queue(query: str):
    async with queue_lock:
        request_queue.append({
            "query": query[:50] + "..." if len(query) > 50 else query,
            "timestamp": datetime.now().isoformat(),
            "status": "queued"
        })
        QueueStats.set_queue_size(len(request_queue))


async def remove_from_queue():
    async with queue_lock:
        if request_queue:
            request_queue.popleft()
        QueueStats.set_queue_size(len(request_queue))


async def process_with_queue(query: str, timeout: int = 120):
    await add_to_queue(query)
    try:
        async with asyncio_timeout(timeout):
            async with request_semaphore:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, ragu, query)
                QueueStats.increment_completed()
                return response
    except asyncio.TimeoutError:
        QueueStats.increment_failed()
        raise HTTPException(status_code=504, detail="Request timeout. Please try again.")
    except Exception as e:
        QueueStats.increment_failed()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    finally:
        await remove_from_queue()


# Initialize FastAPI app
app = FastAPI(
    title="Turn2Law API",
    version="1.0.0",
    description="Legal query API with RAG (Retrieval Augmented Generation)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warmup():
    """
    Pre-warm the RAG stack in a background thread immediately after uvicorn
    binds the port. This way the port is open instantly (Render is happy) and
    the model loads in the background. Queries that arrive before warmup
    completes will still work — get_rag_chain() is idempotent and thread-safe
    enough for our single-worker setup.
    """
    def _warm():
        try:
            logger.info("Background warmup: initialising RAG stack...")
            get_rag_chain()
            logger.info("Background warmup complete — RAG stack ready.")
        except Exception as exc:
            logger.error("Background warmup failed: %s", exc, exc_info=True)

    thread = threading.Thread(target=_warm, daemon=True)
    thread.start()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    model: Optional[str] = Field(None)


class QueryResponse(BaseModel):
    response: str
    model_used: str


@app.get("/")
@app.head("/")
async def root():
    return {"message": "Turn2Law API is running", "status": "healthy"}


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(content=b"", media_type="image/x-icon")


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    logger.info(f"New query received: {request.query[:100]}...")
    QueueStats.increment_total()
    response = await process_with_queue(request.query, timeout=120)
    return QueryResponse(response=response, model_used=groq_model_name)


@app.get("/api/health")
@app.head("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model": groq_model_name,
        "rag_enabled": True,
        "pinecone_index": index_name,
        "pinecone_namespace": pinecone_namespace or "",
        "embedding_model": embedding_model_name,
        "embedding_provider": embedding_provider,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "queue_size": QueueStats.current_queue_size,
        "total_requests": QueueStats.total_requests,
        "completed_requests": QueueStats.completed_requests,
        "failed_requests": QueueStats.failed_requests,
    }


@app.get("/api/queue-status")
async def queue_status():
    return {
        "queue_size": QueueStats.current_queue_size,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "total_requests_received": QueueStats.total_requests,
        "completed_requests": QueueStats.completed_requests,
        "failed_requests": QueueStats.failed_requests,
        "success_rate": (
            QueueStats.completed_requests / QueueStats.total_requests * 100
            if QueueStats.total_requests > 0 else 0
        ),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
