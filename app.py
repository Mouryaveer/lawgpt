from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import logging
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
    groq_model_name,
    index_name,
    pinecone_namespace,
    embedding_model_name,
    embedding_provider,
)

# ===== CONCURRENCY & QUEUE MANAGEMENT =====
# Semaphore to limit concurrent requests (max 2 concurrent requests to avoid crashes)
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Request queue tracking
request_queue = deque()
queue_lock = asyncio.Lock()

class QueueStats:
    """Track queue statistics"""
    total_requests = 0
    completed_requests = 0
    failed_requests = 0
    current_queue_size = 0
    
    @classmethod
    def increment_total(cls):
        cls.total_requests += 1
    
    @classmethod
    def increment_completed(cls):
        cls.completed_requests += 1
    
    @classmethod
    def increment_failed(cls):
        cls.failed_requests += 1
    
    @classmethod
    def set_queue_size(cls, size):
        cls.current_queue_size = size


async def add_to_queue(query: str):
    """Add request to queue"""
    async with queue_lock:
        request_queue.append({
            "query": query[:50] + "..." if len(query) > 50 else query,
            "timestamp": datetime.now().isoformat(),
            "status": "queued"
        })
        QueueStats.set_queue_size(len(request_queue))
        logger.info(f"Queue size: {len(request_queue)}")


async def remove_from_queue():
    """Remove request from queue"""
    async with queue_lock:
        if request_queue:
            request_queue.popleft()
        QueueStats.set_queue_size(len(request_queue))
        logger.info(f"Queue size: {len(request_queue)}")


async def process_with_queue(query: str, timeout: int = 120):
    """
    Process query with concurrency control and queue management.
    
    Args:
        query: The legal query to process
        timeout: Maximum time to wait for processing (seconds)
        
    Returns:
        The RAG response
        
    Raises:
        HTTPException: If processing fails or timeout occurs
    """
    await add_to_queue(query)
    
    try:
        # Acquire semaphore with timeout
        async with asyncio_timeout(timeout):
            async with request_semaphore:
                logger.info(f"Processing query (queue size before: {QueueStats.current_queue_size})")
                
                # Run the blocking ragu function in a thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, ragu, query)
                
                QueueStats.increment_completed()
                logger.info(f"Query processed successfully")
                return response
                
    except asyncio.TimeoutError:
        QueueStats.increment_failed()
        logger.error("Request timeout - processing took too long")
        raise HTTPException(
            status_code=504,
            detail="Request timeout. The query took too long to process. Please try again."
        )
    except Exception as e:
        QueueStats.increment_failed()
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
    finally:
        await remove_from_queue()

# Initialize FastAPI app
app = FastAPI(
    title="Turn2Law API",
    version="1.0.0",
    description="Legal query API with RAG (Retrieval Augmented Generation)"
)

# Configure CORS for frontend access
# In production, replace "*" with your specific frontend domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production, e.g., ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)



class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The legal query to process")
    model: Optional[str] = Field(None, description="Optional model selection (currently not used)")


class QueryResponse(BaseModel):
    response: str = Field(..., description="The generated response from the RAG system")
    model_used: str = Field(..., description="The model used for generation")


@app.get("/")
@app.head("/")
async def root():
    """Root endpoint - supports both GET and HEAD for health checks"""
    return {"message": "Turn2Law API is running", "status": "healthy"}


@app.get("/favicon.ico")
async def favicon():
    """Favicon endpoint to prevent 404 errors"""
    from fastapi.responses import Response
    # Return empty response with proper content type
    return Response(content=b"", media_type="image/x-icon")


@app.post("/query")
@app.get("/query")
async def query_redirect():
    """Redirect users to the correct endpoint"""
    raise HTTPException(
        status_code=404,
        detail="This endpoint has been moved. Please use /api/query instead."
    )


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query and return a model-generated response using RAG.
    
    This endpoint accepts a legal query from the frontend, processes it through
    the RAG (Retrieval Augmented Generation) system, and returns a response.
    Uses concurrent request limiting to prevent crashes.
    
    Args:
        request: QueryRequest containing the user's query
        
    Returns:
        QueryResponse with the model's response
        
    Raises:
        HTTPException: If there's an error processing the query or timeout
    """
    logger.info(f"New query received: {request.query[:100]}...")
    QueueStats.increment_total()
    
    # Process with queue and concurrency control
    response = await process_with_queue(request.query, timeout=120)
    
    return QueryResponse(
        response=response,
        model_used=groq_model_name,
    )


@app.get("/api/health")
@app.head("/api/health")
async def health_check():
    """Health check endpoint - supports both GET and HEAD"""
    return {
        "status": "healthy",
        "model": groq_model_name,
        "rag_enabled": True,
        "pinecone_index": index_name,
        "pinecone_namespace": pinecone_namespace or "<default>",
        "embedding_model": embedding_model_name,
        "embedding_provider": embedding_provider,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "queue_size": QueueStats.current_queue_size,
        "total_requests": QueueStats.total_requests,
        "completed_requests": QueueStats.completed_requests,
        "failed_requests": QueueStats.failed_requests
    }


@app.get("/api/queue-status")
async def queue_status():
    """Get current queue status and statistics"""
    return {
        "queue_size": QueueStats.current_queue_size,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "total_requests_received": QueueStats.total_requests,
        "completed_requests": QueueStats.completed_requests,
        "failed_requests": QueueStats.failed_requests,
        "success_rate": (
            QueueStats.completed_requests / QueueStats.total_requests * 100
            if QueueStats.total_requests > 0
            else 0
        ),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)


