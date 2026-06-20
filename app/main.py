"""
Production-Ready FastAPI + LangGraph Application

Wires together:
- Security pipeline (input sanitization, PII masking)
- Response caching
- Rate limiting (slowapi)
- LangGraph agent (with retries + fallback)
- Structured logging + metrics
- LangSmith tracing
- Health checks
"""

from fastapi import FastAPI, Request, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv

from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    HealthResponse,
    MetricsResponse,
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.agent import ProductionAgent
from app.document_store import DocumentStore
from app.ingestion import ingest_document

load_dotenv()

# === Global instances (initialized in lifespan) ===
#  Because initialization might fail (e.g., invalid API key), and you want that to happen during the lifespan startup phase where FastAPI can handle it properly, not at module import time.
security: SecurityPipeline = None  # type: ignore[assignment]
cache: ResponseCache = None  # type: ignore[assignment]
metrics: MetricsCollector = None  # type: ignore[assignment]
agent: ProductionAgent = None  # type: ignore[assignment]
document_store: DocumentStore | None = None
logger = get_logger()


# === Lifespan (startup/shutdown) ===
async def lifespan(app: FastAPI):
    """
    Initialize all components on startup, clean up on shutdown.
    Modern FASTAPI development (replaces @app.on_event)
    """

    global security, cache, metrics, agent, document_store

    settings = get_settings()

    logger.info(
        "Starting production API...",
        extra={
            "extra_data": {
                "environment": settings.app_env,
                "primary_model": settings.primary_model,
                "tracing_enabled": settings.langchain_tracing_v2,
            }
        },
    )

    # Initialize components
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    if settings.rag_enabled:
        document_store = DocumentStore(settings)
        logger.info("Supabase document store initialized (RAG enabled)")
    else:
        logger.info("Supabase not configured (RAG disabled)")

    agent = ProductionAgent(document_store=document_store)

    logger.info("All components initialized. Ready to serve requests")

    yield  # App is running

    # Shutdown
    logger.info("Shutting down...", extra={"extra_data": metrics.summary})


# === Rate Limiter Setup ===
limiter = Limiter(key_func=get_remote_address)

# === FastAPI App ===
app = FastAPI(
    title="Production LangGraph API",
    description="A production-ready chat API with security, caching, and observability.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter

# === Exception Handlers ===


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors"""
    logger.warning(
        "Rate limit exceeded",
        extra={"extra_data": {"client_ip": get_remote_address(request)}},
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate Limit Exceeded",
            "detail": "Too many requests. Please slow down",
        },
    )


# =============================================
# ENDPOINTS
# =============================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker/Kubernetes."""
    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
        "document_store": (
            document_store.health_check() if document_store else "not_configured"
        ),
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.
    Flow:
    1. Security check (Injection + PII Masking)
    2. Cache lookup
    3. Langgraph Agent Invoke (if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """
    with RequestTimer() as timer:
        security_notes = []

        # ---- Step 1: Security Check ----
        is_allowed, cleaned_message, notes = security.check_input(body.message)

        if not is_allowed:
            metrics.record_request(latency_ms=0, error=True)
            logger.warning(
                "Request blocked by security",
                extra={
                    "extra_data": {
                        "reason": notes,
                        "thread_id": body.thread_id,
                    }
                },
            )

            raise HTTPException(
                status_code=400, detail="Your message was blocked by security filters"
            )

        # ---- Step 2: Cache Lookup ----
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info(
                "Cache hit",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                    }
                },
            )

            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )

        # ---- Step 3: Invoke LangGraph Agent ----
        try:
            result = agent.invoke(cleaned_message)
        except Exception as e:
            metrics.record_request(latency_ms=0, error=True)
            logger.error(
                f"Agent invocation failed: {e}",
                extra={
                    "extra_data": {
                        "thread_id": body.thread_id,
                        "error": str(e),
                    }
                },
            )
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request.",
            )

        response_text = result["response"]
        model_used = result["model_used"]
        sources = result.get("sources", [])

        # ---- Step 4: Output Validation ----
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # ---- Step 5: Cache Store ----
        cache.set(cleaned_message, validated_response)

    # ---- Step 6: Log & Record Metrics ----
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    if security_notes:
        logger.info(
            "Security notes",
            extra={
                "extra_data": {
                    "notes": security_notes,
                    "thread_id": body.thread_id,
                }
            },
        )

    logger.info(
        "Request completed",
        extra={
            "extra_data": {
                "thread_id": body.thread_id,
                "model_used": model_used,
                "latency_ms": round(timer.elapsed_ms, 2),
            }
        },
    )

    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        processing_time_ms=round(timer.elapsed_ms, 2),
        security_notes=security_notes,
        sources=sources or None,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Metrics for monitoring dashboards"""
    summary = metrics.summary
    return MetricsResponse(**summary)


@app.get("/cache/stats")
async def cache_stats():
    """Cache performance statistics."""
    return cache.stats


@app.post("/documents", response_model=DocumentUploadResponse)
@limiter.limit("5/minute")
async def upload_document(request: Request, file: UploadFile):
    """Upload a document for RAG ingestion."""
    if document_store is None:
        raise HTTPException(status_code=503, detail="Document store not configured")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    settings = get_settings()

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    file_bytes = await file.read()
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB",
        )

    with RequestTimer() as timer:
        try:
            result = ingest_document(
                file_bytes, file.filename, document_store, settings
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(
                "Document ingestion failed",
                extra={"extra_data": {"filename": file.filename, "error": str(e)}},
            )
            raise HTTPException(status_code=500, detail="Document ingestion failed")

    return DocumentUploadResponse(
        doc_id=result["doc_id"],
        filename=result["filename"],
        chunks_stored=result["chunks_stored"],
        status="success",
        processing_time_ms=round(timer.elapsed_ms, 2),
    )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all ingested documents."""
    if document_store is None:
        raise HTTPException(status_code=503, detail="Document store not configured")

    rows = document_store.list_documents()
    documents = [
        DocumentInfo(
            doc_id=row["doc_id"],
            source=row["source"],
            chunk_count=row["chunk_count"],
        )
        for row in rows
    ]
    return DocumentListResponse(documents=documents, total_documents=len(documents))


@app.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str):
    """Delete a document and all its chunks."""
    if document_store is None:
        raise HTTPException(status_code=503, detail="Document store not configured")

    chunks_deleted = document_store.delete_document(doc_id)
    if chunks_deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDeleteResponse(
        doc_id=doc_id,
        chunks_deleted=chunks_deleted,
        status="deleted",
    )
