"""
API Request and Response Models
Pydantic models for input validation and response structure.
"""

from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ChatRequest(BaseModel):
    """Incoming chat request"""

    message: str = Field(
        ..., min_length=1, max_length=10000, description="The user message to agent"
    )


class SourceReference(BaseModel):
    """A retrieved document chunk used to inform the response."""

    source: str
    # Optional: tool-calling mode recovers sources from the search tool's text
    # output, which carries the source name and chunk but not a similarity score.
    similarity: float | None = None
    chunk_preview: str


class ChatResponse(BaseModel):
    """Chat response returned to Client"""

    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time_ms: float
    security_notes: list[str] = Field(default_factory=list)
    sources: list[SourceReference] | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HealthResponse(BaseModel):
    """Health response check"""

    status: str = "healthy"
    environment: str
    version: str = "1.0.0"
    checks: dict = {}


class MetricsResponse(BaseModel):
    """Metrics endpoint response"""

    total_requests: int
    total_errors: int
    error_rate: str
    avg_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    request_id: str | None = None


class DocumentUploadResponse(BaseModel):
    """Response after a document has been ingested."""

    doc_id: str
    filename: str
    chunks_stored: int
    status: str
    processing_time_ms: float
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class DocumentInfo(BaseModel):
    """Summary of a single ingested document."""

    doc_id: str
    source: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response listing all ingested documents."""

    documents: list[DocumentInfo]
    total_documents: int


class DocumentDeleteResponse(BaseModel):
    """Response after deleting a document."""

    doc_id: str
    chunks_deleted: int
    status: str
