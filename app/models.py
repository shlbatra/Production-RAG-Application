"""
API Request and Response Models
Pydantic models for input validation and response structure.
"""

from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid

class ChatRequest(BaseModel):
    """Incoming chat request"""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user message to agent"
    )
    thread_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Conversation Thread Id")

class ChatResponse(BaseModel):
    """Chat response returned to Client"""
    response: str
    thread_id: str
    model_used: str
    cached: bool = False
    processing_time_ms: float
    security_notes: list[str] = Field(default_factory=list)
    timestamp: str=Field(
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