"""
Centralized Configuration
Uses pydantic-settings for validated environment variables.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

load_dotenv()


class Settings(BaseSettings):
    # LLM Config
    openai_api_key: str
    primary_model: str = "gpt-4.1-mini"
    fallback_model: str = "gpt-4.1-nano"

    # Langsmith
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_endpoint: str = "https://eu.api.smith.langchain.com"
    langsmith_project: str = "Prod RAG Project"

    # Supabase
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_database_url: str = ""

    # RAG Settings
    embedding_model: str = "text-embedding-3-small"
    rag_chunking_strategy: str = "contextual"
    rag_context_header_lines: int = 5
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_retrieval_strategy: str = "hybrid"
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.55
    max_upload_size_mb: int = 10
    max_tool_calls: int = 3  # max search rounds per request (tool-calling guard)

    # Redis
    redis_url: str = ""  # Local: "redis://localhost:6379/0", Upstash: "rediss://default:xxx@xxx.upstash.io:6379"

    # Connection Pool
    db_pool_min_conn: int = 2
    db_pool_max_conn: int = 10

    # Security
    enable_pii_detection: bool = False

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def redis_enabled(self) -> bool:
        return bool(self.redis_url)

    @property
    def rag_enabled(self) -> bool:
        return bool(self.supabase_database_url)


@lru_cache
def get_settings() -> Settings:
    """Cache settings instance - loaded once, reused everywhere"""
    return Settings()  # type: ignore[call-arg]
