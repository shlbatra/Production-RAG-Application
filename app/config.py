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
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "production-api"

    # Supabase
    supabase_url: str = ""
    supabase_service_key: str = ""

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
    def rag_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key)


@lru_cache
def get_settings() -> Settings:
    """Cache settings instance - loaded once, reused everywhere"""
    return Settings()  # type: ignore[call-arg]
