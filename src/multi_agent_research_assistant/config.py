from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Application settings, loaded from environment variables (prefix RA_)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RA_",
        extra="ignore",
    )

    app_name: str = "research-assistant"
    environment: Literal["local", "test", "production"] = "local"
    log_level: LogLevel = "INFO"

    # LLM. The model name is a LiteLLM string; "gemini/" routes to Google AI Studio.
    llm_model: str = "gemini/gemini-3.1-flash-lite"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_tokens: int = Field(default=2048, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    gemini_api_key: SecretStr | None = None
    # Tavily. The API key is optional; if not set, the search endpoint will be disabled.
    search_max_results: int = Field(default=5, gt=0, le=10)
    search_timeout_seconds: float = Field(default=10.0, gt=0)
    tavily_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
