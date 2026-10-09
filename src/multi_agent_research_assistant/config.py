from functools import lru_cache
from typing import Literal

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
