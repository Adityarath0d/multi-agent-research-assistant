import logging

from fastapi import FastAPI

from multi_agent_research_assistant.api.health import router as health_router
from multi_agent_research_assistant.config import Settings, get_settings
from multi_agent_research_assistant.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory: builds a fresh app, optionally with custom settings."""
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(title=settings.app_name)
    app.state.settings = settings
    app.include_router(health_router)

    logger.info(
        "app_created name=%s environment=%s",
        settings.app_name,
        settings.environment,
    )
    return app
