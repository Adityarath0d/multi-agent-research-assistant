from fastapi import FastAPI

from multi_agent_research_assistant.api.health import router as health_router
from multi_agent_research_assistant.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory: builds a fresh app, optionally with custom settings."""
    settings = settings or get_settings()

    app = FastAPI(title=settings.app_name)
    app.state.settings = settings
    app.include_router(health_router)
    return app
