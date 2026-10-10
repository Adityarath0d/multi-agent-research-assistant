import logging

from fastapi import FastAPI

from multi_agent_research_assistant.api.health import router as health_router
from multi_agent_research_assistant.api.middleware import RequestIDMiddleware
from multi_agent_research_assistant.api.research import router as research_router
from multi_agent_research_assistant.config import Settings, get_settings
from multi_agent_research_assistant.llm import LiteLLMClient, LLMClient
from multi_agent_research_assistant.logging_config import configure_logging
from multi_agent_research_assistant.search import SearchClient, TavilySearchClient
from multi_agent_research_assistant.tracing import configure_tracing

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    search: SearchClient | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    """Application factory. Tests can inject fake search and LLM clients."""
    settings = settings or get_settings()
    configure_logging(settings)
    configure_tracing(settings)

    app = FastAPI(title=settings.app_name)
    app.add_middleware(RequestIDMiddleware)
    app.state.settings = settings
    app.state.search = search if search is not None else TavilySearchClient(settings)
    app.state.llm = llm if llm is not None else LiteLLMClient(settings)
    app.include_router(health_router)
    app.include_router(research_router)

    logger.info(
        "app_created name=%s environment=%s",
        settings.app_name,
        settings.environment,
    )
    return app
