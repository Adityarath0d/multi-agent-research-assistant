import logging
import sys

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.request_context import request_id_var

LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s %(message)s"


class RequestIDFilter(logging.Filter):
    """Stamp every log record with the current request ID."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(settings: Settings) -> None:
    """Configure root logging once per app creation."""
    logging.basicConfig(
        level=settings.log_level,
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIDFilter())
    # HTTP client libraries log full request URLs at INFO, which can leak
    # query parameters. Keep them quiet unless something goes wrong.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
