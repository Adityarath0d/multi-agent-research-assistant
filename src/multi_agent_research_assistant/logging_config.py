import logging
import sys

from multi_agent_research_assistant.config import Settings

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(settings: Settings) -> None:
    """Configure root logging once per app creation."""
    logging.basicConfig(
        level=settings.log_level,
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    # HTTP client libraries log full request URLs at INFO, which can leak
    # query parameters. Keep them quiet unless something goes wrong.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
