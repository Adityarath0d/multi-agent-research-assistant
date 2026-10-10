import logging
import os
from typing import Literal

import litellm

from multi_agent_research_assistant.config import Settings

logger = logging.getLogger(__name__)

LANGFUSE_CALLBACK: Literal["langfuse_otel"] = "langfuse_otel"


def configure_tracing(settings: Settings) -> bool:
    """Turn Langfuse tracing on if both keys are set. Returns whether it is on.

    LiteLLM's Langfuse callback reads its credentials from environment
    variables, so we copy them from our settings here, in one place.
    Missing keys just mean tracing is off; the app must work either way.
    """
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key

    if public_key is None or secret_key is None:
        while LANGFUSE_CALLBACK in litellm.callbacks:
            litellm.callbacks.remove(LANGFUSE_CALLBACK)
        logger.info("tracing_disabled reason=no_langfuse_keys")
        return False

    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key.get_secret_value()
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key.get_secret_value()
    os.environ["LANGFUSE_OTEL_HOST"] = settings.langfuse_host
    if LANGFUSE_CALLBACK not in litellm.callbacks:
        litellm.callbacks.append(LANGFUSE_CALLBACK)
    logger.info("tracing_enabled host=%s", settings.langfuse_host)
    return True
