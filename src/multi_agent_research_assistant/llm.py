import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

import litellm

from multi_agent_research_assistant.config import Settings

logger = logging.getLogger(__name__)


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(Exception):
    """Base class for LLM client errors."""


class LLMNotConfiguredError(LLMError):
    """Raised when the client is used without the settings it needs."""


class LLMCallError(LLMError):
    """Raised when the provider call fails or returns nothing usable."""


class LLMClient(Protocol):
    """What the rest of the app depends on. Real and fake clients both fit."""

    async def complete(self, messages: Sequence[Message]) -> LLMResult: ...


class LiteLLMClient:
    """LLMClient backed by LiteLLM, so the provider is a config string."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.llm_model
        self._timeout = settings.llm_timeout_seconds
        self._max_tokens = settings.llm_max_tokens
        self._max_retries = settings.llm_max_retries
        self._api_key = settings.gemini_api_key

    async def complete(self, messages: Sequence[Message]) -> LLMResult:
        if self._api_key is None:
            raise LLMNotConfiguredError("RA_GEMINI_API_KEY is not set")

        started = time.perf_counter()
        try:
            # Typed as Any: LiteLLM's response type is a union that makes
            # strict mypy awkward, and we only read a few fields below.
            response: Any = await litellm.acompletion(
                model=self._model,
                messages=list(messages),
                api_key=self._api_key.get_secret_value(),
                timeout=self._timeout,
                max_tokens=self._max_tokens,
                num_retries=self._max_retries,
            )
        except Exception as exc:
            logger.warning("llm_call_failed model=%s error=%s", self._model, type(exc).__name__)
            raise LLMCallError(f"LLM call failed: {type(exc).__name__}") from exc

        latency = time.perf_counter() - started

        text = response.choices[0].message.content
        if not text:
            logger.warning("llm_empty_response model=%s", self._model)
            raise LLMCallError("LLM returned an empty response")

        usage = getattr(response, "usage", None)
        result = LLMResult(
            text=text,
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_seconds=latency,
        )
        logger.info(
            "llm_call_ok model=%s prompt_tokens=%d completion_tokens=%d latency=%.2fs",
            result.model,
            result.prompt_tokens,
            result.completion_tokens,
            result.latency_seconds,
        )
        return result
