from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from pydantic import SecretStr

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.llm import (
    LiteLLMClient,
    LLMCallError,
    LLMNotConfiguredError,
    Message,
)

MESSAGES: list[Message] = [{"role": "user", "content": "What is pgvector?"}]


def _response(content: str | None, prompt: int = 10, completion: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        gemini_api_key=SecretStr("test-key"),
        llm_model="gemini/test-model",
        llm_timeout_seconds=7,
        llm_max_retries=1,
    )


async def test_complete_returns_text_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return _response("an answer", prompt=12, completion=34)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    result = await LiteLLMClient(_settings()).complete(MESSAGES)

    assert result.text == "an answer"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 34
    assert result.total_tokens == 46
    assert result.latency_seconds >= 0
    assert captured["model"] == "gemini/test-model"
    assert captured["api_key"] == "test-key"
    assert captured["timeout"] == 7
    assert captured["num_retries"] == 1


async def test_missing_api_key_raises_without_calling_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RA_GEMINI_API_KEY", raising=False)
    called = False

    async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
        nonlocal called
        called = True
        return _response("never returned")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(LLMNotConfiguredError):
        await LiteLLMClient(Settings(_env_file=None)).complete(MESSAGES)

    assert called is False


async def test_provider_error_is_wrapped_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
        raise RuntimeError("upstream said: key test-key is invalid")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(LLMCallError) as exc_info:
        await LiteLLMClient(_settings()).complete(MESSAGES)

    assert "test-key" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


async def test_empty_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**kwargs: Any) -> SimpleNamespace:
        return _response(None)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(LLMCallError):
        await LiteLLMClient(_settings()).complete(MESSAGES)
