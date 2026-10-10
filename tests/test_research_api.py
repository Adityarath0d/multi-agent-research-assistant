from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.llm import (
    LLMNotConfiguredError,
    LLMResult,
    Message,
)
from multi_agent_research_assistant.main import create_app
from multi_agent_research_assistant.search import (
    SearchCallError,
    SearchNotConfiguredError,
    SearchResult,
)

RESULTS = [SearchResult(title="First", url="https://a.example", content="alpha")]


class FakeSearch:
    async def search(self, query: str) -> list[SearchResult]:
        return RESULTS


class FakeLLM:
    async def complete(self, messages: Sequence[Message]) -> LLMResult:
        return LLMResult(
            text="An answer [1].",
            model="fake-model",
            prompt_tokens=100,
            completion_tokens=20,
            latency_seconds=0.5,
        )


class FailingSearch:
    async def search(self, query: str) -> list[SearchResult]:
        raise SearchCallError("secret provider detail")


class UnconfiguredSearch:
    async def search(self, query: str) -> list[SearchResult]:
        raise SearchNotConfiguredError("no key")


class UnconfiguredLLM:
    async def complete(self, messages: Sequence[Message]) -> LLMResult:
        raise LLMNotConfiguredError("no key")


def _client(**kwargs: object) -> TestClient:
    settings = Settings(_env_file=None, environment="test")
    return TestClient(create_app(settings, **kwargs))  # type: ignore[arg-type]


def test_baseline_returns_answer_sources_and_usage() -> None:
    client = _client(search=FakeSearch(), llm=FakeLLM())

    response = client.post("/research/baseline", json={"question": "What is X?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "An answer [1]."
    assert body["sources"] == [{"index": 1, "title": "First", "url": "https://a.example"}]
    assert body["model"] == "fake-model"
    assert body["prompt_tokens"] == 100
    assert body["completion_tokens"] == 20


@pytest.mark.parametrize("question", ["", "hi", "x" * 501])
def test_invalid_question_is_rejected(question: str) -> None:
    client = _client(search=FakeSearch(), llm=FakeLLM())

    response = client.post("/research/baseline", json={"question": question})

    assert response.status_code == 422


def test_provider_failure_maps_to_502_without_leaking_details() -> None:
    client = _client(search=FailingSearch(), llm=FakeLLM())

    response = client.post("/research/baseline", json={"question": "What is X?"})

    assert response.status_code == 502
    assert "secret provider detail" not in response.text


def test_unconfigured_search_maps_to_503() -> None:
    client = _client(search=UnconfiguredSearch(), llm=FakeLLM())

    response = client.post("/research/baseline", json={"question": "What is X?"})

    assert response.status_code == 503


def test_unconfigured_llm_maps_to_503() -> None:
    client = _client(search=FakeSearch(), llm=UnconfiguredLLM())

    response = client.post("/research/baseline", json={"question": "What is X?"})

    assert response.status_code == 503


def test_real_clients_without_keys_return_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RA_TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("RA_GEMINI_API_KEY", raising=False)
    client = _client()  # no fakes: uses the real clients, with no keys

    response = client.post("/research/baseline", json={"question": "What is X?"})

    assert response.status_code == 503
