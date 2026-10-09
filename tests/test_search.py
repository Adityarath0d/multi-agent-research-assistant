import json

import httpx
import pytest
from pydantic import SecretStr

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.search import (
    SearchCallError,
    SearchNotConfiguredError,
    TavilySearchClient,
)


def _client(handler: httpx.MockTransport) -> TavilySearchClient:
    settings = Settings(
        _env_file=None,
        tavily_api_key=SecretStr("test-key"),
        search_max_results=3,
    )
    return TavilySearchClient(settings, transport=handler)


async def test_search_parses_results_and_sends_expected_request() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "A", "url": "https://a.example", "content": "alpha"},
                    {"title": "B", "url": "https://b.example", "content": "beta"},
                ]
            },
        )

    results = await _client(httpx.MockTransport(handler)).search("what is pgvector")

    assert [r.url for r in results] == ["https://a.example", "https://b.example"]
    assert results[0].title == "A"
    assert results[0].content == "alpha"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["query"] == "what is pgvector"
    assert body["max_results"] == 3


async def test_results_without_url_are_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "no url", "content": "x"},
                    {"title": "ok", "url": "https://ok.example", "content": "y"},
                ]
            },
        )

    results = await _client(httpx.MockTransport(handler)).search("q")

    assert [r.url for r in results] == ["https://ok.example"]


async def test_missing_key_raises_without_any_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RA_TAVILY_API_KEY", raising=False)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"results": []})

    client = TavilySearchClient(Settings(_env_file=None), transport=httpx.MockTransport(handler))

    with pytest.raises(SearchNotConfiguredError):
        await client.search("q")

    assert calls == 0


async def test_http_error_is_wrapped_without_leaking_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "rate limited"})

    with pytest.raises(SearchCallError) as exc_info:
        await _client(httpx.MockTransport(handler)).search("q")

    assert "test-key" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)


async def test_invalid_json_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    with pytest.raises(SearchCallError):
        await _client(httpx.MockTransport(handler)).search("q")
