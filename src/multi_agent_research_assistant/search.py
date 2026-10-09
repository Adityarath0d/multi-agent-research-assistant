import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from multi_agent_research_assistant.config import Settings

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str


class SearchError(Exception):
    """Base class for search client errors."""


class SearchNotConfiguredError(SearchError):
    """Raised when the client is used without the settings it needs."""


class SearchCallError(SearchError):
    """Raised when the search request fails or returns something unusable."""


class SearchClient(Protocol):
    """What the rest of the app depends on. Real and fake clients both fit."""

    async def search(self, query: str) -> list[SearchResult]: ...


class TavilySearchClient:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = settings.tavily_api_key
        self._max_results = settings.search_max_results
        self._timeout = settings.search_timeout_seconds
        self._transport = transport  # tests pass a fake network here

    async def search(self, query: str) -> list[SearchResult]:
        if self._api_key is None:
            raise SearchNotConfiguredError("RA_TAVILY_API_KEY is not set")

        payload = {
            "query": query,
            "max_results": self._max_results,
            "search_depth": "basic",
        }
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(TAVILY_SEARCH_URL, json=payload, headers=headers)
                response.raise_for_status()
                data: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("search_failed error=%s", type(exc).__name__)
            raise SearchCallError(f"Search failed: {type(exc).__name__}") from exc
        latency = time.perf_counter() - started

        results = [
            SearchResult(
                title=str(item.get("title") or ""),
                url=str(item["url"]),
                content=str(item.get("content") or ""),
            )
            for item in data.get("results", [])
            if item.get("url")
        ]
        logger.info(
            "search_ok results=%d latency=%.2fs",
            len(results),
            latency,
        )
        return results
