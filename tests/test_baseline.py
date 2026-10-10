from collections.abc import Sequence

import pytest

from multi_agent_research_assistant.llm import LLMResult, Message
from multi_agent_research_assistant.pipelines.baseline import (
    NO_SOURCES_ANSWER,
    build_messages,
    run_baseline,
)
from multi_agent_research_assistant.search import SearchCallError, SearchResult

RESULTS = [
    SearchResult(title="First", url="https://a.example", content="alpha facts"),
    SearchResult(title="Second", url="https://b.example", content="beta facts"),
]


class FakeSearch:
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results if results is not None else []
        self.queries: list[str] = []

    async def search(self, query: str) -> list[SearchResult]:
        self.queries.append(query)
        return self._results


class FailingSearch:
    async def search(self, query: str) -> list[SearchResult]:
        raise SearchCallError("boom")


class FakeLLM:
    def __init__(self, text: str = "An answer [1].") -> None:
        self._text = text
        self.calls: list[list[Message]] = []

    async def complete(self, messages: Sequence[Message]) -> LLMResult:
        self.calls.append(list(messages))
        return LLMResult(
            text=self._text,
            model="fake-model",
            prompt_tokens=100,
            completion_tokens=20,
            latency_seconds=0.5,
        )


def test_build_messages_numbers_sources_and_includes_question() -> None:
    messages = build_messages("What is pgvector?", RESULTS)

    assert messages[0]["role"] == "system"
    assert "untrusted" in messages[0]["content"]
    user = messages[1]["content"]
    assert "[1] First" in user
    assert "[2] Second" in user
    assert "https://b.example" in user
    assert "Question: What is pgvector?" in user


async def test_run_baseline_returns_answer_sources_and_usage() -> None:
    search = FakeSearch(RESULTS)
    llm = FakeLLM("pgvector adds vectors to Postgres [1].")

    result = await run_baseline("What is pgvector?", search, llm)

    assert search.queries == ["What is pgvector?"]
    assert len(llm.calls) == 1
    assert result.answer == "pgvector adds vectors to Postgres [1]."
    assert [s.url for s in result.sources] == ["https://a.example", "https://b.example"]
    assert [s.index for s in result.sources] == [1, 2]
    assert result.model == "fake-model"
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 20
    assert result.llm_seconds == 0.5
    assert result.total_seconds >= result.search_seconds


async def test_no_sources_skips_the_model_call() -> None:
    llm = FakeLLM()

    result = await run_baseline("obscure question", FakeSearch([]), llm)

    assert result.answer == NO_SOURCES_ANSWER
    assert result.sources == []
    assert result.prompt_tokens == 0
    assert llm.calls == []


async def test_search_errors_propagate() -> None:
    with pytest.raises(SearchCallError):
        await run_baseline("q", FailingSearch(), FakeLLM())
