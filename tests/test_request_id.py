import logging

from fastapi.testclient import TestClient

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.logging_config import RequestIDFilter
from multi_agent_research_assistant.main import create_app
from multi_agent_research_assistant.request_context import request_id_var


def _client() -> TestClient:
    return TestClient(create_app(Settings(_env_file=None, environment="test")))


def test_response_includes_a_request_id() -> None:
    response = _client().get("/health")

    request_id = response.headers["x-request-id"]
    assert len(request_id) == 32


def test_each_request_gets_a_different_id() -> None:
    client = _client()

    first = client.get("/health").headers["x-request-id"]
    second = client.get("/health").headers["x-request-id"]

    assert first != second


def test_filter_stamps_records_with_the_current_request_id() -> None:
    token = request_id_var.set("abc123")
    try:
        record = logging.LogRecord("t", logging.INFO, "f.py", 1, "msg", None, None)
        assert RequestIDFilter().filter(record) is True
        assert record.request_id == "abc123"  # type: ignore[attr-defined]
    finally:
        request_id_var.reset(token)


def test_filter_uses_dash_outside_a_request() -> None:
    record = logging.LogRecord("t", logging.INFO, "f.py", 1, "msg", None, None)

    RequestIDFilter().filter(record)

    assert record.request_id == "-"  # type: ignore[attr-defined]
