import os

import litellm
import pytest
from pydantic import SecretStr

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.tracing import (
    LANGFUSE_CALLBACK,
    configure_tracing,
)


@pytest.fixture(autouse=True)
def isolate_global_state(monkeypatch: pytest.MonkeyPatch) -> None:
    # Registering these makes pytest restore them after each test, even
    # though configure_tracing writes to os.environ directly.
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_OTEL_HOST"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(litellm, "callbacks", [])


def test_tracing_is_off_without_keys() -> None:
    assert configure_tracing(Settings(_env_file=None)) is False
    assert LANGFUSE_CALLBACK not in litellm.callbacks


def test_tracing_is_off_with_only_one_key() -> None:
    settings = Settings(_env_file=None, langfuse_public_key=SecretStr("pk"))

    assert configure_tracing(settings) is False


def test_tracing_turns_on_with_both_keys() -> None:
    settings = Settings(
        _env_file=None,
        langfuse_public_key=SecretStr("pk-test"),
        langfuse_secret_key=SecretStr("sk-test"),
        langfuse_host="https://example.test",
    )

    assert configure_tracing(settings) is True

    assert LANGFUSE_CALLBACK in litellm.callbacks
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-test"
    assert os.environ["LANGFUSE_OTEL_HOST"] == "https://example.test"


def test_configuring_twice_does_not_duplicate_the_callback() -> None:
    settings = Settings(
        _env_file=None,
        langfuse_public_key=SecretStr("pk"),
        langfuse_secret_key=SecretStr("sk"),
    )

    configure_tracing(settings)
    configure_tracing(settings)

    assert litellm.callbacks.count(LANGFUSE_CALLBACK) == 1
