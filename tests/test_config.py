import pytest
from pydantic import SecretStr, ValidationError

from multi_agent_research_assistant.config import Settings


def test_log_level_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RA_LOG_LEVEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.log_level == "INFO"


def test_log_level_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RA_LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="LOUD")  # type: ignore[arg-type]


def test_api_key_is_not_exposed_in_repr() -> None:
    settings = Settings(_env_file=None, gemini_api_key=SecretStr("super-secret"))

    assert "super-secret" not in repr(settings)
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "super-secret"
