import logging
from collections.abc import Iterator

import pytest

from multi_agent_research_assistant.config import Settings
from multi_agent_research_assistant.logging_config import configure_logging


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    yield
    root.setLevel(level)
    root.handlers[:] = handlers


def test_configure_logging_sets_root_level() -> None:
    configure_logging(Settings(_env_file=None, log_level="DEBUG"))
    assert logging.getLogger().level == logging.DEBUG

    configure_logging(Settings(_env_file=None, log_level="WARNING"))
    assert logging.getLogger().level == logging.WARNING


def test_configure_logging_does_not_duplicate_handlers() -> None:
    settings = Settings(_env_file=None)

    configure_logging(settings)
    configure_logging(settings)

    assert len(logging.getLogger().handlers) == 1
