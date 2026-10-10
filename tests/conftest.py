import os

import pytest


@pytest.fixture(autouse=True)
def clean_ra_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every RA_* variable so no test depends on the developer's shell."""
    for name in list(os.environ):
        if name.startswith("RA_"):
            monkeypatch.delenv(name)
