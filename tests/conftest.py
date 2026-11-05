from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def png(tmp_path: Path) -> Path:
    path = tmp_path / "pic.png"
    Image.new("RGB", (640, 400), (30, 120, 200)).save(path)
    return path


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path_factory):
    """Keep the developer's real configuration out of the tests.

    Settings come from DECK_AGENT_* variables, the VLLM_* / MODEL_NAME fallbacks for server mode,
    and ./.env; a real GPU-server URL or key in the environment must never reach a test.
    """
    import os

    clear_deck_env(monkeypatch)
    before = set(os.environ)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))  # no stray ./.env
    yield
    for name in set(os.environ) - before:  # values a test loaded from a .env file
        if _is_deck_setting(name):
            os.environ.pop(name, None)


def _is_deck_setting(name: str) -> bool:
    return name.startswith(("DECK_AGENT_", "VLLM_")) or name == "MODEL_NAME"


def clear_deck_env(monkeypatch) -> None:
    import os

    for name in list(os.environ):
        if _is_deck_setting(name):
            monkeypatch.delenv(name, raising=False)
