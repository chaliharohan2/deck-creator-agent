from __future__ import annotations

import os
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


# --------------------------------------------------------------------------- optional tools
# Tests marked `pdf` need LibreOffice; tests marked `browser` need Playwright and its Chromium.
# Locally they are skipped when the tool is missing. CI sets DECK_AGENT_CI_REQUIRE=pdf (or browser)
# in the job that installs the tool, so there a missing tool is an error instead of a silent skip:
# a job can never pass without running the tests it exists for.
_OPTIONAL = {"pdf": "LibreOffice (soffice)", "browser": "Playwright + Chromium (pip install -e '.[browser]' "
                                                        "&& python -m playwright install chromium)"}


def _browser_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            return Path(p.chromium.executable_path).exists()
    except Exception:  # noqa: BLE001 - driver problems mean "not available"
        return False


def pytest_collection_modifyitems(config, items):
    from deck_agent.deck.render import find_soffice

    required = {x.strip() for x in os.environ.get("DECK_AGENT_CI_REQUIRE", "").split(",") if x.strip()}
    unknown = required - set(_OPTIONAL)
    if unknown:
        raise pytest.UsageError(f"DECK_AGENT_CI_REQUIRE: unknown requirement(s) {sorted(unknown)}")
    used = {name for name in _OPTIONAL for item in items if name in item.keywords} | required
    checks = {"pdf": lambda: find_soffice() is not None, "browser": _browser_available}
    available = {name: checks[name]() for name in used}
    missing = sorted(name for name in required if not available[name])
    if missing:
        raise pytest.UsageError("this CI job requires " + ", ".join(_OPTIONAL[m] for m in missing)
                                + " but it is not installed")
    for item in items:
        for name, ok in available.items():
            if name in item.keywords and not ok:
                item.add_marker(pytest.mark.skip(reason=f"needs {_OPTIONAL[name]}"))
