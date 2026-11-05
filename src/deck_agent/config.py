"""Runtime settings, read from environment variables (and an optional ``.env`` file) and
overridden by CLI flags.

Two connection *modes* are supported and can be switched at any time:

* ``cpu``    - a model on this computer, served by Ollama (or another local server).
* ``server`` - a model on a remote GPU server, e.g. vLLM's OpenAI-compatible API with an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

MODES = ("cpu", "server")
MODE_LABELS = {"cpu": "CPU (local)", "server": "GPU server"}
_MODE_ALIASES = {"cpu": "cpu", "local": "cpu", "ollama": "cpu", "server": "server", "gpu": "server", "remote": "server"}


def normalize_mode(mode: str) -> str:
    key = (mode or "").strip().lower()
    if key not in _MODE_ALIASES:
        raise ValueError(f"unknown mode {mode!r}; use 'cpu' or 'server'")
    return _MODE_ALIASES[key]


@dataclass
class ConnectionProfile:
    """How to reach one model server. The API key is never logged or sent to the browser."""

    backend: str  # "ollama" | "openai"
    base_url: str | None
    model: str
    api_key: str | None = None
    auth_header: str | None = None  # send the key in this header instead of "Authorization: Bearer <key>"
    think: bool = False
    num_ctx: int = 32768

    @property
    def is_remote(self) -> bool:
        host = urlparse(self.base_url or "http://localhost").hostname or "localhost"
        return host not in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

    def describe(self) -> str:
        where = self.base_url or "default local address"
        return f"{self.model or '(first model the server offers)'} via {self.backend} at {where}"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(f"DECK_AGENT_{name}")
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # --- active model connection (filled from the profile of the current mode)
    backend: str = "ollama"  # "ollama" | "openai" (llama.cpp server, vLLM, SGLang, LocalAI, ...)
    base_url: str | None = None  # default depends on backend
    model: str = "qwen3.5:9b"
    api_key: str | None = None  # only if the server requires one
    auth_header: str | None = None
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    num_ctx: int = 32768
    think: bool = False
    timeout: float = 900.0
    # --- agent
    max_steps: int = 60
    default_slides: int = 10
    # --- output
    output_dir: Path = Path("decks")
    formats: list[str] = field(default_factory=lambda: ["pptx"])
    theme: str | None = None
    # --- web research
    web: bool = True
    search_backend: str = "ddgs"  # "ddgs" (no key, metasearch) | "searxng" (self-hosted) | "wikipedia"
    wikipedia_fallback: bool = True  # query Wikipedia's API when the main search returns nothing
    searxng_url: str = "http://localhost:8888"
    search_region: str = "us-en"
    image_license: str = "Share"  # any | Public | Share | ShareCommercially | Modify | ModifyCommercially
    max_page_chars: int = 6000
    max_searches: int = 4  # per request; keeps small models from researching forever
    max_reads: int = 4
    allow_private_urls: bool = False
    verify_numbers: bool = True  # reject chart values / flag figures that appear in no source the agent saw
    # --- modes
    mode: str = "cpu"
    local: ConnectionProfile | None = None
    server: ConnectionProfile | None = None
    env_file: Path | None = None  # the .env file that was loaded, if any

    # ------------------------------------------------------------------ modes
    def _active_profile(self) -> ConnectionProfile:
        return ConnectionProfile(self.backend, self.base_url, self.model, self.api_key, self.auth_header,
                                 self.think, self.num_ctx)

    def save_active(self) -> None:
        """Remember changes made to the active connection (e.g. a different model) in its mode's profile."""
        if self.mode == "server":
            self.server = self._active_profile()
        else:
            self.local = self._active_profile()

    @property
    def server_configured(self) -> bool:
        return self.server is not None and bool(self.server.base_url)

    @property
    def is_remote(self) -> bool:
        return self._active_profile().is_remote

    def profile(self, mode: str) -> ConnectionProfile | None:
        mode = normalize_mode(mode)
        if mode == self.mode:
            return self._active_profile()
        return self.server if mode == "server" else self.local

    def use_mode(self, mode: str) -> None:
        """Switch the active connection to the ``cpu`` or ``server`` profile."""
        mode = normalize_mode(mode)
        if mode == "server" and not self.server_configured:
            raise ValueError("server mode is not configured: set DECK_AGENT_SERVER_URL (and DECK_AGENT_SERVER_API_KEY, "
                             "DECK_AGENT_SERVER_MODEL) in your .env file; see .env.example")
        self.save_active()
        target = self.server if mode == "server" else self.local
        assert target is not None
        self.mode = mode
        (self.backend, self.base_url, self.model, self.api_key, self.auth_header, self.think,
         self.num_ctx) = (target.backend, target.base_url, target.model, target.api_key, target.auth_header,
                          target.think, target.num_ctx)

    @classmethod
    def from_env(cls) -> Settings:
        s = cls()
        s.backend = _env("BACKEND", s.backend)
        s.base_url = _env("BASE_URL", s.base_url)
        s.model = _env("MODEL", s.model)
        s.api_key = _env("API_KEY", s.api_key)
        s.temperature = float(_env("TEMPERATURE", str(s.temperature)))
        s.top_p = float(_env("TOP_P", str(s.top_p)))
        s.top_k = int(_env("TOP_K", str(s.top_k)))
        s.num_ctx = int(_env("NUM_CTX", str(s.num_ctx)))
        s.think = _env_bool("THINK", s.think)
        s.timeout = float(_env("TIMEOUT", str(s.timeout)))
        s.max_steps = int(_env("MAX_STEPS", str(s.max_steps)))
        s.output_dir = Path(_env("OUTPUT_DIR", str(s.output_dir)))
        formats = _env("FORMATS")
        if formats:
            s.formats = [f.strip() for f in formats.split(",") if f.strip()]
        s.theme = _env("THEME", s.theme)
        s.web = _env_bool("WEB", s.web)
        s.search_backend = _env("SEARCH_BACKEND", s.search_backend)
        s.wikipedia_fallback = _env_bool("WIKIPEDIA_FALLBACK", s.wikipedia_fallback)
        s.searxng_url = _env("SEARXNG_URL", s.searxng_url)
        s.search_region = _env("SEARCH_REGION", s.search_region)
        s.image_license = _env("IMAGE_LICENSE", s.image_license)
        s.max_page_chars = int(_env("MAX_PAGE_CHARS", str(s.max_page_chars)))
        s.max_searches = int(_env("MAX_SEARCHES", str(s.max_searches)))
        s.max_reads = int(_env("MAX_READS", str(s.max_reads)))
        s.allow_private_urls = _env_bool("ALLOW_PRIVATE_URLS", s.allow_private_urls)
        s.verify_numbers = _env_bool("VERIFY_NUMBERS", s.verify_numbers)
        s.local = s._active_profile()
        # The GPU server. VLLM_BASE_URL / VLLM_API_KEY / MODEL_NAME are accepted as fallbacks.
        server_url = _env("SERVER_URL") or os.environ.get("VLLM_BASE_URL") or None
        if server_url:
            s.server = ConnectionProfile(
                backend=_env("SERVER_BACKEND", "openai"),
                base_url=server_url,
                model=_env("SERVER_MODEL") or os.environ.get("VLLM_MODEL") or os.environ.get("MODEL_NAME") or "",
                api_key=_env("SERVER_API_KEY") or os.environ.get("VLLM_API_KEY") or None,
                auth_header=_env("SERVER_AUTH_HEADER"),
                think=_env_bool("SERVER_THINK", False),
                num_ctx=int(_env("SERVER_NUM_CTX", "32768")),
            )
        mode = normalize_mode(_env("MODE", "cpu"))
        if mode == "server":
            s.use_mode("server")
        return s

