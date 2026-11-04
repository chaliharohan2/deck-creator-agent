"""Runtime settings, read from environment variables and overridden by CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


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
    # --- model server
    backend: str = "ollama"  # "ollama" | "openai" (llama.cpp server, vLLM, SGLang, LocalAI, ...)
    base_url: str | None = None  # default depends on backend
    model: str = "qwen3.5:9b"
    api_key: str | None = None  # only if your local server was started with an API key
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
        return s
