"""Tools available to the agent."""

from __future__ import annotations

from pydantic import Field

from deck_agent.config import Settings
from deck_agent.tools.base import Tool, ToolArgs, ToolError, ToolRegistry, ToolResult
from deck_agent.tools.deck_tools import DeckSession, register_deck_tools
from deck_agent.tools.web import (
    SearchBackend,
    fetch_page,
    format_image_results,
    format_search_results,
    make_search_backend,
)

__all__ = [
    "DeckSession",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_registry",
    "register_web_tools",
]


class SearchArgs(ToolArgs):
    query: str = Field(description="Focused search query; add the year for recent topics")
    max_results: int = Field(5, ge=1, le=10, description="Number of results (1-10)")


class ReadArgs(ToolArgs):
    url: str = Field(description="URL of the page (or PDF) to read, usually from web_search results")


class ImageSearchArgs(ToolArgs):
    query: str = Field(description="What the image should show, e.g. 'wind turbines offshore'")
    max_results: int = Field(5, ge=1, le=10, description="Number of results (1-10)")


def register_web_tools(registry: ToolRegistry, backend: SearchBackend, *, max_page_chars: int = 6000,
                       allow_private: bool = False, images: bool = True, max_searches: int = 4,
                       max_reads: int = 4, session: DeckSession | None = None) -> None:
    """Register web tools. Searches/reads are budgeted per request: small models tend to
    keep researching instead of building, so after the budget the tool tells them to move on."""
    used = {"web_search": 0, "read_webpage": 0, "empty_in_a_row": 0}
    registry.on_new_request(lambda: used.update(web_search=0, read_webpage=0, empty_in_a_row=0))

    def spend(tool: str, limit: int) -> None:
        if used[tool] >= limit:
            registry.disable(tool)
            raise ToolError(
                f"research limit reached ({limit} {tool} calls for this request). You have enough material: "
                "stop researching and build the deck now (set_deck_info, then add slides)."
            )
        used[tool] += 1

    @registry.add("web_search", "Search the web. Returns titles, URLs and snippets.", SearchArgs)
    def web_search(a: SearchArgs) -> str:
        if used["empty_in_a_row"] >= 2:
            registry.disable("web_search")
            raise ToolError(
                "web search is not returning results right now (the search engines may be rate-limiting), "
                "so it has been switched off for this request. Build the deck now from the material you already "
                "have and general knowledge, and do not invent precise statistics."
            )
        spend("web_search", max_searches)
        rows = backend.text(a.query, a.max_results)
        used["empty_in_a_row"] = 0 if rows else used["empty_in_a_row"] + 1
        if session is not None:
            session.note_source(" ".join(f"{r.get('url', '')} {r.get('title', '')} {r.get('snippet', '')}" for r in rows))
        return format_search_results(rows)

    @registry.add("read_webpage", "Read the main text of a web page or PDF to get facts and numbers.", ReadArgs)
    def read_webpage(a: ReadArgs) -> str:
        spend("read_webpage", max_reads)
        text = fetch_page(a.url, max_chars=max_page_chars, allow_private=allow_private)
        if session is not None:
            session.note_source(a.url + " " + text)  # requested + final URL, and the page's numbers
        return text

    if images:
        @registry.add("search_images",
                      "Find freely licensed (Creative Commons) images for a slide. Use sparingly.", ImageSearchArgs)
        def search_images(a: ImageSearchArgs) -> str:
            rows = backend.images(a.query, a.max_results)
            if session is not None:
                session.note_urls(" ".join(f"{r.get('page_url', '')} {r.get('image_url', '')}" for r in rows))
            return format_image_results(rows)


def build_registry(settings: Settings, session: DeckSession, backend: SearchBackend | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    if settings.web:
        backend = backend or make_search_backend(
            settings.search_backend, searxng_url=settings.searxng_url, region=settings.search_region,
            image_license=settings.image_license, wikipedia_fallback=settings.wikipedia_fallback,
        )
        register_web_tools(registry, backend, max_page_chars=settings.max_page_chars,
                           allow_private=settings.allow_private_urls, max_searches=settings.max_searches,
                           max_reads=settings.max_reads, session=session)
    register_deck_tools(registry, session)
    return registry
