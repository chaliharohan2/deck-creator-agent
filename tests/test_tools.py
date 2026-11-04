import json

import httpx
import pytest
from pptx import Presentation

from deck_agent.config import Settings
from deck_agent.tools import DeckSession, ToolError, build_registry
from deck_agent.tools.web import (
    FallbackBackend,
    WikipediaBackend,
    check_url,
    extract_page_text,
    format_search_results,
    make_search_backend,
    save_image,
)


class FakeSearch:
    name = "fake"

    def __init__(self):
        self.queries = []

    def text(self, query, max_results):
        self.queries.append(query)
        return [{"title": f"Result {i}", "url": f"https://example.com/{i}", "snippet": "snip " * 100}
                for i in range(max_results)]

    def images(self, query, max_results):
        return [{"title": "img", "image_url": "https://example.com/a.png", "page_url": "https://example.com/p",
                 "width": 800, "height": 600, "license": "Share"}]


@pytest.fixture
def setup(tmp_path):
    settings = Settings(output_dir=tmp_path, formats=["pptx"])
    session = DeckSession(output_dir=tmp_path, formats=["pptx"])
    search = FakeSearch()
    registry = build_registry(settings, session, backend=search)
    return registry, session, search


def test_schemas_are_flat_and_keep_title_properties(setup):
    registry, _, _ = setup
    schemas = registry.schemas()
    dumped = json.dumps(schemas)
    assert "$ref" not in dumped and "$defs" not in dumped and "anyOf" not in dumped
    bullet = next(s for s in schemas if s["function"]["name"] == "add_bullet_slide")["function"]["parameters"]
    assert "title" in bullet["properties"]  # a property called "title" must survive title-stripping
    assert bullet["required"] == ["title", "bullets"]
    assert set(registry.names) >= {"web_search", "read_webpage", "search_images", "save_deck"}


def test_no_web_registry(tmp_path):
    settings = Settings(web=False)
    registry = build_registry(settings, DeckSession(output_dir=tmp_path))
    assert "web_search" not in registry


def test_full_deck_flow(setup, tmp_path, png):
    registry, session, _ = setup
    ok = lambda name, **kw: registry.execute(name, kw)  # noqa: E731

    r = ok("set_deck_info", title="Heat Pumps", subtitle="Europe 2026", theme="forest")
    assert r.ok and session.deck.theme == "forest"
    assert ok("add_bullet_slide", title="Why", bullets="- a\n- b\n- c").ok
    r = ok("add_chart_slide", title="Sales", chart_type="column", categories=[2022, 2023],
           series=[{"name": "Units (m)", "values": ["2.1", "3,000"]}], sources=["https://ehpa.org"])
    assert r.ok, r.content
    assert ok("add_table_slide", title="T", headers=["a", "b"], rows=[[1, 2], ["x", "y"]]).ok
    assert ok("add_two_column_slide", title="C", left_heading="L", left_bullets=["x"], right_heading="R",
              right_bullets=["y"]).ok
    assert ok("add_image_slide", title="Pic", image=str(png), caption="cap").ok
    assert ok("add_quote_slide", quote="Q", attribution="A").ok
    assert ok("add_section_slide", title="Part 2", position=2).ok
    assert [s.kind for s in session.deck.slides] == [
        "title", "section", "bullets", "chart", "table", "two_column", "image", "quote"]

    assert ok("move_slide", slide_number=2, new_position=8).ok
    assert session.deck.slides[-1].kind == "section"
    assert ok("update_slide", slide_number=2, title="Why it matters", bullets=["x", "y"]).ok
    assert session.deck.slides[1].title == "Why it matters"
    assert not ok("update_slide", slide_number=4, bullets=["x"]).ok  # table slide has no bullets
    assert ok("delete_slide", slide_number=8).ok
    assert "Why it matters" in ok("list_slides").content

    r = ok("save_deck", filename="../../evil name", formats=["pptx", "html"])
    assert r.ok, r.content
    assert (tmp_path / "evil_name.pptx").exists() and (tmp_path / "evil_name.html").exists()
    assert not session.dirty
    assert len(Presentation(tmp_path / "evil_name.pptx").slides) == 7


def test_errors_are_returned_not_raised(setup):
    registry, session, _ = setup
    r = registry.execute("add_bullet_slide", {"title": "x"})
    assert not r.ok and "bullets: this required field is missing" in r.content
    r = registry.execute("add_chart_slide", {"title": "c", "categories": ["a", "b"], "series": [{"name": "s", "values": [1]}]})
    assert not r.ok and "one-to-one" in r.content
    r = registry.execute("add_bulet_slide", {})
    assert not r.ok and "Did you mean" in r.content or "Unknown tool" in r.content
    r = registry.execute("delete_slide", {"slide_number": 5})
    assert not r.ok and "does not exist" in r.content
    r = registry.execute("save_deck", {})
    assert not r.ok and "no slides" in r.content
    r = registry.execute("add_image_slide", {"title": "x", "image": "/definitely/missing.png"})
    assert not r.ok and "not found" in r.content
    assert session.deck.slides == []


def test_bullet_slide_creates_title_slide_from_deck_info(setup):
    registry, session, _ = setup
    session.deck.title = "Untitled presentation"
    registry.execute("add_bullet_slide", {"title": "a", "bullets": ["x"]})
    assert session.deck.slides[0].kind == "bullets"  # no title known yet -> no title slide invented
    registry.execute("set_deck_info", {"title": "Real"})
    assert session.deck.slides[0].kind == "title"


def test_long_bullets_get_a_tip(setup):
    registry, _, _ = setup
    r = registry.execute("add_bullet_slide", {"title": "a", "bullets": ["word " * 30]})
    assert r.ok and "Tip" in r.content


def test_web_tools_use_backend(setup):
    registry, _, search = setup
    r = registry.execute("web_search", {"query": "heat pumps 2026", "max_results": 3})
    assert r.ok and r.content.count("URL:") == 3 and search.queries == ["heat pumps 2026"]
    assert "…" in r.content  # long snippets are shortened
    r = registry.execute("search_images", {"query": "x"})
    assert r.ok and "image: https://example.com/a.png" in r.content


def test_research_budget_resets_per_request(setup):
    registry, _, _ = setup
    for i in range(4):
        assert registry.execute("web_search", {"query": f"q{i}"}).ok
    r = registry.execute("web_search", {"query": "q5"})
    assert not r.ok and "research limit reached" in r.content and "build the deck now" in r.content
    assert not registry.is_enabled("web_search") and registry.is_enabled("read_webpage")
    registry.new_request()
    assert registry.execute("web_search", {"query": "q6"}).ok


class EmptySearch(FakeSearch):
    def text(self, query, max_results):
        self.queries.append(query)
        return []


def test_consecutive_empty_searches_stop_research(tmp_path):
    search = EmptySearch()
    registry = build_registry(Settings(), DeckSession(output_dir=tmp_path), backend=search)
    assert "No results" in registry.execute("web_search", {"query": "a"}).content
    assert "No results" in registry.execute("web_search", {"query": "b"}).content
    r = registry.execute("web_search", {"query": "c"})
    assert not r.ok and "switched off" in r.content
    assert search.queries == ["a", "b"]  # the third query never hit the backend
    # the tool disappears from the schemas until the next request
    assert "web_search" not in [s["function"]["name"] for s in registry.schemas()]
    assert "not available" in registry.execute("web_search", {"query": "d"}).content
    registry.new_request()
    assert registry.is_enabled("web_search")


def test_fallback_backend():
    class Broken(FakeSearch):
        name = "broken"

        def text(self, q, n):
            raise ToolError("rate limited")

    primary_empty = FallbackBackend(EmptySearch(), FakeSearch())
    assert len(primary_empty.text("q", 2)) == 2
    assert len(FallbackBackend(Broken(), FakeSearch()).text("q", 3)) == 3
    with pytest.raises(ToolError, match="rate limited"):
        FallbackBackend(Broken(), EmptySearch()).text("q", 3)
    assert FallbackBackend(EmptySearch(), EmptySearch()).text("q", 3) == []
    assert primary_empty.name == "fake+fake"


def test_wikipedia_backend_parses_api_response(monkeypatch):
    def handler(request: httpx.Request):
        assert request.url.host == "en.wikipedia.org"
        assert request.url.params["srsearch"] == "heat pumps"
        assert request.headers["user-agent"].startswith("deck-creator-agent/")  # honest, identifiable UA
        return httpx.Response(200, json={"query": {"search": [
            {"title": "Heat pump", "snippet": 'A <span class="searchmatch">heat</span> pump &amp; more'}]}})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    assert WikipediaBackend().text("heat pumps", 3) == [{
        "title": "Heat pump - Wikipedia", "url": "https://en.wikipedia.org/wiki/Heat_pump", "snippet": "A heat pump & more"}]


def test_make_search_backend_variants():
    assert make_search_backend("ddgs").name == "ddgs+wikipedia"
    assert make_search_backend("ddgs", wikipedia_fallback=False).name == "ddgs"
    assert make_search_backend("searxng", searxng_url="http://x").name == "searxng+wikipedia"
    with pytest.raises(ValueError):
        make_search_backend("google")


def test_read_webpage_blocks_private_hosts(setup):
    registry, _, _ = setup
    for url in ("http://localhost:11434/api/tags", "http://127.0.0.1/", "http://192.168.1.10/", "file:///etc/passwd"):
        r = registry.execute("read_webpage", {"url": url})
        assert not r.ok, url


def test_check_url_allows_private_when_enabled():
    check_url("http://127.0.0.1:8080/", allow_private=True)
    with pytest.raises(ToolError):
        check_url("ftp://example.com/")


def test_extract_page_text_html_and_text():
    html = ("<html><head><title>Heat pumps</title></head><body><nav>menu</nav><article><h1>Heat pumps</h1>"
            + "<p>Heat pumps moved 3 units of heat per unit of electricity in typical conditions. </p>" * 20
            + "</article></body></html>").encode()
    title, text = extract_page_text(html, "text/html; charset=utf-8", "https://example.com")
    assert title == "Heat pumps" and "units of heat" in text
    assert extract_page_text(b"plain text", "text/plain", "u") == ("", "plain text")
    with pytest.raises(ToolError):
        extract_page_text(b"\x00\x01", "application/octet-stream", "u")


def test_format_search_results_empty():
    assert "No results" in format_search_results([])


def test_save_image_converts_and_validates(tmp_path, png):
    from PIL import Image

    webp = tmp_path / "x.webp"
    Image.open(png).save(webp, "WEBP")
    out = save_image(str(webp), tmp_path / "assets")
    assert out.suffix == ".png" and out.exists()
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (20, 20)).save(tiny)
    with pytest.raises(ToolError, match="too small"):
        save_image(str(tiny), tmp_path / "assets")
    svg = tmp_path / "x.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    with pytest.raises(ToolError, match="SVG"):
        save_image(str(svg), tmp_path / "assets")
    with pytest.raises(ToolError):
        save_image("http://localhost/x.png", tmp_path / "assets")
