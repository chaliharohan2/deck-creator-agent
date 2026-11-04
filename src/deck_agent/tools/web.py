"""Web research tools: search, page reading and Creative-Commons image search.

Everything here is open source and needs no API key:

* ``ddgs`` (MIT) - metasearch over DuckDuckGo, Bing, Brave, Mojeek, Wikipedia, ...
* SearXNG (AGPL-3.0, self-hosted) - optional fully self-hosted metasearch engine.
* ``trafilatura`` (Apache-2.0) - main-content extraction from HTML.
* ``pypdf`` (BSD-3) - text extraction when a result is a PDF.

Note: open-source *clients* still query public search engines; no open-source
project offers a fresh, full-web index you can run locally. See docs/LICENSES.md.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import io
import ipaddress
import os
import re
import socket
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlparse

import httpx
from PIL import Image

from deck_agent.tools.base import ToolError

# An honest, identifiable User-Agent (as e.g. Wikimedia's API policy requires). Sites that
# refuse bots will refuse this; the agent then simply uses a different source.
USER_AGENT = os.environ.get(
    "DECK_AGENT_USER_AGENT",
    "deck-creator-agent/0.1 (+https://github.com/chaliharohan2/deck-creator-agent)",
)
MAX_PAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024
IMAGE_MAX_SIDE = 2400


# --------------------------------------------------------------------------- safety
def check_url(url: str, allow_private: bool = False) -> None:
    """Reject non-HTTP(S) URLs and (by default) hosts on private/loopback networks.

    This keeps a prompt-injected web page from steering the agent into reading
    services on your machine or LAN.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError(f"only http(s) URLs are supported, got {url!r}")
    host = parsed.hostname
    if not host:
        raise ToolError(f"invalid URL {url!r}")
    if allow_private:
        return
    if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        raise ToolError(f"refusing to fetch private host {host!r}")
    try:
        addrs = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addrs = [ipaddress.ip_address(info[4][0].split("%")[0]) for info in socket.getaddrinfo(host, None)]
        except (socket.gaierror, UnicodeError):
            return  # cannot resolve locally (e.g. behind a proxy); let the HTTP client decide
    for ip in addrs:
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ToolError(f"refusing to fetch {host!r}: it resolves to a private address")


def _http_client(allow_private: bool, timeout: float = 20.0) -> httpx.Client:
    def guard(request: httpx.Request) -> None:  # runs for every redirect hop too
        check_url(str(request.url), allow_private)

    return httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        },
        event_hooks={"request": [guard]},
    )


def _read_capped(response: httpx.Response, limit: int) -> bytes:
    chunks, size = [], 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > limit:
            raise ToolError(f"response is larger than {limit // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)


# --------------------------------------------------------------------------- search backends
class SearchBackend(Protocol):
    name: str

    def text(self, query: str, max_results: int) -> list[dict[str, str]]: ...

    def images(self, query: str, max_results: int) -> list[dict[str, Any]]: ...


def _env_proxy() -> str | None:
    return (os.environ.get("DECK_AGENT_PROXY") or os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy") or None)


def _env_ca_bundle() -> str | bool:
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        path = os.environ.get(var)
        if path and Path(path).is_file():
            return path
    return True


class DDGSBackend:
    """Keyless metasearch via the ``ddgs`` library (MIT)."""

    name = "ddgs"

    def __init__(self, region: str = "us-en", safesearch: str = "moderate", image_license: str = "Share",
                 timeout: int = 15):
        self.region = region
        self.safesearch = safesearch
        self.image_license = None if image_license.lower() == "any" else image_license
        self.timeout = timeout

    def _client(self):
        from ddgs import DDGS  # imported lazily: optional at import time for tests

        return DDGS(proxy=_env_proxy(), timeout=self.timeout, verify=_env_ca_bundle())

    def text(self, query: str, max_results: int) -> list[dict[str, str]]:
        from ddgs.exceptions import DDGSException

        try:
            rows = self._client().text(query, region=self.region, safesearch=self.safesearch, max_results=max_results)
        except DDGSException as exc:
            if "No results" in str(exc):
                return []
            raise ToolError(f"search failed ({exc}); try again with a different query") from exc
        return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in rows]

    def images(self, query: str, max_results: int) -> list[dict[str, Any]]:
        from ddgs.exceptions import DDGSException

        try:
            rows = self._client().images(query, region=self.region, safesearch=self.safesearch,
                                         max_results=max_results, license_image=self.image_license)
        except DDGSException as exc:
            if "No results" in str(exc):
                return []
            raise ToolError(f"image search failed ({exc})") from exc
        return [
            {"title": r.get("title", ""), "image_url": r.get("image", ""), "page_url": r.get("url", ""),
             "width": r.get("width"), "height": r.get("height"), "source": r.get("source", ""),
             "license": self.image_license or "unknown"}
            for r in rows
        ]


class SearxngBackend:
    """Self-hosted SearXNG instance (enable the ``json`` format in its settings.yml)."""

    name = "searxng"

    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _search(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as c:
                r = c.get(f"{self.base_url}/search", params={**params, "format": "json"})
        except httpx.HTTPError as exc:
            raise ToolError(f"SearXNG at {self.base_url} is not reachable ({exc.__class__.__name__})") from exc
        if r.status_code == 403:
            raise ToolError("SearXNG refused JSON output: add `json` to `search.formats` in settings.yml")
        r.raise_for_status()
        return r.json().get("results", [])

    def text(self, query: str, max_results: int) -> list[dict[str, str]]:
        rows = self._search({"q": query, "safesearch": 1})[:max_results]
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")} for r in rows]

    def images(self, query: str, max_results: int) -> list[dict[str, Any]]:
        rows = self._search({"q": query, "categories": "images", "safesearch": 1})[:max_results]
        out = []
        for r in rows:
            w = h = None
            m = re.match(r"(\d+)\s*x\s*(\d+)", str(r.get("resolution", "")))
            if m:
                w, h = int(m.group(1)), int(m.group(2))
            out.append({"title": r.get("title", ""), "image_url": r.get("img_src", ""), "page_url": r.get("url", ""),
                        "width": w, "height": h, "source": r.get("engine", ""), "license": "unknown - check page"})
        return out


class WikipediaBackend:
    """Wikipedia full-text search through the official MediaWiki API (free, no key, no scraping)."""

    name = "wikipedia"

    def __init__(self, lang: str = "en", timeout: float = 15.0):
        self.lang = lang
        self.timeout = timeout

    def text(self, query: str, max_results: int) -> list[dict[str, str]]:
        params = {"action": "query", "list": "search", "srsearch": query, "srlimit": max_results,
                  "srprop": "snippet", "format": "json", "utf8": 1}
        try:
            with httpx.Client(timeout=self.timeout, headers={"User-Agent": USER_AGENT}) as c:
                r = c.get(f"https://{self.lang}.wikipedia.org/w/api.php", params=params)
        except httpx.HTTPError as exc:
            raise ToolError(f"Wikipedia search failed ({exc.__class__.__name__})") from exc
        if r.status_code != 200:
            raise ToolError(f"Wikipedia search failed (HTTP {r.status_code})")
        rows = r.json().get("query", {}).get("search", [])
        return [
            {"title": f"{row['title']} - Wikipedia",
             "url": f"https://{self.lang}.wikipedia.org/wiki/{quote(row['title'].replace(' ', '_'))}",
             "snippet": html_lib.unescape(re.sub(r"<[^>]+>", "", row.get("snippet", "")))}
            for row in rows
        ]

    def images(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return []


class FallbackBackend:
    """Use ``primary``; when it returns nothing or fails, ask ``fallback`` instead.

    Key-less metasearch is regularly rate-limited by the upstream engines; the
    fallback keeps the agent supplied with citable material when that happens.
    """

    def __init__(self, primary: SearchBackend, fallback: SearchBackend):
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}+{fallback.name}"

    def text(self, query: str, max_results: int) -> list[dict[str, str]]:
        try:
            rows = self.primary.text(query, max_results)
            primary_error = None
        except ToolError as exc:
            rows, primary_error = [], exc
        if rows:
            return rows
        try:
            rows = self.fallback.text(query, max_results)
        except ToolError:
            rows = []
        if not rows and primary_error:
            raise primary_error
        return rows

    def images(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return self.primary.images(query, max_results)


def make_search_backend(name: str, *, searxng_url: str = "", region: str = "us-en",
                        image_license: str = "Share", wikipedia_fallback: bool = True) -> SearchBackend:
    if name == "searxng":
        backend: SearchBackend = SearxngBackend(searxng_url)
    elif name == "ddgs":
        backend = DDGSBackend(region=region, image_license=image_license)
    elif name == "wikipedia":
        return WikipediaBackend()
    else:
        raise ValueError(f"unknown search backend {name!r}; use 'ddgs', 'searxng' or 'wikipedia'")
    return FallbackBackend(backend, WikipediaBackend()) if wikipedia_fallback else backend


# --------------------------------------------------------------------------- formatting / fetching
def format_search_results(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No results. Try one shorter, broader query (search engines may also be rate-limiting)."
    lines = []
    for i, r in enumerate(rows, 1):
        snippet = re.sub(r"\s+", " ", r.get("snippet", "")).strip()
        if len(snippet) > 280:
            snippet = snippet[:279] + "…"
        lines.append(f"{i}. {r.get('title', '').strip()}\n   URL: {r.get('url', '')}\n   {snippet}")
    return "\n".join(lines)


def _html_title(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    return re.sub(r"\s+", " ", m.group(1)).strip()[:200] if m else ""


def extract_page_text(data: bytes, content_type: str, url: str) -> tuple[str, str]:
    """Return ``(title, text)`` for an HTML/PDF/text payload."""
    ctype = content_type.lower()
    if "pdf" in ctype or data[:5] == b"%PDF-":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [p.extract_text() or "" for p in reader.pages[:40]]
        except Exception as exc:  # noqa: BLE001 - malformed PDFs are common
            raise ToolError(f"could not read PDF ({exc.__class__.__name__})") from exc
        title = (reader.metadata.title if reader.metadata and reader.metadata.title else "") or ""
        return str(title), re.sub(r"[ \t]+", " ", "\n".join(pages)).strip()
    text_types = ("html", "xml", "text/plain", "json", "markdown")
    if ctype and not any(t in ctype for t in text_types):
        raise ToolError(f"unsupported content type {content_type!r}")
    html = data.decode("utf-8", errors="replace")
    if "text/plain" in ctype or "markdown" in ctype or "json" in ctype:
        return "", html.strip()
    import trafilatura

    text = trafilatura.extract(html, url=url, include_tables=True, include_comments=False,
                               output_format="markdown", favor_precision=True)
    if not text:
        text = trafilatura.extract(html, url=url, include_tables=True, include_comments=False,
                                   output_format="markdown", favor_recall=True)
    if not text:  # last resort: crude tag stripping
        body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()
    return _html_title(html), text.strip()


def fetch_page(url: str, max_chars: int = 6000, allow_private: bool = False) -> str:
    check_url(url, allow_private)
    try:
        with _http_client(allow_private) as client, client.stream("GET", url) as r:
            if r.status_code >= 400:
                raise ToolError(f"HTTP {r.status_code} for {url}; try another result")
            data = _read_capped(r, MAX_PAGE_BYTES)
            ctype = r.headers.get("content-type", "")
            final_url = str(r.url)
    except httpx.HTTPError as exc:
        raise ToolError(f"could not fetch {url} ({exc.__class__.__name__}); try another result") from exc
    title, text = extract_page_text(data, ctype, final_url)
    if len(text) < 200:
        raise ToolError("the page has little readable text (it may need JavaScript or block bots); try another result")
    total = len(text)
    if total > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        text = text[: cut if cut > max_chars * 0.7 else max_chars] + f"\n\n[... truncated, {total - max_chars:,} more characters]"
    header = f"Title: {title}\nURL: {final_url}\n\n" if title else f"URL: {final_url}\n\n"
    return header + text


def format_image_results(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No images found. Try a simpler query, or continue without an image."
    lines = []
    for i, r in enumerate(rows, 1):
        size = f"{r['width']}x{r['height']}" if r.get("width") and r.get("height") else "unknown size"
        lines.append(
            f"{i}. {r.get('title', '').strip()[:100]}\n   image: {r.get('image_url', '')}\n"
            f"   page: {r.get('page_url', '')}  ({size}, license filter: {r.get('license')})"
        )
    lines.append("Pass an `image` URL to add_image_slide and put the page URL in `credit` or `sources`.")
    return "\n".join(lines)


def save_image(src: str, asset_dir: Path, allow_private: bool = False) -> Path:
    """Download (or copy) an image, validate it with Pillow and store it as PNG/JPEG."""
    src = src.strip()
    if re.match(r"^https?://", src):
        check_url(src, allow_private)
        try:
            with _http_client(allow_private, timeout=30) as client, client.stream("GET", src) as r:
                if r.status_code >= 400:
                    raise ToolError(f"HTTP {r.status_code} when downloading the image; pick another image")
                data = _read_capped(r, MAX_IMAGE_BYTES)
        except httpx.HTTPError as exc:
            raise ToolError(f"could not download image ({exc.__class__.__name__}); pick another image") from exc
    else:
        path = Path(src).expanduser()
        if not path.is_file():
            raise ToolError(f"image file not found: {src}")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ToolError("image file is larger than 15 MB")
        data = path.read_bytes()
    if data.lstrip()[:5].lower() in (b"<svg ", b"<?xml"):
        raise ToolError("SVG images are not supported; choose a PNG or JPEG image")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # noqa: BLE001
        raise ToolError("the file is not a readable image; pick another one") from exc
    if min(image.size) < 120:
        raise ToolError(f"image is too small ({image.size[0]}x{image.size[1]}); pick a larger one")
    if getattr(image, "is_animated", False):
        image.seek(0)
    if max(image.size) > IMAGE_MAX_SIDE:
        image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
    asset_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(data).hexdigest()[:12]
    if image.format == "JPEG" and image.mode in ("RGB", "L"):
        out = asset_dir / f"img_{digest}.jpg"
        image.save(out, "JPEG", quality=90)
    else:
        if image.mode not in ("RGB", "RGBA", "L", "LA"):
            image = image.convert("RGBA")
        out = asset_dir / f"img_{digest}.png"
        image.save(out, "PNG", optimize=True)
    return out
