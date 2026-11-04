"""Local web UI: chat with the agent, watch the deck being built, preview it, download it.

Built on Starlette + uvicorn (both BSD-3). The front end is plain HTML/CSS/JS with no CDN,
so the UI works fully offline. The agent runs in a background thread; its progress is
streamed to the browser with Server-Sent Events.

Security: the server binds to 127.0.0.1 by default and has no login. It rejects requests
whose Host header is not local (DNS rebinding) and POSTs from other origins (CSRF), so a
web page you visit cannot drive the agent on your machine.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from deck_agent import __version__
from deck_agent.agent import DeckAgent
from deck_agent.config import Settings
from deck_agent.deck.html_renderer import HtmlRenderer
from deck_agent.deck.model import Deck
from deck_agent.deck.render import PdfExportError, find_soffice, render_deck, safe_filename
from deck_agent.deck.themes import THEMES
from deck_agent.llm import BaseLLM, make_llm
from deck_agent.tools import DeckSession, build_registry
from deck_agent.tools.web import SearchBackend

STATIC_DIR = Path(__file__).parent / "static"
MAX_ATTACHMENT_CHARS = 40_000  # all attachments together; keeps the prompt inside the context window
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MEDIA_TYPES = {
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "pdf": "application/pdf",
    "html": "text/html; charset=utf-8",
    "json": "application/json",
}
_SLIDE_REF = re.compile(r"(?:Added|Updated) slide (\d+)|to position (\d+)")

LLMFactory = Callable[[Settings], BaseLLM]


def default_llm_factory(s: Settings) -> BaseLLM:
    return make_llm(s.backend, s.model, s.base_url, temperature=s.temperature, top_p=s.top_p, top_k=s.top_k,
                    num_ctx=s.num_ctx, think=s.think, timeout=s.timeout, api_key=s.api_key)


# --------------------------------------------------------------------------- event log
class EventLog:
    """Append-only, thread-safe list of UI events with increasing ids (replayed on page reload)."""

    def __init__(self, limit: int = 5000):
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._next_id = 1
        self._limit = limit

    def add(self, type_: str, **data: Any) -> dict[str, Any]:
        with self._lock:
            event = {"id": self._next_id, "type": type_, **data}
            self._next_id += 1
            # Progress ticks are only interesting while they are the latest event.
            if type_ == "progress" and self._events and self._events[-1]["type"] == "progress":
                self._events[-1] = event
            else:
                self._events.append(event)
            del self._events[: max(0, len(self._events) - self._limit)]
            return event

    def since(self, last_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._events if e["id"] > last_id]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    @property
    def last_id(self) -> int:
        with self._lock:
            return self._next_id - 1


def _args_preview(args: dict[str, Any]) -> str:
    parts = []
    for key in ("query", "url", "title", "quote", "slide_number", "filename", "theme"):
        if key in args and args[key] not in (None, ""):
            value = str(args[key])
            parts.append(value if len(value) <= 90 else value[:89] + "…")
    return " · ".join(parts)


def _first_line(text: str, n: int = 240) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= n else line[: n - 1] + "…"


# --------------------------------------------------------------------------- application state
class UIState:
    def __init__(self, settings: Settings, llm_factory: LLMFactory | None = None,
                 search_backend: SearchBackend | None = None, initial_deck: Deck | None = None,
                 deck_name: str = ""):
        self.settings = settings
        self.llm_factory = llm_factory or default_llm_factory
        self.search_backend = search_backend
        self.events = EventLog()
        self.session = DeckSession(output_dir=settings.output_dir, formats=["pptx"], default_theme=settings.theme,
                                   allow_private_urls=settings.allow_private_urls,
                                   verify_numbers=settings.verify_numbers)
        self.agent = DeckAgent(self.llm_factory(settings), self._registry(), self.session, settings,
                               interactive=True, on_event=self._on_agent_event)
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.deck_version = 0
        self._deck_fp = self._fingerprint()
        self._last_progress = 0.0
        self._turn_started = 0.0
        self.preview_dir = Path(tempfile.mkdtemp(prefix="deck-agent-ui-"))
        self._pdf_lock = threading.Lock()
        self._pdf_cache: tuple[int, Path] | None = None
        self._pages_cache: tuple[int, list[Path]] | None = None
        if initial_deck is not None:
            self.session.deck = initial_deck
            # The opened file is the user's own material: its figures and links count as sources.
            self.session.note_source(initial_deck.model_dump_json())
            self.check_deck(focus=1)
            self.events.add("note", text=f"Opened {deck_name or 'deck'} ({len(initial_deck.slides)} slides). "
                                         "Preview it, ask for changes, or download it.")

    # ------------------------------------------------------------------ helpers
    def _registry(self):
        return build_registry(self.settings, self.session, backend=self.search_backend)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def snapshot(self) -> Deck:
        """A consistent copy of the deck (the agent thread may be editing it)."""
        for _ in range(5):
            try:
                return Deck.model_validate_json(self.session.deck.model_dump_json())
            except (RuntimeError, ValueError):
                time.sleep(0.05)
        return Deck.model_validate_json(self.session.deck.model_dump_json())

    def _fingerprint(self) -> str:
        try:
            return hashlib.sha1(self.session.deck.model_dump_json().encode()).hexdigest()
        except RuntimeError:  # list mutated during serialisation; try again on the next event
            return ""

    def deck_info(self) -> dict[str, Any]:
        deck = self.snapshot()
        return {
            "version": self.deck_version,
            "title": deck.title if deck.slides else "",
            "theme": deck.theme,
            "count": len(deck.slides),
            "slides": [{"n": i, "kind": s.kind, "title": s.heading} for i, s in enumerate(deck.slides, 1)],
        }

    def check_deck(self, focus: int | None = None) -> None:
        fp = self._fingerprint()
        if fp and fp != self._deck_fp:
            self._deck_fp = fp
            self.deck_version += 1
            self.events.add("deck", focus=focus, **self.deck_info())

    def state(self) -> dict[str, Any]:
        return {
            "version": __version__,
            "backend": self.settings.backend,
            "model": self.settings.model,
            "web": self.settings.web,
            "verify_numbers": self.settings.verify_numbers,
            "themes": [{"name": t.name, "description": t.description, "accent": t.accent,
                        "background": t.cover_background} for t in THEMES.values()],
            "pdf_available": find_soffice() is not None,
            "running": self.running,
            "output_dir": str(self.settings.output_dir.resolve()),
            "deck": self.deck_info(),
        }

    def models(self) -> list[str]:
        try:
            names = self.agent.llm.list_models()
        except Exception:  # noqa: BLE001 - server may be down; the UI shows the configured model only
            names = []
        if self.settings.model not in names:
            names.insert(0, self.settings.model)
        return names

    # ------------------------------------------------------------------ agent events -> UI events
    def _on_agent_event(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "llm_start":
            self._last_progress = 0.0
            self.events.add("llm_start", step=data["step"])
        elif kind == "llm_progress":
            now = time.time()
            if now - self._last_progress >= 0.5:
                self._last_progress = now
                self.events.add("progress", tokens=data["tokens"], phase=data["phase"],
                                seconds=round(now - self._turn_started))
        elif kind == "llm_error":
            self.events.add("error", text=data["error"])
        elif kind == "tool_call":
            self.events.add("tool_call", name=data["name"], summary=_args_preview(data["arguments"]))
        elif kind == "tool_result":
            out = data["output"]
            summary = _first_line(out) if data["ok"] else out[:600]
            if data["name"] == "read_webpage" and data["ok"]:
                summary = f"read {len(out):,} characters"
            elif data["name"] in ("web_search", "search_images") and data["ok"]:
                summary = f"{out.count('URL:') or out.count('image:')} results"
            self.events.add("tool_result", name=data["name"], ok=data["ok"], summary=summary)
            m = _SLIDE_REF.search(out)
            self.check_deck(focus=int(m.group(1) or m.group(2)) if m else None)
        elif kind == "nudge":
            self.events.add("note", text="The model described a plan instead of building; asked it to continue.")
        elif kind == "autosave":
            self.events.add("saved", files=data["files"])

    # ------------------------------------------------------------------ actions
    def start_chat(self, message: str, attachments: list[dict[str, str]]) -> tuple[bool, str]:
        with self.lock:
            if self.running:
                return False, "The agent is still working on the previous message."
            prompt, names, truncated = message.strip(), [], False
            budget = MAX_ATTACHMENT_CHARS
            for att in attachments:
                text = str(att.get("text", "")).strip()
                name = str(att.get("name", "attachment"))[:120]
                if not text:
                    continue
                if len(text) > budget:
                    text, truncated = text[:budget] + "\n[... truncated]", True
                budget -= len(text)
                names.append(name)
                prompt += f"\n\nSource material from the file '{name}':\n{text}"
                if budget <= 0:
                    break
            if not prompt:
                return False, "Empty message."
            self.events.add("user", text=message.strip(), attachments=names, truncated=truncated)
            self.events.add("status", running=True)
            self._turn_started = time.time()
            self.thread = threading.Thread(target=self._run, args=(prompt,), daemon=True, name="deck-agent")
            self.thread.start()
            return True, ""

    def _run(self, prompt: str) -> None:
        try:
            result = self.agent.run(prompt)
            self.check_deck()
            if result.stopped_reason == "cancelled":
                self.events.add("note", text="Stopped.")
            elif result.text and result.stopped_reason != "llm_error":
                self.events.add("assistant", text=result.text, stopped_reason=result.stopped_reason,
                                tool_calls=result.tool_calls, steps=result.steps)
        except Exception as exc:  # noqa: BLE001 - surface any crash in the chat instead of dying silently
            self.events.add("error", text=f"{exc.__class__.__name__}: {exc}")
        finally:
            self.check_deck()
            self.events.add("status", running=False, seconds=round(time.time() - self._turn_started))

    def stop(self) -> None:
        self.agent.cancel()

    def reset(self) -> None:
        self.agent.reset()
        self.events.clear()
        self._pdf_cache = None
        self.deck_version += 1
        self._deck_fp = self._fingerprint()
        self.events.add("reset", **self.deck_info())

    def update_settings(self, changes: dict[str, Any]) -> None:
        s = self.settings
        if "model" in changes and changes["model"] and changes["model"] != s.model:
            s.model = str(changes["model"])
            old = self.agent.llm
            self.agent.llm = self.llm_factory(s)
            old.close()
        rebuild = False
        if "web" in changes and bool(changes["web"]) != s.web:
            s.web, rebuild = bool(changes["web"]), True
        if "verify_numbers" in changes:
            s.verify_numbers = self.session.verify_numbers = bool(changes["verify_numbers"])
        if rebuild:
            self.agent.set_registry(self._registry())
        self.events.add("settings", model=s.model, web=s.web, verify_numbers=s.verify_numbers)

    def set_theme(self, theme: str) -> None:
        if theme not in THEMES:
            raise ValueError(f"unknown theme {theme!r}")
        self.session.deck.theme = theme
        self.session.dirty = True
        self.check_deck()

    # ------------------------------------------------------------------ rendering
    def preview_pdf(self) -> Path:
        with self._pdf_lock:
            version = self.deck_version
            if self._pdf_cache and self._pdf_cache[0] == version and self._pdf_cache[1].exists():
                return self._pdf_cache[1]
            files = render_deck(self.snapshot(), self.preview_dir, f"preview_{version}", ["pdf"])
            self._pdf_cache = (version, files["pdf"])
            return files["pdf"]

    def preview_pages(self) -> tuple[int, list[Path]]:
        """Slide images of the real PPTX (PPTX -> PDF via LibreOffice -> PNG via PDFium).

        Images display in every browser, unlike PDFs in iframes (no viewer in headless
        browsers or when the browser is set to download PDFs).
        """
        import pypdfium2 as pdfium

        pdf = self.preview_pdf()
        with self._pdf_lock:
            version = self.deck_version
            if self._pages_cache and self._pages_cache[0] == version and all(p.exists() for p in self._pages_cache[1]):
                return self._pages_cache
            out_dir = self.preview_dir / f"pages_{version}"
            out_dir.mkdir(parents=True, exist_ok=True)
            doc = pdfium.PdfDocument(str(pdf))
            pages = []
            try:
                for i, page in enumerate(doc, 1):
                    path = out_dir / f"{i}.png"
                    page.render(scale=1600 / page.get_width()).to_pil().save(path, optimize=True)
                    pages.append(path)
            finally:
                doc.close()
            self._pages_cache = (version, pages)
            return self._pages_cache

    def download(self, fmt: str) -> tuple[Path, str]:
        deck = self.snapshot()
        base = safe_filename(deck.title, default="deck")
        if fmt == "json":
            path = deck.save_json(self.settings.output_dir / f"{base}.deck.json")
        else:
            path = render_deck(deck, self.settings.output_dir, base, [fmt])[fmt]
        return path, f"{base}.deck.json" if fmt == "json" else f"{base}.{fmt}"


# --------------------------------------------------------------------------- security middleware
class SameOriginPostMiddleware:
    """Reject state-changing requests that come from another origin (CSRF protection)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            origin, host = headers.get("origin"), headers.get("host", "")
            if origin and urlparse(origin).netloc != host:
                await JSONResponse({"error": "cross-origin request refused"}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


# --------------------------------------------------------------------------- routes
def create_app(settings: Settings, *, llm_factory: LLMFactory | None = None,
               search_backend: SearchBackend | None = None, allowed_hosts: list[str] | None = None,
               initial_deck: Deck | None = None, deck_name: str = "") -> Starlette:
    state = UIState(settings, llm_factory, search_backend, initial_deck=initial_deck, deck_name=deck_name)

    async def json_body(request: Request) -> dict[str, Any]:
        if "application/json" not in request.headers.get("content-type", ""):
            raise ValueError("expected a JSON body")
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def error(message: str, status: int = 400) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    async def index(_: Request) -> Response:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    async def get_state(_: Request) -> Response:
        data = state.state()
        data["models"] = await run_in_threadpool(state.models)
        return JSONResponse(data)

    async def chat(request: Request) -> Response:
        try:
            body = await json_body(request)
        except ValueError as exc:
            return error(str(exc))
        attachments = body.get("attachments") or []
        if not isinstance(attachments, list):
            return error("attachments must be a list")
        message = str(body.get("message", ""))
        if not message.strip() and not any(str(a.get("text", "")).strip() for a in attachments if isinstance(a, dict)):
            return error("Empty message.")
        ok, message = state.start_chat(message, [a for a in attachments if isinstance(a, dict)])
        return JSONResponse({"ok": True}, status_code=202) if ok else error(message, 409)

    async def stop(_: Request) -> Response:
        state.stop()
        return JSONResponse({"ok": True})

    async def reset(_: Request) -> Response:
        if state.running:
            return error("Stop the agent before starting a new deck.", 409)
        state.reset()
        return JSONResponse({"ok": True})

    async def update_settings(request: Request) -> Response:
        if state.running:
            return error("Settings can be changed when the agent is idle.", 409)
        try:
            await run_in_threadpool(state.update_settings, await json_body(request))
        except ValueError as exc:
            return error(str(exc))
        return JSONResponse({"ok": True})

    async def set_theme(request: Request) -> Response:
        try:
            state.set_theme(str((await json_body(request)).get("theme", "")))
        except ValueError as exc:
            return error(str(exc))
        return JSONResponse({"ok": True})

    async def events(request: Request) -> Response:
        try:
            last = int(request.headers.get("last-event-id") or request.query_params.get("since") or 0)
        except ValueError:
            last = 0

        async def stream():
            nonlocal last
            yield "retry: 2000\n\n"
            idle = 0.0
            while not await request.is_disconnected():
                batch = state.events.since(last)
                for event in batch:
                    last = event["id"]
                    yield f"id: {event['id']}\ndata: {json.dumps(event, default=str)}\n\n"
                if batch:
                    idle = 0.0
                    continue
                await asyncio.sleep(0.2)
                idle += 0.2
                if idle >= 15:
                    idle = 0.0
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def events_json(request: Request) -> Response:
        since = int(request.query_params.get("since", 0))
        return JSONResponse({"events": state.events.since(since), "running": state.running})

    async def preview_html(_: Request) -> Response:
        deck = state.snapshot()
        if not deck.slides:
            return HTMLResponse(_EMPTY_PREVIEW, headers={"Cache-Control": "no-store"})
        html = await run_in_threadpool(lambda: HtmlRenderer(deck).to_html())
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    async def preview_pdf(_: Request) -> Response:
        if not state.session.deck.slides:
            return error("The deck has no slides yet.", 404)
        try:
            path = await run_in_threadpool(state.preview_pdf)
        except PdfExportError as exc:
            return error(str(exc), 501)
        return FileResponse(path, media_type="application/pdf", content_disposition_type="inline",
                            filename="preview.pdf", headers={"Cache-Control": "no-store"})

    async def preview_pages(_: Request) -> Response:
        if not state.session.deck.slides:
            return error("The deck has no slides yet.", 404)
        try:
            version, pages = await run_in_threadpool(state.preview_pages)
        except PdfExportError as exc:
            return error(str(exc), 501)
        return JSONResponse({"version": version, "count": len(pages)})

    async def preview_page(request: Request) -> Response:
        cache = state._pages_cache
        n = request.path_params["n"]
        if not cache or not 1 <= n <= len(cache[1]):
            return error("Render the pages first (GET /api/preview/pages).", 404)
        return FileResponse(cache[1][n - 1], media_type="image/png", headers={"Cache-Control": "no-store"})

    async def download(request: Request) -> Response:
        fmt = request.path_params["fmt"]
        if fmt not in MEDIA_TYPES:
            return error(f"unknown format {fmt!r}", 404)
        if not state.session.deck.slides:
            return error("The deck has no slides yet.", 404)
        try:
            path, filename = await run_in_threadpool(state.download, fmt)
        except PdfExportError as exc:
            return error(str(exc), 501)
        return FileResponse(path, media_type=MEDIA_TYPES[fmt], filename=filename)

    async def extract(request: Request) -> Response:
        """Turn an attached file (PDF or text) into plain text for the prompt."""
        name = request.query_params.get("name", "file")
        data = b""
        async for chunk in request.stream():
            data += chunk
            if len(data) > MAX_UPLOAD_BYTES:
                return error("File is larger than 20 MB.", 413)
        if name.lower().endswith(".pdf") or data[:5] == b"%PDF-":
            from pypdf import PdfReader

            try:
                reader = await run_in_threadpool(PdfReader, io.BytesIO(data))
                text = "\n".join((p.extract_text() or "") for p in reader.pages[:60])
            except Exception as exc:  # noqa: BLE001 - malformed PDFs are common
                return error(f"Could not read the PDF ({exc.__class__.__name__}).", 422)
        else:
            text = data.decode("utf-8", errors="replace")
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text:
            return error("No text found in the file (scanned PDFs need OCR first).", 422)
        return JSONResponse({"name": name, "text": text, "chars": len(text)})

    routes = [
        Route("/", index),
        Route("/api/state", get_state),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/stop", stop, methods=["POST"]),
        Route("/api/reset", reset, methods=["POST"]),
        Route("/api/settings", update_settings, methods=["POST"]),
        Route("/api/theme", set_theme, methods=["POST"]),
        Route("/api/events", events),
        Route("/api/events.json", events_json),
        Route("/api/preview.html", preview_html),
        Route("/api/preview.pdf", preview_pdf),
        Route("/api/preview/pages", preview_pages),
        Route("/api/preview/page/{n:int}.png", preview_page),
        Route("/api/download/{fmt}", download),
        Route("/api/extract", extract, methods=["POST"]),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ]
    hosts = allowed_hosts or ["127.0.0.1", "localhost", "[::1]", "::1"]
    middleware = [Middleware(TrustedHostMiddleware, allowed_hosts=hosts), Middleware(SameOriginPostMiddleware)]
    app = Starlette(routes=routes, middleware=middleware)
    app.state.ui = state
    return app


_EMPTY_PREVIEW = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;display:flex;align-items:center;justify-content:center;
font:15px system-ui,sans-serif;color:#8a94a6;background:transparent}
</style></head><body><div>Slides will appear here as the agent builds them.</div></body></html>"""


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 7860, open_browser: bool = False,
          deck_path: Path | None = None) -> None:
    import uvicorn

    allowed = None
    if host not in ("127.0.0.1", "localhost", "::1"):
        allowed = ["*"] if host in ("0.0.0.0", "::") else [host, "127.0.0.1", "localhost"]
    deck = Deck.load_json(deck_path) if deck_path else None
    app = create_app(settings, allowed_hosts=allowed, initial_deck=deck, deck_name=deck_path.name if deck_path else "")
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    print(f"deck-agent UI running at {url}  (model: {settings.model}; press Ctrl+C to stop)")
    if allowed:
        print("WARNING: the UI has no login. Anyone who can reach this address can use the agent.")
    if open_browser:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
