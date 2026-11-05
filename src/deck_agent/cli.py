"""Command-line interface: ``deck-agent run|chat|render|sample|themes|doctor``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from deck_agent import __version__
from deck_agent.config import Settings
from deck_agent.deck.model import Deck
from deck_agent.deck.render import FORMATS, PdfExportError, find_soffice, render_deck
from deck_agent.deck.themes import THEMES

console = Console()


# --------------------------------------------------------------------------- argument parsing
def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model server")
    g.add_argument("--backend", choices=["ollama", "openai"], help="ollama (default) or any OpenAI-compatible "
                   "local server: llama.cpp llama-server, vLLM, SGLang, LocalAI")
    g.add_argument("--base-url", help="server URL (default: http://localhost:11434 for ollama, "
                   "http://localhost:8080/v1 for openai)")
    g.add_argument("-m", "--model", help="model name/tag (default: qwen3.5:9b)")
    g.add_argument("--api-key", help="API key, only if your local server requires one")
    g.add_argument("--num-ctx", type=int, help="context window in tokens (ollama only; default 32768)")
    g.add_argument("--temperature", type=float)
    g.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                   help="enable the model's reasoning mode (slower on CPU, sometimes better plans)")
    g.add_argument("--timeout", type=float, help="seconds to wait for one model response (default 900)")


def _add_agent_args(p: argparse.ArgumentParser) -> None:
    _add_model_args(p)
    g = p.add_argument_group("agent")
    g.add_argument("-o", "--out-dir", type=Path, help="output directory (default: ./decks)")
    g.add_argument("-f", "--formats", nargs="+", choices=FORMATS, help="output formats (default: pptx)")
    g.add_argument("-t", "--theme", choices=list(THEMES), help="preferred theme")
    g.add_argument("--slides", type=int, help="default number of slides when the request does not say")
    g.add_argument("--max-steps", type=int, help="maximum model turns per request (default 60)")
    g.add_argument("--no-web", action="store_true", help="disable web research tools (fully offline)")
    g.add_argument("--no-fact-check", action="store_true",
                   help="allow numbers that do not appear in any source (e.g. for illustrative decks)")
    g.add_argument("--search-backend", choices=["ddgs", "searxng", "wikipedia"],
                   help="web search provider (default ddgs, with Wikipedia as fallback)")
    g.add_argument("--searxng-url", help="URL of your SearXNG instance")
    g.add_argument("-v", "--verbose", action="store_true", help="show full tool outputs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deck-agent",
        description="A local, open-source AI agent that researches topics and builds presentation decks.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="build a deck from a single request (autonomous)")
    p.add_argument("request", nargs="+", help='what to build, e.g. "10 slides on heat pumps for homeowners"')
    p.add_argument("--request-file", type=Path, help="append the contents of a text/markdown file to the request")
    _add_agent_args(p)

    p = sub.add_parser("chat", help="interactive session: build a deck and refine it over several turns")
    _add_agent_args(p)

    p = sub.add_parser("ui", help="open the browser UI: chat, live deck preview and downloads")
    _add_agent_args(p)
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1: this computer only)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--open", action="store_true", help="open the UI in your web browser")
    p.add_argument("--deck", type=Path, help="open a saved .deck.json to preview, edit or download it")

    p = sub.add_parser("render", help="re-render a saved .deck.json without the LLM")
    p.add_argument("spec", type=Path)
    p.add_argument("-o", "--out-dir", type=Path)
    p.add_argument("-f", "--formats", nargs="+", choices=FORMATS, default=["pptx"])
    p.add_argument("-t", "--theme", choices=list(THEMES), help="override the deck's theme")

    p = sub.add_parser("sample", help="render a built-in sample deck (no LLM needed)")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("decks"))
    p.add_argument("-f", "--formats", nargs="+", choices=FORMATS, default=["pptx", "html"])
    p.add_argument("-t", "--theme", choices=list(THEMES), default="light")

    sub.add_parser("themes", help="list available themes")

    p = sub.add_parser("doctor", help="check the model server, tool calling, web search and PDF export")
    _add_model_args(p)
    p.add_argument("--skip-llm-test", action="store_true", help="do not send a test prompt to the model")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    mapping = {
        "backend": "backend", "base_url": "base_url", "model": "model", "api_key": "api_key",
        "num_ctx": "num_ctx", "temperature": "temperature", "think": "think", "timeout": "timeout",
        "out_dir": "output_dir", "formats": "formats", "theme": "theme", "slides": "default_slides",
        "max_steps": "max_steps", "search_backend": "search_backend", "searxng_url": "searxng_url",
    }
    for arg_name, attr in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(s, attr, value)
    if getattr(args, "no_web", False):
        s.web = False
    if getattr(args, "no_fact_check", False):
        s.verify_numbers = False
    return s


# --------------------------------------------------------------------------- agent wiring
def _make_llm(s: Settings):
    from deck_agent.llm import make_llm

    return make_llm(s.backend, s.model, s.base_url, temperature=s.temperature, top_p=s.top_p, top_k=s.top_k,
                    num_ctx=s.num_ctx, think=s.think, timeout=s.timeout, api_key=s.api_key)


class ConsoleReporter:
    """Prints agent events and shows a live status line while the model writes."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.status = None
        self.t0 = 0.0

    def stop(self) -> None:
        """Stop the live status line (e.g. after Ctrl+C) so it does not draw over the prompt."""
        if self.status:
            self.status.stop()
            self.status = None

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "llm_start":
            self.t0 = time.time()
            self.status = console.status(f"[cyan]Model is working (step {data['step']})…", spinner="dots")
            self.status.start()
        elif kind == "llm_progress" and self.status:
            verb = "thinking" if data["phase"] == "thinking" else "writing"
            self.status.update(f"[cyan]Model {verb}… {data['tokens']} tokens, {time.time() - self.t0:.0f}s")
        elif kind in ("llm_end", "llm_error"):
            self.stop()
            if kind == "llm_error":
                console.print(f"[red]✗ {escape(data['error'])}")
            elif self.verbose:
                console.print(f"[dim]  step {data['step']}: {data['prompt_tokens']} prompt + "
                              f"{data['completion_tokens']} output tokens in {time.time() - self.t0:.1f}s"
                              + (" (tool call recovered from text)" if data.get("recovered") else ""))
        elif kind == "tool_call":
            args = ", ".join(f"{k}={_preview(v)}" for k, v in data["arguments"].items()
                             if k not in ("notes", "sources") or self.verbose)
            console.print(f"[bold magenta]→ {data['name']}[/]([dim]{escape(args)}[/])")
        elif kind == "tool_result":
            out = data["output"]
            first = out.strip().splitlines()[0] if out.strip() else ""
            if data["ok"]:
                if self.verbose:
                    console.print(Panel(escape(out[:3000]), border_style="dim", expand=False))
                else:
                    summary = first if data["name"] not in ("web_search", "search_images") else \
                        f"{out.count('URL:') or out.count('image:')} results"
                    if data["name"] == "read_webpage":
                        summary = f"read {len(out):,} characters"
                    console.print(f"  [green]✓[/] [dim]{escape(summary[:160])}[/]")
            else:
                console.print(f"  [yellow]✗ {escape(out[:600])}[/]")
        elif kind == "nudge":
            console.print("[dim]  (model replied without using tools; asking it to continue building)[/]")
        elif kind == "trimmed" and self.verbose:
            console.print(f"[dim]  (trimmed old tool output, ~{data['tokens']} tokens in context)[/]")
        elif kind == "autosave":
            console.print(f"[dim]  auto-saved deck{escape(data.get('note', ''))}[/]")
        elif kind == "autosave_error":
            console.print(f"[red]  auto-save failed: {escape(data['error'])}[/]")


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return repr(text[:60] + "…") if len(text) > 60 else repr(text) if isinstance(value, str) else text[:60]


def _build_agent(s: Settings, interactive: bool, verbose: bool):
    from deck_agent.agent import DeckAgent
    from deck_agent.tools import DeckSession, build_registry

    session = DeckSession(output_dir=s.output_dir, formats=s.formats, default_theme=s.theme,
                          theme_locked=bool(s.theme), allow_private_urls=s.allow_private_urls,
                          verify_numbers=s.verify_numbers)
    registry = build_registry(s, session)
    llm = _make_llm(s)
    agent = DeckAgent(llm, registry, session, s, interactive=interactive, on_event=ConsoleReporter(verbose))
    return agent


def _print_files(files: dict[str, Path]) -> None:
    if not files:
        return
    table = Table(show_header=False, box=None, padding=(0, 2))
    for kind, path in files.items():
        label = "spec (editable)" if kind == "json" else kind
        table.add_row(f"[bold]{label}", str(path))
    console.print(Panel(table, title="Saved files", border_style="green", expand=False))


def _header(s: Settings) -> None:
    web = f"web: {s.search_backend}" if s.web else "web: off"
    console.print(f"[bold]deck-agent[/] {__version__} · model [cyan]{s.model}[/] via {s.backend} · {web} · "
                  f"output → {s.output_dir}/ ({', '.join(s.formats)})")


# --------------------------------------------------------------------------- commands
def cmd_run(args: argparse.Namespace) -> int:
    s = settings_from_args(args)
    request = " ".join(args.request)
    if args.request_file:
        request += "\n\nSource material:\n" + args.request_file.read_text(encoding="utf-8")
    _header(s)
    agent = _build_agent(s, interactive=False, verbose=args.verbose)
    t0 = time.time()
    try:
        result = agent.run(request)  # on Ctrl+C the deck built so far is auto-saved before this raises
    except BaseException:
        agent.on_event.stop()
        _print_files(agent.session.saved)
        agent.llm.close()
        raise
    console.print()
    if result.text:
        console.print(Panel(escape(result.text), title="Agent", border_style="cyan"))
    _print_files(result.files)
    console.print(f"[dim]{len(agent.session.deck.slides)} slides · {result.steps} model turns · "
                  f"{result.tool_calls} tool calls ({result.errors} corrected) · {time.time() - t0:.0f}s[/]")
    agent.llm.close()
    # A model failure is a failed run even though the partial deck was saved (scripts check this).
    return 0 if result.files and result.stopped_reason != "llm_error" else 1


CHAT_HELP = """Commands: /slides (outline) · /save [pptx pdf html] · /new (start over) · /help · /quit
Anything else is sent to the agent, e.g. "make slide 4 a chart" or "add a slide about risks after slide 6"."""


def cmd_chat(args: argparse.Namespace) -> int:
    try:
        import readline  # noqa: F401 - enables line editing/history where available
    except ImportError:
        pass
    s = settings_from_args(args)
    _header(s)
    console.print(f"[dim]{escape(CHAT_HELP)}[/]")
    agent = _build_agent(s, interactive=True, verbose=args.verbose)
    while True:
        try:
            line = console.input("\n[bold green]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/help":
            console.print(escape(CHAT_HELP))
            continue
        if line == "/slides":
            console.print(escape(agent.session.deck.outline()))
            continue
        if line == "/new":
            agent.reset()
            console.print("[dim]Started a new deck.[/]")
            continue
        if line.startswith("/save"):
            fmts = line.split()[1:] or None
            try:
                paths, note = agent.session.save(formats=fmts)
                _print_files(paths)
                if note:
                    console.print(f"[yellow]{escape(note)}")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]{escape(str(exc))}")
            continue
        try:
            result = agent.run(line)
        except KeyboardInterrupt:
            agent.on_event.stop()
            console.print("[yellow]Interrupted.[/]")
            continue
        if result.text:
            console.print(Panel(escape(result.text), title="Agent", border_style="cyan"))
        _print_files(result.files)
    agent.llm.close()
    return 0


def _load_deck(path: Path) -> Deck | None:
    """Load a .deck.json, printing a readable error instead of a traceback."""
    try:
        return Deck.load_json(path)
    except FileNotFoundError:
        console.print(f"[red]No such file: {escape(str(path))}")
    except OSError as exc:
        console.print(f"[red]Cannot read {escape(str(path))}: {escape(str(exc))}")
    except ValueError as exc:  # invalid JSON, non-UTF-8 text, or a spec that fails validation
        console.print(f"[red]{escape(str(path))} is not a valid deck spec:[/]\n{escape(str(exc)[:1500])}")
    return None


def cmd_ui(args: argparse.Namespace) -> int:
    from deck_agent.web import serve

    if args.deck and _load_deck(args.deck) is None:
        return 1
    serve(settings_from_args(args), host=args.host, port=args.port, open_browser=args.open, deck_path=args.deck)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    deck = _load_deck(args.spec)
    if deck is None:
        return 1
    if args.theme:
        deck.theme = args.theme
    out_dir = args.out_dir or args.spec.parent
    base = args.spec.name.removesuffix(".deck.json").removesuffix(".json")
    try:
        files = render_deck(deck, out_dir, base, args.formats)
    except PdfExportError as exc:
        console.print(f"[red]{escape(str(exc))}")
        return 1
    _print_files(files)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from deck_agent.sample import build_sample_deck

    deck = build_sample_deck(args.theme, args.out_dir / "assets")
    try:
        files = render_deck(deck, args.out_dir, f"sample_{args.theme}", args.formats)
    except PdfExportError as exc:
        console.print(f"[red]{escape(str(exc))}")
        return 1
    _print_files(files)
    return 0


def cmd_themes(_: argparse.Namespace) -> int:
    table = Table(title="Themes")
    table.add_column("name", style="bold")
    table.add_column("description")
    table.add_column("colours")
    for t in THEMES.values():
        swatch = " ".join(f"[on #{c}]   [/]" for c in (t.cover_background, t.background, t.accent, *t.palette[:3]))
        table.add_row(t.name, t.description, swatch)
    console.print(table)
    return 0


def _timed_agent_request(s: Settings, llm):
    """Send one full-size agent request (system prompt + all tool schemas), as every deck starts with.

    A random tag at the start of the prompt defeats Ollama's prompt cache, so the timing is
    that of a real first step. Returns (response, registry tool names, approx prompt tokens).
    """
    import tempfile
    import threading
    import uuid

    from deck_agent.agent import DeckAgent
    from deck_agent.diagnostics import fmt_duration, ollama_model_loaded
    from deck_agent.tools import DeckSession, build_registry

    with tempfile.TemporaryDirectory(prefix="deck-agent-doctor-") as tmp:
        session = DeckSession(output_dir=Path(tmp))
        registry = build_registry(s, session)
        system = DeckAgent(llm, registry, session, s).messages[0]["content"]
    tools = registry.schemas()
    messages = [
        {"role": "system", "content": f"(diagnostic run {uuid.uuid4().hex[:8]})\n{system}"},
        {"role": "user", "content": "Create a 3-slide deck about the history of tea for a book club."},
    ]
    approx_tokens = int((len(system) + len(json.dumps(tools))) / 3.2)
    done = threading.Event()
    with console.status(f"[cyan]Timing a full-size agent request (~{approx_tokens:,} tokens, the size every "
                        "deck starts with)…") as status:
        def tick() -> None:
            t0, loaded = time.time(), None
            while not done.wait(1):
                elapsed = time.time() - t0
                if s.backend == "ollama" and not loaded and int(elapsed) % 5 == 0:
                    loaded = ollama_model_loaded(llm.base_url, s.model)
                doing = ("Ollama is loading the model into memory" if loaded is False
                         else f"processing a full-size agent request (~{approx_tokens:,} tokens)")
                status.update(f"[cyan]{doing}… {fmt_duration(elapsed)} (on a CPU this can take a few minutes)")

        threading.Thread(target=tick, daemon=True).start()
        try:
            resp = llm.chat(messages, tools)
        finally:
            done.set()
    return resp, set(registry.names), approx_tokens


def cmd_doctor(args: argparse.Namespace) -> int:
    from deck_agent.diagnostics import (
        SLOW_ADVICE,
        SLOW_LOAD_ADVICE,
        SLOW_LOAD_SECONDS,
        estimate_speed,
        fmt_duration,
        ollama_placement,
    )

    s = settings_from_args(args)
    rows: list[tuple[str, str, str]] = []  # (check, "ok" | "warn" | "optional" | "fail", details)
    llm = _make_llm(s)
    try:
        models = llm.list_models()
        rows.append(("model server", "ok", f"{s.backend} at {llm.base_url}"))
        present = s.model in models or f"{s.model}:latest" in models or s.backend == "openai"
        hint = "" if present else f" - run: ollama pull {s.model}"
        rows.append(("model installed", "ok" if present else "fail",
                     (s.model if present else f"{s.model} not found") + hint))
        if present and s.backend == "ollama":
            caps = llm.capabilities()
            rows.append(("tool support", "ok" if "tools" in caps else "fail", ", ".join(caps) or "unknown"))
        if present and not args.skip_llm_test:
            resp, tool_names, approx_tokens = _timed_agent_request(s, llm)
            called = [c.name for c in resp.tool_calls]
            ok = any(name in tool_names for name in called)
            detail = f"called {', '.join(called)}" if called else "no tool call returned"
            if resp.recovered_from_text:
                detail += " (parsed from text: the server's tool-call parser may be missing)"
            rows.append(("tool-call test", "ok" if ok else "fail", detail))
            if resp.load_seconds >= 0.5:
                slow_load = resp.load_seconds >= SLOW_LOAD_SECONDS
                rows.append(("model load", "warn" if slow_load else "ok",
                             f"{fmt_duration(resp.load_seconds)} to load into memory"
                             + (f". {SLOW_LOAD_ADVICE}" if slow_load else "")))
            if s.backend == "ollama":
                placement = ollama_placement(llm.base_url, s.model)
                if placement:
                    rows.append(("model placement", "ok", placement.text))
            speed = estimate_speed(resp, resp.prompt_tokens or approx_tokens)
            if speed is None:
                rows.append(("speed", "warn", f"could not measure (request took {resp.seconds:.0f}s)"))
            else:
                slow = speed.verdict in ("slow", "very slow")
                rows.append(("speed", "warn" if slow else "ok",
                             speed.summary() + (f". {SLOW_ADVICE}" if slow else "")))
    except Exception as exc:  # noqa: BLE001
        rows.append(("model server", "fail", str(exc)[:300]))
    finally:
        llm.close()

    if s.web:
        try:
            from deck_agent.tools.web import make_search_backend

            backend = make_search_backend(s.search_backend, searxng_url=s.searxng_url, region=s.search_region,
                                          image_license=s.image_license, wikipedia_fallback=s.wikipedia_fallback)
            res = backend.text("python programming language", 2)
            rows.append(("web search", "ok" if res else "fail", f"{s.search_backend}: {len(res)} results"))
        except Exception as exc:  # noqa: BLE001
            rows.append(("web search", "fail", f"{s.search_backend}: {str(exc)[:160]}"))
    soffice = find_soffice()
    rows.append(("PDF export", "ok" if soffice else "optional",
                 soffice or "LibreOffice not found: PDF export and the UI's PowerPoint render are off; "
                            "PPTX and HTML work"))

    styles = {"ok": "[green]ok", "warn": "[yellow]warn", "optional": "[cyan]optional", "fail": "[red]fail"}
    table = Table(title="deck-agent doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("details")
    for name, status, detail in rows:
        table.add_row(name, styles[status], escape(detail))
    console.print(table)
    return 1 if any(status == "fail" for _, status, _ in rows) else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"run": cmd_run, "chat": cmd_chat, "ui": cmd_ui, "render": cmd_render, "sample": cmd_sample,
                "themes": cmd_themes, "doctor": cmd_doctor}
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
