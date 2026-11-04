"""Web UI server tests with a scripted fake model (no Ollama needed)."""

import io
import threading
import time

import pytest
from pptx import Presentation
from starlette.testclient import TestClient

from deck_agent.config import Settings
from deck_agent.deck.render import find_soffice
from deck_agent.factcheck import number_figure
from deck_agent.llm import LLMResponse, ToolCall
from deck_agent.web.server import EventLog, create_app

BUILD_SCRIPT = [
    [("set_deck_info", {"title": "Heat Pumps", "theme": "forest"})],
    [("add_bullet_slide", {"title": "Why heat pumps", "bullets": ["Efficient", "Clean"]})],
    [("add_chart_slide", {"title": "Sales", "categories": ["2024", "2025"],
                          "series": [{"name": "m units", "values": [2.6, 2.9]}]})],
    "Your deck is ready.",
]


class ScriptLLM:
    def __init__(self, script, gate: threading.Event | None = None):
        self.script = list(script)
        self.gate = gate
        self.prompts = []

    def chat(self, messages, tools, on_progress=None):
        self.prompts.append(messages[-1]["content"])
        if self.gate is not None:
            while not self.gate.wait(0.02):
                if on_progress:
                    on_progress(1, "writing")  # raises AgentCancelled after /api/stop
        step = self.script.pop(0) if self.script else "done"
        if isinstance(step, str):
            return LLMResponse(content=step)
        return LLMResponse(tool_calls=[ToolCall(name=n, arguments=a) for n, a in step])

    def list_models(self):
        return ["fake:4b", "fake:9b"]

    def close(self):
        pass


class Factory:
    def __init__(self, script=BUILD_SCRIPT, gate=None):
        self.script, self.gate, self.models, self.instances = script, gate, [], []

    def __call__(self, settings):
        self.models.append(settings.model)
        llm = ScriptLLM(self.script, self.gate)
        self.instances.append(llm)
        return llm


def make_client(tmp_path, factory=None, **settings_kw):
    settings = Settings(output_dir=tmp_path / "decks", web=False, model="fake:4b", **settings_kw)
    factory = factory or Factory()
    app = create_app(settings, llm_factory=factory)
    return TestClient(app, base_url="http://127.0.0.1:7860"), app.state.ui, factory


def wait_idle(state, timeout=10):
    deadline = time.time() + timeout
    while state.running and time.time() < deadline:
        time.sleep(0.02)
    assert not state.running


def post(client, path, body=None):
    return client.post(path, json=body or {})


def test_index_and_state(tmp_path):
    client, _, _ = make_client(tmp_path)
    page = client.get("/")
    assert page.status_code == 200 and "Deck Agent" in page.text
    assert client.get("/static/app.js").status_code == 200
    s = client.get("/api/state").json()
    assert s["model"] == "fake:4b" and s["models"] == ["fake:4b", "fake:9b"]
    assert {t["name"] for t in s["themes"]} >= {"light", "dark"}
    assert s["deck"]["count"] == 0 and s["running"] is False


def test_chat_builds_deck_and_records_events(tmp_path):
    client, state, factory = make_client(tmp_path, Factory(script=[
        [("set_deck_info", {"title": "Heat Pumps", "theme": "forest"})],
        [("add_bullet_slide", {"title": "Why heat pumps", "bullets": ["Efficient", "Clean"]})],
        [("add_chart_slide", {"title": "Sales", "categories": ["2024", "2025"],
                              "series": [{"name": "m units", "values": [2.6, 2.9]}]})],
        "Your deck is ready.",
    ]))
    r = post(client, "/api/chat", {"message": "Heat pump deck. Sales: 2.6 million (2024), 2.9 million (2025)."})
    assert r.status_code == 202
    wait_idle(state)
    events = client.get("/api/events.json?since=0").json()["events"]
    types = [e["type"] for e in events]
    assert types[0] == "user" and types[1] == "status" and types[-1] == "status"
    assert events[-1]["running"] is False
    assert types.count("tool_call") == 3 and types.count("tool_result") == 3
    assert "assistant" in types and events[types.index("assistant")]["text"] == "Your deck is ready."
    decks = [e for e in events if e["type"] == "deck"]
    assert decks[-1]["count"] == 3 and decks[-1]["theme"] == "forest"
    assert decks[-1]["slides"][2] == {"n": 3, "kind": "chart", "title": "Sales"}
    assert decks[-1]["focus"] == 3  # the newly added slide is focused in the preview
    # incremental fetch
    last = events[-1]["id"]
    assert client.get(f"/api/events.json?since={last}").json()["events"] == []


def test_preview_and_downloads(tmp_path):
    client, state, _ = make_client(tmp_path, Factory(script=BUILD_SCRIPT[:2] + ["ok"]))
    assert "Slides will appear here" in client.get("/api/preview.html").text
    assert client.get("/api/download/pptx").status_code == 404  # nothing to download yet
    post(client, "/api/chat", {"message": "go"})
    wait_idle(state)
    html = client.get("/api/preview.html").text
    assert html.count('<section class="slide') == 2 and "Why heat pumps" in html
    r = client.get("/api/download/pptx")
    assert r.status_code == 200 and 'filename="Heat_Pumps.pptx"' in r.headers["content-disposition"]
    assert len(Presentation(io.BytesIO(r.content)).slides) == 2
    assert (tmp_path / "decks" / "Heat_Pumps.pptx").exists()  # also kept in the output folder
    assert client.get("/api/download/json").json()["title"] == "Heat Pumps"
    assert client.get("/api/download/html").headers["content-type"].startswith("text/html")
    assert client.get("/api/download/exe").status_code == 404


@pytest.mark.skipif(find_soffice() is None, reason="LibreOffice not installed")
def test_pdf_preview_is_cached_per_version(tmp_path):
    client, state, _ = make_client(tmp_path, Factory(script=BUILD_SCRIPT[:2] + ["ok"]))
    assert client.get("/api/preview.pdf").status_code == 404
    post(client, "/api/chat", {"message": "go"})
    wait_idle(state)
    r = client.get("/api/preview.pdf?v=1")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    assert r.headers["content-disposition"].startswith("inline")
    first = state._pdf_cache
    client.get("/api/preview.pdf?v=1")
    assert state._pdf_cache == first  # not re-rendered
    post(client, "/api/theme", {"theme": "dark"})
    client.get("/api/preview.pdf")
    assert state._pdf_cache != first  # deck changed -> re-rendered
    dl = client.get("/api/download/pdf")
    assert dl.status_code == 200 and 'filename="Heat_Pumps.pdf"' in dl.headers["content-disposition"]

    # slide images of the real PPTX for the "PowerPoint render" tab
    assert client.get("/api/preview/page/1.png").status_code in (200, 404)
    pages = client.get("/api/preview/pages").json()
    assert pages == {"version": state.deck_version, "count": 2}
    img = client.get("/api/preview/page/2.png")
    assert img.status_code == 200 and img.content.startswith(b"\x89PNG")
    assert client.get("/api/preview/page/3.png").status_code == 404


def test_busy_agent_rejects_second_message_and_can_be_stopped(tmp_path):
    gate = threading.Event()
    client, state, _ = make_client(tmp_path, Factory(gate=gate))
    assert post(client, "/api/chat", {"message": "first"}).status_code == 202
    second = post(client, "/api/chat", {"message": "second"})
    assert second.status_code == 409 and "still working" in second.json()["error"]
    assert post(client, "/api/reset").status_code == 409
    assert post(client, "/api/settings", {"model": "fake:9b"}).status_code == 409
    assert post(client, "/api/stop").status_code == 200
    wait_idle(state)
    events = client.get("/api/events.json").json()["events"]
    assert any(e["type"] == "note" and e["text"] == "Stopped." for e in events)
    assert events[-1] == {**events[-1], "type": "status", "running": False}


def test_theme_reset_and_settings(tmp_path):
    client, state, factory = make_client(tmp_path, Factory(script=BUILD_SCRIPT[:2] + ["ok"]))
    post(client, "/api/chat", {"message": "go"})
    wait_idle(state)
    assert post(client, "/api/theme", {"theme": "dark"}).status_code == 200
    assert state.session.deck.theme == "dark"
    assert client.get("/api/events.json").json()["events"][-1]["theme"] == "dark"
    assert post(client, "/api/theme", {"theme": "neon"}).status_code == 400

    assert post(client, "/api/settings", {"model": "fake:9b", "web": True, "verify_numbers": False}).status_code == 200
    assert factory.models == ["fake:4b", "fake:9b"] and state.agent.llm is factory.instances[-1]
    assert "web_search" in state.agent.registry and "web_search" in state.agent.messages[0]["content"]
    assert state.session.verify_numbers is False

    assert post(client, "/api/reset").status_code == 200
    events = client.get("/api/events.json").json()["events"]
    assert [e["type"] for e in events] == ["reset"] and events[0]["count"] == 0
    assert state.session.deck.slides == [] and len(state.agent.messages) == 1


def test_attachments_are_added_to_the_prompt(tmp_path):
    client, state, factory = make_client(tmp_path, Factory(script=["ok"]))
    big = "x" * 50_000
    post(client, "/api/chat", {"message": "Use my notes", "attachments": [
        {"name": "notes.md", "text": "Sales rose 13% in 2025."}, {"name": "big.txt", "text": big}]})
    wait_idle(state)
    prompt = factory.instances[0].prompts[0]
    assert prompt.startswith("Use my notes") and "Source material from the file 'notes.md'" in prompt
    assert "[... truncated]" in prompt and len(prompt) < 45_000
    user = client.get("/api/events.json").json()["events"][0]
    assert user["attachments"] == ["notes.md", "big.txt"] and user["truncated"] is True
    assert state.session.facts.supports(number_figure(13))  # attached numbers count as sources
    empty = post(client, "/api/chat", {"message": "  "})
    assert empty.status_code == 400 and "Empty" in empty.json()["error"]


def test_extract_text_and_pdf(tmp_path):
    client, _, _ = make_client(tmp_path)
    r = client.post("/api/extract?name=notes.md", content=b"# Notes\nSales  rose 13%",
                    headers={"Content-Type": "application/octet-stream"})
    assert r.json() == {"name": "notes.md", "text": "# Notes\nSales rose 13%", "chars": 22}
    empty = client.post("/api/extract?name=a.txt", content=b"   ", headers={"Content-Type": "application/octet-stream"})
    assert empty.status_code == 422
    bad = client.post("/api/extract?name=a.pdf", content=b"%PDF-1.7 broken", headers={"Content-Type": "application/octet-stream"})
    assert bad.status_code == 422


def test_security_guards(tmp_path):
    client, _, _ = make_client(tmp_path)
    evil = client.post("/api/chat", json={"message": "x"}, headers={"Origin": "http://evil.example"})
    assert evil.status_code == 403
    same = client.post("/api/theme", json={"theme": "dark"}, headers={"Origin": "http://127.0.0.1:7860"})
    assert same.status_code == 200
    form = client.post("/api/chat", content="message=x", headers={"Content-Type": "text/plain"})
    assert form.status_code == 400  # only JSON bodies (forces a CORS preflight for cross-site requests)
    rebinding = client.get("/api/state", headers={"Host": "attacker.example"})
    assert rebinding.status_code == 400


def test_open_saved_deck(tmp_path):
    from deck_agent.sample import build_sample_deck

    deck = build_sample_deck("warm", tmp_path / "assets")
    settings = Settings(output_dir=tmp_path / "decks", web=False, model="fake:4b")
    app = create_app(settings, llm_factory=Factory(script=["ok"]), initial_deck=deck, deck_name="mine.deck.json")
    client = TestClient(app, base_url="http://127.0.0.1:7860")
    state = client.get("/api/state").json()
    assert state["deck"]["count"] == len(deck.slides) and state["deck"]["theme"] == "warm"
    events = client.get("/api/events.json").json()["events"]
    assert events[0]["type"] == "deck" and events[1]["type"] == "note" and "mine.deck.json" in events[1]["text"]
    assert client.get("/api/download/pptx").status_code == 200
    # numbers already in the user's deck count as sources for later edits
    assert app.state.ui.session.facts.supports(number_figure(79.9))


def test_event_log_collapses_progress_and_trims():
    log = EventLog(limit=3)
    log.add("tool_call")
    log.add("progress", tokens=1)
    log.add("progress", tokens=2)
    assert [e["type"] for e in log.since(0)] == ["tool_call", "progress"]
    assert log.since(0)[-1]["tokens"] == 2
    for _ in range(5):
        log.add("note")
    assert len(log.since(0)) == 3 and log.last_id == 8


def test_attachments_that_do_not_fit_are_reported(tmp_path):
    client, state, factory = make_client(tmp_path, Factory(script=["ok"]))
    post(client, "/api/chat", {"message": "m", "attachments": [
        {"name": "full.txt", "text": "a" * 40_000}, {"name": "later.md", "text": "important"}]})
    wait_idle(state)
    user = client.get("/api/events.json").json()["events"][0]
    assert user["attachments"] == ["full.txt"] and user["omitted"] == ["later.md"] and user["truncated"] is True
    assert "important" not in factory.instances[0].prompts[0]


def test_empty_reply_and_failed_autosave_are_shown(tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")  # the output "folder" is a file, so saving fails
    settings = Settings(output_dir=blocker, web=False, model="fake:4b", verify_numbers=False)
    app = create_app(settings, llm_factory=Factory(script=BUILD_SCRIPT[:2] + [""]))
    client, state = TestClient(app, base_url="http://127.0.0.1:7860"), app.state.ui
    post(client, "/api/chat", {"message": "go"})
    wait_idle(state)
    events = client.get("/api/events.json").json()["events"]
    assert any(e["type"] == "error" and "Could not save" in e["text"] for e in events)
    assert any(e["type"] == "note" and "without a written reply" in e["text"] for e in events)


def test_new_deck_forgets_old_sources(tmp_path):
    client, state, _ = make_client(tmp_path, Factory(script=["ok"]))
    post(client, "/api/chat", {"message": "Sales were 4,321 units, see https://old.example/r"})
    wait_idle(state)
    assert state.session.facts.supports(number_figure(4321))
    post(client, "/api/reset")
    assert not state.session.facts.supports(number_figure(4321)) and not state.session.known_urls


def test_theme_picked_in_the_ui_is_kept(tmp_path):
    client, state, _ = make_client(tmp_path, Factory(script=[
        [("set_deck_info", {"title": "T"})], "ok",
        [("set_deck_info", {"title": "T2", "theme": "forest"})], "ok",
        [("set_deck_info", {"title": "T3", "theme": "forest"})], "ok"]))
    post(client, "/api/chat", {"message": "go"})
    wait_idle(state)
    post(client, "/api/theme", {"theme": "dark"})
    post(client, "/api/chat", {"message": "rename it to T2"})
    wait_idle(state)
    assert state.session.deck.title == "T2" and state.session.deck.theme == "dark"
    post(client, "/api/chat", {"message": "use the forest theme"})
    wait_idle(state)
    assert state.session.deck.theme == "forest"


def test_bad_query_parameters(tmp_path):
    client, _, _ = make_client(tmp_path)
    assert client.get("/api/events.json?since=abc").status_code == 400


def _tiny_pdf(text: str) -> bytes:
    """A minimal one-page PDF with a line of text (enough for pypdf's extractor)."""
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_extract_reads_a_real_pdf(tmp_path):
    client, _, _ = make_client(tmp_path)
    r = client.post("/api/extract?name=report.pdf", content=_tiny_pdf("Sales rose 13 percent"),
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200 and "Sales rose 13 percent" in r.json()["text"]


def test_preview_folder_is_removed_on_shutdown(tmp_path):
    settings = Settings(output_dir=tmp_path / "decks", web=False, model="fake:4b")
    app = create_app(settings, llm_factory=Factory())
    preview = app.state.ui.preview_dir
    with TestClient(app, base_url="http://127.0.0.1:7860") as client:
        assert client.get("/api/state").status_code == 200 and preview.exists()
    assert not preview.exists()


@pytest.mark.skipif(find_soffice() is None, reason="LibreOffice not installed")
def test_old_previews_are_deleted(tmp_path):
    client, state, _ = make_client(tmp_path, Factory(script=BUILD_SCRIPT[:2] + ["ok"]))
    post(client, "/api/chat", {"message": "go"})
    wait_idle(state)
    assert client.get("/api/preview/pages").status_code == 200
    post(client, "/api/theme", {"theme": "dark"})
    pages = client.get("/api/preview/pages").json()
    assert pages["version"] == state.deck_version
    names = {p.name for p in state.preview_dir.iterdir()}
    assert names == {f"preview_{pages['version']}.{ext}" for ext in ("pdf", "pptx", "deck.json")} | {
        f"pages_{pages['version']}"}
