"""Slow-model diagnostics: timings, explanations, the UI's activity log and `deck-agent doctor`."""

import json
import threading
import time

import httpx
import pytest
from starlette.testclient import TestClient

from deck_agent import cli, diagnostics
from deck_agent.config import Settings
from deck_agent.diagnostics import (
    estimate_speed,
    explain_wait,
    fmt_duration,
    ollama_model_loaded,
    ollama_placement,
)
from deck_agent.llm import LLMError, LLMResponse, OllamaLLM, OpenAICompatLLM, ToolCall
from deck_agent.web import server
from deck_agent.web.server import create_app

REAL_CLIENT = httpx.Client
TOOLS = [{"type": "function", "function": {"name": "web_search", "description": "",
                                           "parameters": {"type": "object", "properties": {}}}}]
MESSAGES = [{"role": "user", "content": "hi"}]


def _client(llm, handler):
    llm._http = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


# --------------------------------------------------------------------------- timings from the clients
def test_ollama_reports_timings():
    body = "\n".join(json.dumps(c) for c in [
        {"message": {"content": "", "tool_calls": [{"function": {"name": "web_search", "arguments": {}}}]}},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 3500, "eval_count": 170,
         "load_duration": 12e9, "prompt_eval_duration": 70e9, "eval_duration": 34e9},
    ]).encode()
    llm = _client(OllamaLLM("m", "http://h"), lambda r: httpx.Response(200, content=body))
    resp = llm.chat(MESSAGES, TOOLS)
    assert resp.load_seconds == 12 and resp.prompt_seconds == 70 and resp.generation_seconds == 34
    assert resp.prompt_rate == 50 and resp.generation_rate == 5
    assert resp.seconds >= 0 and resp.first_output_seconds is not None


def test_openai_reports_wall_clock_timings():
    events = [{"choices": [{"delta": {"content": "Hello"}}]},
              {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}}]
    body = ("\n\n".join(f"data: {json.dumps(e)}" for e in events) + "\n\ndata: [DONE]\n\n").encode()
    llm = _client(OpenAICompatLLM("m", "http://h/v1"),
                  lambda r: httpx.Response(200, content=body, headers={"content-type": "text/event-stream"}))
    resp = llm.chat(MESSAGES, TOOLS)
    assert resp.content == "Hello" and resp.first_output_seconds is not None and resp.seconds >= resp.first_output_seconds


@pytest.mark.parametrize(("error", "expected"), [
    ("timed out waiting for llama-server to start", "OLLAMA_LOAD_TIMEOUT"),
    ("model requires more system memory (12 GiB) than is available (8 GiB)", "Not enough free memory"),
])
def test_ollama_load_errors_are_explained(error, expected):
    llm = _client(OllamaLLM("qwen3.5:9b", "http://h"), lambda r: httpx.Response(500, json={"error": error}))
    with pytest.raises(LLMError, match=expected):
        llm.chat(MESSAGES, TOOLS)


# --------------------------------------------------------------------------- estimates and wording
def test_estimate_speed_matches_measured_runs():
    # Measured with qwen3.5:9b on 4 vCPU: ~28 tok/s prompt, ~3.5 tok/s writing; a 10-slide deck took ~15 min.
    resp = LLMResponse(prompt_tokens=3428, completion_tokens=170, prompt_seconds=3428 / 28,
                       generation_seconds=170 / 3.5, load_seconds=20)
    est = estimate_speed(resp, 3428)
    assert 2.5 * 60 < est.first_step_seconds < 3 * 60  # load time is reported separately
    assert 55 < est.later_step_seconds < 65
    assert 14 * 60 < est.deck_seconds < 18 * 60 and est.verdict == "slow"
    assert "first step ~2m" in est.summary()


def test_estimate_speed_without_server_timings():
    resp = LLMResponse(completion_tokens=100, seconds=30, first_output_seconds=10)
    est = estimate_speed(resp, 3000)  # 3000 tokens in 10 s, 100 tokens in 20 s
    assert est.prompt_rate == 300 and est.generation_rate == 5
    assert estimate_speed(LLMResponse(), 3000) is None


def test_fmt_duration_and_explanations():
    assert [fmt_duration(x) for x in (5, 65, 3725)] == ["5s", "1m 05s", "1h 02m"]
    assert "processing the ~3,500-token agent prompt" in explain_wait(1, 90, 0, "")
    assert "only shown once complete" in explain_wait(3, 70, 0, "")
    assert "loading the model" in explain_wait(1, 70, 0, "", loading=True)
    assert "thinking mode is on" in explain_wait(2, 40, 300, "thinking")


def _fake_ps(monkeypatch, models):
    handler = lambda r: httpx.Response(200, json={"models": models})  # noqa: E731
    monkeypatch.setattr(httpx, "Client", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))


def test_ollama_placement_and_loading(monkeypatch):
    _fake_ps(monkeypatch, [{"name": "qwen3.5:9b", "size": 7_400_000_000, "size_vram": 0, "context_length": 32768}])
    p = ollama_placement("http://h", "qwen3.5:9b")
    assert p.text.startswith("system RAM only") and "context 32,768" in p.text and not p.split
    assert ollama_model_loaded("http://h", "qwen3.5:9b") is True
    assert ollama_model_loaded("http://h", "qwen3.5:4b") is False  # not listed while still loading
    _fake_ps(monkeypatch, [{"name": "m:latest", "size": 100, "size_vram": 60}])
    assert ollama_placement("http://h", "m").split


def test_loading_check_when_ollama_is_down():
    assert ollama_model_loaded("http://127.0.0.1:9", "m", timeout=0.5) is None


# --------------------------------------------------------------------------- UI server activity log
class SlowLLM:
    """Answers each step after ``delay`` seconds, with Ollama-like timings."""

    def __init__(self, script, delay=0.0, gate=None):
        self.script, self.delay, self.gate = list(script), delay, gate
        self.base_url = "http://ollama.invalid"

    def chat(self, messages, tools, on_progress=None):
        if self.gate is not None:
            while not self.gate.wait(0.02):
                pass
        time.sleep(self.delay)
        step = self.script.pop(0) if self.script else "done"
        timing = dict(prompt_tokens=3400, completion_tokens=150, seconds=self.delay + 2.0, load_seconds=1.5,
                      prompt_seconds=1.0, generation_seconds=1.0)
        if isinstance(step, str):
            return LLMResponse(content=step, **timing)
        return LLMResponse(tool_calls=[ToolCall(name=n, arguments=a) for n, a in step], **timing)

    def list_models(self):
        return ["fake:9b"]

    def close(self):
        pass


def _app(tmp_path, llm, logs):
    settings = Settings(output_dir=tmp_path / "decks", web=False, model="fake:9b")
    app = create_app(settings, llm_factory=lambda s: llm, log=logs.append)
    return TestClient(app, base_url="http://127.0.0.1:7860"), app.state.ui


def _wait(state, timeout=10):
    deadline = time.time() + timeout
    while state.running and time.time() < deadline:
        time.sleep(0.02)
    assert not state.running


def test_ui_logs_each_step_to_the_terminal(tmp_path):
    logs = []
    llm = SlowLLM([[("set_deck_info", {"title": "Tea"})],
                   [("add_bullet_slide", {"title": "Origins", "bullets": ["China", "Trade"]})], "Done."])
    client, state = _app(tmp_path, llm, logs)
    assert client.post("/api/chat", json={"message": "Tea deck"}).status_code == 202
    _wait(state)
    text = "\n".join(logs)
    assert logs[0] == "New request: Tea deck"
    assert "step 1: model answered in 2s (loading the model 2s; reading the 3,400-token prompt 1s; " \
           "writing 150 tokens at 150.0 tok/s)" in text
    assert "  -> set_deck_info: Tea" in text and "ok: Created title slide" in text
    assert logs[-1].startswith("Finished in") and "2 slides" in logs[-1]
    starts = [e for e in client.get("/api/events.json").json()["events"] if e["type"] == "llm_start"]
    assert starts and all(isinstance(e["t"], float) for e in starts)  # lets the browser time each step


def test_heartbeat_reports_a_step_that_is_still_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "HEARTBEAT_SECONDS", 0.3)
    monkeypatch.setattr(server, "LOADING_CHECK_AFTER", 0.1)
    monkeypatch.setattr(server, "LOADING_CHECK_EVERY", 0.2)
    loaded = iter([False, False, False, True])
    monkeypatch.setattr(server, "ollama_model_loaded", lambda url, model: next(loaded, True))
    logs, gate = [], threading.Event()
    client, state = _app(tmp_path, SlowLLM(["Done."], gate=gate), logs)
    client.post("/api/chat", json={"message": "x"})

    def waiting_events():
        return [e for e in client.get("/api/events.json").json()["events"] if e["type"] == "waiting"]

    deadline = time.time() + 10
    while len(waiting_events()) < 2 and time.time() < deadline:
        time.sleep(0.05)
    gate.set()
    _wait(state)
    # loading -> loaded, reported to the browser; the terminal says why the step is slow
    assert [(e["step"], e["loading"]) for e in waiting_events()] == [(1, True), (1, False)]
    assert any("Ollama is still loading the model" in line for line in logs)


# --------------------------------------------------------------------------- deck-agent doctor
class DoctorLLM:
    base_url = "http://ollama.invalid"

    def __init__(self, called=("set_deck_info",), load=0.0):
        self.called, self.load = called, load

    def list_models(self):
        return ["qwen3.5:9b"]

    def capabilities(self):
        return ["completion", "tools"]

    def chat(self, messages, tools, on_progress=None):
        assert "(diagnostic run " in messages[0]["content"]  # defeats the prompt cache
        assert len(tools) > 10  # the real agent tool set
        return LLMResponse(tool_calls=[ToolCall(name=n, arguments={}) for n in self.called],
                           prompt_tokens=3500, completion_tokens=170, seconds=200,
                           load_seconds=self.load, prompt_seconds=3500 / 25, generation_seconds=170 / 3)

    def close(self):
        pass


def _doctor(monkeypatch, capsys, llm, placement=None, soffice=None):
    monkeypatch.setenv("DECK_AGENT_WEB", "false")
    monkeypatch.setenv("COLUMNS", "250")
    monkeypatch.setattr(cli, "_make_llm", lambda s: llm)
    monkeypatch.setattr(cli, "find_soffice", lambda: soffice)
    monkeypatch.setattr(diagnostics, "ollama_placement", lambda url, model: placement)
    monkeypatch.setattr(diagnostics, "ollama_model_loaded", lambda url, model: True)
    code = cli.main(["doctor", "-m", "qwen3.5:9b"])
    return code, capsys.readouterr().out


def test_doctor_reports_speed_load_and_optional_pdf(monkeypatch, capsys):
    placement = diagnostics.Placement("system RAM only, i.e. running on the CPU (7.4 GB)", False)
    code, out = _doctor(monkeypatch, capsys, DoctorLLM(load=95), placement)
    assert code == 0  # slow and no LibreOffice are warnings, not failures
    assert "called set_deck_info" in out
    assert "1m 35s to load into memory" in out and "OLLAMA_LOAD_TIMEOUT" in out
    assert "system RAM only" in out
    assert "prompt 25 tok/s, writing 3.0 tok/s" in out and "(slow)" in out and "qwen3.5:4b" in out
    assert "optional" in out and "LibreOffice not found" in out


def test_doctor_fails_when_the_model_does_not_call_tools(monkeypatch, capsys):
    code, out = _doctor(monkeypatch, capsys, DoctorLLM(called=()), soffice="/usr/bin/soffice")
    assert code == 1 and "no tool call returned" in out
