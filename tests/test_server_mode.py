"""CPU / GPU-server modes, .env credentials, and talking to a vLLM-style server (mocked over real HTTP)."""

import json
import time

import httpx
import pytest
from conftest import clear_deck_env
from mock_vllm import BackgroundServer, create_mock_vllm
from pptx import Presentation
from starlette.testclient import TestClient

from deck_agent import cli
from deck_agent.config import ConnectionProfile, Settings, normalize_mode
from deck_agent.envfile import find_env_file, load_env_file, parse_env_file
from deck_agent.llm import LLMError, OllamaLLM, OpenAICompatLLM
from deck_agent.web.server import create_app

KEY = "s3cret-test-key"
MODEL = "mock/Qwen3.8-27B"


def deck_script(messages, tools):
    """Build a 2-slide deck, then reply."""
    done = sum(1 for m in messages if m["role"] == "tool")
    plan = [
        [("set_deck_info", {"title": "Green Tea", "theme": "forest"})],
        [("add_bullet_slide", {"title": "Why green tea", "bullets": ["Rich in antioxidants", "Less caffeine"]})],
        [("save_deck", {})],
    ]
    return plan[done] if done < len(plan) else "Done: your 2-slide deck is saved."


@pytest.fixture
def mock_server():
    app = create_mock_vllm(api_key=KEY, model=MODEL, script=deck_script)
    with BackgroundServer(app) as srv:
        srv.requests = app.state.requests
        yield srv


def write_env(path, **values):
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    return path


# --------------------------------------------------------------------------- .env files
def test_parse_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# GPU server\n"
        "export DECK_AGENT_SERVER_URL=http://10.0.0.5:8000   # inline comment\n"
        'DECK_AGENT_SERVER_API_KEY="abc # not a comment"\n'
        "DECK_AGENT_SERVER_MODEL='Qwen/Qwen3.8-27B'\n"
        "not a setting\n\n"
        "EMPTY=\n"
    )
    assert parse_env_file(env) == {
        "DECK_AGENT_SERVER_URL": "http://10.0.0.5:8000",
        "DECK_AGENT_SERVER_API_KEY": "abc # not a comment",
        "DECK_AGENT_SERVER_MODEL": "Qwen/Qwen3.8-27B",
        "EMPTY": "",
    }


def test_env_file_does_not_override_real_environment(tmp_path, monkeypatch):
    env = write_env(tmp_path / ".env", DECK_AGENT_MODEL="from-file", DECK_AGENT_THEME="dark")
    monkeypatch.setenv("DECK_AGENT_MODEL", "from-shell")
    loaded = load_env_file(env)
    assert loaded == ["DECK_AGENT_THEME"]  # names only, never values
    s = Settings.from_env()
    assert s.model == "from-shell" and s.theme == "dark"


def test_find_env_file(tmp_path, monkeypatch):
    assert find_env_file() is None  # cwd is an empty temp dir in tests
    (tmp_path / ".env").write_text("X=1\n")
    monkeypatch.chdir(tmp_path)
    assert find_env_file() == tmp_path / ".env"
    monkeypatch.setenv("DECK_AGENT_ENV_FILE", str(tmp_path / "other.env"))
    assert find_env_file() == tmp_path / "other.env"
    assert find_env_file("explicit.env").name == "explicit.env"
    with pytest.raises(FileNotFoundError):
        load_env_file(tmp_path / "missing.env")


# --------------------------------------------------------------------------- modes
def test_modes_from_env(monkeypatch):
    monkeypatch.setenv("DECK_AGENT_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("DECK_AGENT_SERVER_URL", "http://gpu.example.com:8000")
    monkeypatch.setenv("DECK_AGENT_SERVER_API_KEY", KEY)
    monkeypatch.setenv("DECK_AGENT_SERVER_MODEL", MODEL)
    s = Settings.from_env()
    assert (s.mode, s.backend, s.model, s.api_key) == ("cpu", "ollama", "qwen3.5:4b", None)
    s.use_mode("gpu")  # alias
    assert (s.mode, s.backend, s.model, s.api_key, s.is_remote) == ("server", "openai", MODEL, KEY, True)
    s.model = "other-served-model"
    s.use_mode("cpu")
    assert s.model == "qwen3.5:4b" and not s.is_remote
    s.use_mode("server")
    assert s.model == "other-served-model"  # each mode remembers its own model
    assert KEY not in s.profile("server").describe()


def test_vllm_variable_fallbacks_and_server_default(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://203.0.113.7:8000")
    monkeypatch.setenv("VLLM_API_KEY", KEY)
    monkeypatch.setenv("MODEL_NAME", MODEL)
    monkeypatch.setenv("DECK_AGENT_MODE", "server")
    s = Settings.from_env()
    assert (s.mode, s.base_url, s.api_key, s.model) == ("server", "http://203.0.113.7:8000", KEY, MODEL)


def test_server_mode_needs_configuration():
    s = Settings.from_env()
    assert not s.server_configured
    with pytest.raises(ValueError, match="DECK_AGENT_SERVER_URL"):
        s.use_mode("server")
    with pytest.raises(ValueError):
        normalize_mode("cloud")


def test_profile_remote_detection():
    assert not ConnectionProfile("ollama", None, "m").is_remote
    assert not ConnectionProfile("openai", "http://127.0.0.1:8000", "m").is_remote
    assert ConnectionProfile("openai", "http://203.0.113.10:8000", "m").is_remote


# --------------------------------------------------------------------------- client: auth, network, errors
def _capture(llm, status=200, body=None):
    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        seen["url"] = str(request.url)
        return httpx.Response(status, json=body if body is not None else {"data": [{"id": MODEL}]})

    llm._http = httpx.Client(headers=llm._http.headers, transport=httpx.MockTransport(handler))
    return seen


def test_api_key_headers():
    seen = _capture(OpenAICompatLLM(MODEL, "http://gpu:8000", api_key=KEY))
    assert OpenAICompatLLM(MODEL, "http://gpu:8000", api_key=KEY)._http.headers["authorization"] == f"Bearer {KEY}"
    custom = OpenAICompatLLM(MODEL, "http://gpu:8000", api_key=KEY, auth_header="X-API-Key")
    assert custom._http.headers["x-api-key"] == KEY and "authorization" not in custom._http.headers
    llm = OpenAICompatLLM(MODEL, "http://gpu:8000", api_key=KEY)
    seen = _capture(llm)
    assert llm.list_models() == [MODEL]
    assert seen["url"] == "http://gpu:8000/v1/models"  # /v1 added when the URL has no path
    assert seen["headers"]["authorization"] == f"Bearer {KEY}"


def test_remote_servers_honour_proxy_settings():
    assert OpenAICompatLLM(MODEL, "http://203.0.113.10:8000").remote is True
    assert OllamaLLM("m", "http://localhost:11434").remote is False  # local servers bypass proxies


@pytest.mark.parametrize(("status", "body", "expected"), [
    (401, {"object": "error", "message": "Unauthorized", "code": 401}, "rejected the API key"),
    (404, {"object": "error", "message": "The model `x` does not exist.", "type": "NotFoundError"}, "does not serve a model"),
    (404, {"detail": "Not Found"}, "vLLM serves its API under /v1"),
    (400, {"object": "error", "message": '"auto" tool choice requires --enable-auto-tool-choice and '
                                         "--tool-call-parser to be set"}, "--tool-call-parser"),
    (400, {"object": "error", "message": "This model's maximum context length is 32768 tokens."}, "SERVER_NUM_CTX"),
])
def test_vllm_errors_are_explained(status, body, expected):
    llm = OpenAICompatLLM("x", "http://gpu:8000", api_key=KEY)
    llm._http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(status, json=body)))
    with pytest.raises(LLMError, match=expected):
        llm.chat([{"role": "user", "content": "hi"}], [])


# --------------------------------------------------------------------------- end to end over HTTP (mock vLLM)
def test_run_in_server_mode_from_env_file(tmp_path, mock_server):
    env = write_env(tmp_path / "gpu.env", DECK_AGENT_MODE="server", DECK_AGENT_SERVER_URL=mock_server.url,
                    DECK_AGENT_SERVER_API_KEY=KEY, DECK_AGENT_SERVER_MODEL=MODEL, DECK_AGENT_WEB="false")
    out = tmp_path / "decks"
    code = cli.main(["run", "A short deck about green tea", "--env-file", str(env), "-o", str(out)])
    assert code == 0
    assert len(Presentation(out / "Green_Tea.pptx").slides) == 2
    first = mock_server.requests[0]
    assert first["headers"]["authorization"] == f"Bearer {KEY}"
    assert first["body"]["model"] == MODEL and first["body"]["stream"] is True
    assert first["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert {t["function"]["name"] for t in first["body"]["tools"]} >= {"set_deck_info", "add_chart_slide"}


def test_server_model_defaults_to_the_served_one(tmp_path, mock_server, capsys):
    env = write_env(tmp_path / ".env", DECK_AGENT_SERVER_URL=mock_server.url, DECK_AGENT_SERVER_API_KEY=KEY,
                    DECK_AGENT_WEB="false")
    code = cli.main(["run", "deck", "--mode", "server", "--env-file", str(env), "-o", str(tmp_path / "d")])
    assert code == 0 and mock_server.requests[0]["body"]["model"] == MODEL
    assert "GPU server" in capsys.readouterr().out


def test_doctor_server_mode(tmp_path, mock_server, capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "250")
    env = write_env(tmp_path / ".env", DECK_AGENT_MODE="server", DECK_AGENT_SERVER_URL=mock_server.url,
                    DECK_AGENT_SERVER_API_KEY=KEY, DECK_AGENT_SERVER_MODEL=MODEL, DECK_AGENT_WEB="false")
    code = cli.main(["doctor", "--env-file", str(env)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "GPU server" in out and "(API key set)" in out and "called set_deck_info" in out and "speed" in out
    assert KEY not in out

    clear_deck_env(monkeypatch)  # a second command in the same process: start from a clean environment
    bad = write_env(tmp_path / "bad.env", DECK_AGENT_MODE="server", DECK_AGENT_SERVER_URL=mock_server.url,
                    DECK_AGENT_SERVER_API_KEY="wrong", DECK_AGENT_WEB="false")
    assert cli.main(["doctor", "--env-file", str(bad)]) == 1
    assert "rejected the API key" in capsys.readouterr().out


def test_doctor_explains_a_server_without_tool_calling(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "250")
    app = create_mock_vllm(api_key=KEY, model=MODEL, tool_calling=False)
    with BackgroundServer(app) as srv:
        env = write_env(tmp_path / ".env", DECK_AGENT_MODE="server", DECK_AGENT_SERVER_URL=srv.url,
                        DECK_AGENT_SERVER_API_KEY=KEY, DECK_AGENT_WEB="false")
        assert cli.main(["doctor", "--env-file", str(env)]) == 1
    assert "--enable-auto-tool-choice" in capsys.readouterr().out


def test_ui_switches_between_cpu_and_server(tmp_path, mock_server, monkeypatch):
    monkeypatch.setenv("DECK_AGENT_SERVER_URL", mock_server.url)
    monkeypatch.setenv("DECK_AGENT_SERVER_API_KEY", KEY)
    monkeypatch.setenv("DECK_AGENT_SERVER_MODEL", MODEL)
    settings = Settings.from_env()
    settings.web, settings.output_dir = False, tmp_path / "decks"
    app = create_app(settings)  # real model clients
    client = TestClient(app, base_url="http://127.0.0.1:7860")
    state = client.get("/api/state").json()
    assert state["mode"] == "cpu" and [m["available"] for m in state["modes"]] == [True, True]
    assert KEY not in json.dumps(state)  # the key never reaches the browser

    assert client.post("/api/settings", json={"mode": "server"}).status_code == 200
    state = client.get("/api/state").json()
    assert state["mode"] == "server" and state["model"] == MODEL and state["models"] == [MODEL] and state["remote"] is False
    assert client.post("/api/chat", json={"message": "Green tea deck"}).status_code == 202
    ui = app.state.ui
    deadline = time.time() + 20
    while ui.running and time.time() < deadline:
        time.sleep(0.05)
    assert len(ui.session.deck.slides) == 2 and mock_server.requests

    assert client.post("/api/settings", json={"mode": "cpu"}).status_code == 200
    assert ui.agent.llm.backend == "ollama" and ui.settings.model == "qwen3.5:9b"
    events = client.get("/api/events.json").json()["events"]
    assert [e["mode"] for e in events if e["type"] == "settings"] == ["server", "cpu"]


def test_ui_refuses_unconfigured_server_mode(tmp_path):
    settings = Settings.from_env()
    settings.web = False
    client = TestClient(create_app(settings), base_url="http://127.0.0.1:7860")
    modes = client.get("/api/state").json()["modes"]
    assert modes[1]["available"] is False and "DECK_AGENT_SERVER_URL" in modes[1]["detail"]
    r = client.post("/api/settings", json={"mode": "server"})
    assert r.status_code == 400 and ".env" in r.json()["error"]


def test_tools_are_only_sent_when_there_are_some():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "1 2 3"}}]})

    llm = OpenAICompatLLM(MODEL, "http://gpu:8000", api_key=KEY)
    llm._http = httpx.Client(transport=httpx.MockTransport(handler))
    llm.chat([{"role": "user", "content": "count"}], [])
    assert "tools" not in seen and "tool_choice" not in seen  # vLLM rejects tool_choice without tools
