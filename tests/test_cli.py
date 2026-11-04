"""Command-line interface with a scripted fake model (no model server needed)."""

import json

import pytest

from deck_agent import cli
from deck_agent.llm import LLMResponse, ToolCall


class ScriptLLM:
    def __init__(self, script):
        self.script = list(script)
        self.closed = False

    def chat(self, messages, tools, on_progress=None):
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, str):
            return LLMResponse(content=step)
        return LLMResponse(tool_calls=[ToolCall(name=n, arguments=a) for n, a in step])

    def close(self):
        self.closed = True


BUILD = [
    [("set_deck_info", {"title": "Heat Pumps"})],
    [("add_bullet_slide", {"title": "Why", "bullets": ["Efficient", "Clean"]})],
]


def use_llm(monkeypatch, script):
    llm = ScriptLLM(script)
    monkeypatch.setattr(cli, "_make_llm", lambda s: llm)
    return llm


def test_run_builds_a_deck(tmp_path, monkeypatch):
    use_llm(monkeypatch, [*BUILD, [("save_deck", {})], "Done."])
    assert cli.main(["run", "heat pumps", "--no-web", "-o", str(tmp_path)]) == 0
    assert (tmp_path / "Heat_Pumps.pptx").exists()


def test_ctrl_c_during_run_keeps_the_work(tmp_path, monkeypatch, capsys):
    llm = use_llm(monkeypatch, [*BUILD, KeyboardInterrupt()])
    assert cli.main(["run", "heat pumps", "--no-web", "-o", str(tmp_path)]) == 130
    assert (tmp_path / "Heat_Pumps.pptx").exists() and llm.closed
    out = capsys.readouterr().out
    assert "auto-saved" in out and "Interrupted" in out


def test_theme_flag_wins_over_the_model(tmp_path, monkeypatch):
    use_llm(monkeypatch, [[("set_deck_info", {"title": "T", "theme": "forest"})],
                          [("add_bullet_slide", {"title": "a", "bullets": ["x", "y"]})], "Done."])
    assert cli.main(["run", "deck", "--no-web", "-t", "warm", "-o", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "T.deck.json").read_text())["theme"] == "warm"


@pytest.mark.parametrize(("content", "message"), [(None, "No such file"), ("{not json", "not a valid deck spec"),
                                                  ('{"slides": [{"kind": "nope"}]}', "not a valid deck spec")])
def test_render_reports_bad_specs(tmp_path, capsys, content, message):
    spec = tmp_path / "x.deck.json"
    if content is not None:
        spec.write_text(content)
    assert cli.main(["render", str(spec)]) == 1
    assert message in " ".join(capsys.readouterr().out.split())  # rich wraps long lines


def test_sample_and_render_round_trip(tmp_path):
    assert cli.main(["sample", "-o", str(tmp_path), "-f", "pptx"]) == 0
    spec = tmp_path / "sample_light.deck.json"
    assert cli.main(["render", str(spec), "-t", "dark", "-f", "html"]) == 0
    assert (tmp_path / "sample_light.html").exists()


def test_model_failure_exits_non_zero_but_keeps_the_partial_deck(tmp_path, monkeypatch):
    from deck_agent.llm import LLMError

    use_llm(monkeypatch, [*BUILD, LLMError("Lost the connection to the ollama server")])
    assert cli.main(["run", "heat pumps", "--no-web", "-o", str(tmp_path)]) == 1
    assert (tmp_path / "Heat_Pumps.pptx").exists()
