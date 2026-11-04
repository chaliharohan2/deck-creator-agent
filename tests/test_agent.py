"""Agent loop behaviour with a scripted fake model."""

from deck_agent.agent import NUDGE, DeckAgent
from deck_agent.config import Settings
from deck_agent.llm import LLMError, LLMResponse, ToolCall
from deck_agent.tools import DeckSession, build_registry


class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools, on_progress=None):
        self.calls.append([dict(m) for m in messages])
        self.tool_names = [t["function"]["name"] for t in tools]
        self.tool_history = [*getattr(self, "tool_history", []), self.tool_names]
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, str):
            return LLMResponse(content=step)
        return LLMResponse(tool_calls=[ToolCall(name=n, arguments=a) for n, a in step])


class NoSearch:
    name = "none"

    def text(self, q, n):
        return [{"title": "Heat pump sales", "url": "https://example.org/hp", "snippet": "3 million units"}]

    def images(self, q, n):
        return []


def make_agent(tmp_path, script, interactive=False, **settings_kw):
    settings = Settings(output_dir=tmp_path, formats=["pptx"], max_steps=settings_kw.pop("max_steps", 20), **settings_kw)
    session = DeckSession(output_dir=tmp_path, formats=["pptx"])
    registry = build_registry(settings, session, backend=NoSearch())
    events = []
    agent = DeckAgent(FakeLLM(script), registry, session, settings, interactive=interactive,
                      on_event=lambda k, d: events.append((k, d)))
    return agent, events


def test_happy_path_builds_and_saves(tmp_path):
    script = [
        [("web_search", {"query": "heat pumps europe 2026"})],
        [("set_deck_info", {"title": "Heat Pumps", "theme": "forest"})],
        [("add_bullet_slide", {"title": "Why", "bullets": ["a", "b"]}),
         ("add_chart_slide", {"title": "Sales", "categories": ["2024", "2025"],
                              "series": [{"name": "m units", "values": [2.6, 3.0]}],
                              "sources": ["https://example.org/hp"]})],
        [("save_deck", {"filename": "heat"})],
        "Your 3-slide deck is ready.",
    ]
    agent, events = make_agent(tmp_path, script)
    result = agent.run("Make a deck on heat pumps")
    assert result.stopped_reason == "done" and result.text == "Your 3-slide deck is ready."
    assert (tmp_path / "heat.pptx").exists() and result.files["pptx"].name == "heat.pptx"
    assert result.tool_calls == 5 and result.errors == 0
    assert not any(k == "autosave" for k, _ in events)
    # tool results are fed back with the matching call id
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 5 and tool_msgs[0]["name"] == "web_search"
    assert "Today's date" in agent.messages[0]["content"] and "web_search" in agent.messages[0]["content"]


def test_errors_are_fed_back_and_corrected(tmp_path):
    script = [
        [("set_deck_info", {"title": "T"})],
        [("add_chart_slide", {"title": "c", "categories": ["a", "b"], "series": [{"name": "s", "values": [1]}]})],
        [("add_chart_slide", {"title": "c", "categories": ["a", "b"], "series": [{"name": "s", "values": [1, 2]}]})],
        "done",
    ]
    agent, events = make_agent(tmp_path, script)
    result = agent.run("x")
    assert result.errors == 1
    assert any("one-to-one" in m["content"] for m in agent.messages if m["role"] == "tool")
    assert any(k == "autosave" for k, _ in events)  # the model never called save_deck
    assert (tmp_path / "T.pptx").exists()


def test_nudges_when_model_only_talks(tmp_path):
    script = [
        "Here is my plan: slide 1 ..., slide 2 ...",
        [("set_deck_info", {"title": "Plan"})],
        [("add_bullet_slide", {"title": "a", "bullets": ["b"]})],
        "Finished.",
    ]
    agent, events = make_agent(tmp_path, script)
    result = agent.run("x")
    assert result.text == "Finished."
    assert any(m["role"] == "user" and m["content"] == NUDGE for m in agent.messages)
    assert [k for k, _ in events].count("nudge") == 1


def test_no_nudge_in_chat_mode(tmp_path):
    agent, _ = make_agent(tmp_path, ["What audience is this for?"], interactive=True)
    result = agent.run("make a deck")
    assert result.text == "What audience is this for?"
    assert "ask one short question" in agent.messages[0]["content"]


def test_repeated_identical_calls_are_short_circuited(tmp_path):
    call = [("web_search", {"query": "same"})]
    agent, _ = make_agent(tmp_path, [call, call, call, call, "never reached"])
    result = agent.run("x")
    assert result.stopped_reason == "loop"
    outputs = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert outputs[0].startswith("1. Heat pump sales")
    assert all("You just made this exact web_search call" in o for o in outputs[1:])


def test_exhausted_search_is_removed_from_tool_list(tmp_path):
    script = [[("web_search", {"query": f"q{i}"})] for i in range(5)] + [
        [("set_deck_info", {"title": "T"})], [("add_bullet_slide", {"title": "a", "bullets": ["b"]})], "done"]
    agent, _ = make_agent(tmp_path, script, max_searches=4)
    agent.run("x")
    history = agent.llm.tool_history
    assert "web_search" in history[0]
    assert "web_search" not in history[5]  # hidden after the budget ran out
    agent.registry.new_request()
    assert "web_search" in [s["function"]["name"] for s in agent.registry.schemas()]


def test_llm_error_stops_cleanly_and_keeps_work(tmp_path):
    script = [[("set_deck_info", {"title": "Keep"})], [("add_bullet_slide", {"title": "a", "bullets": ["b"]})],
              LLMError("server went away")]
    agent, _ = make_agent(tmp_path, script)
    result = agent.run("x")
    assert result.stopped_reason == "llm_error" and "server went away" in result.text
    assert (tmp_path / "Keep.pptx").exists()


def test_max_steps(tmp_path):
    script = [[("list_slides", {})], [("set_deck_info", {"title": "A"})], [("list_slides", {})]]
    agent, _ = make_agent(tmp_path, script, max_steps=3)
    result = agent.run("x")
    assert result.stopped_reason == "max_steps"


def test_context_trimming(tmp_path):
    agent, events = make_agent(tmp_path, [], num_ctx=4000)
    agent.messages += [{"role": "tool", "name": "read_webpage", "tool_call_id": str(i), "content": "x" * 5000}
                       for i in range(10)]
    agent._trim_context()
    assert any(k == "trimmed" for k, _ in events)
    trimmed = [m for m in agent.messages if m["role"] == "tool" and "trimmed" in m["content"]]
    assert trimmed and all(len(m["content"]) < 700 for m in trimmed)
    assert len(agent.messages[-1]["content"]) == 5000  # recent output kept intact


def test_offline_prompt(tmp_path):
    agent, _ = make_agent(tmp_path, [], web=False)
    assert "Web research is disabled" in agent.messages[0]["content"]
    assert "web_search" not in agent.registry
