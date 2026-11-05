"""LLM clients against mocked HTTP servers (no model needed)."""

import json

import httpx
import pytest

from deck_agent.llm import LLMError, OllamaLLM, OpenAICompatLLM, make_llm

TOOLS = [{"type": "function", "function": {"name": "web_search", "description": "",
                                           "parameters": {"type": "object", "properties": {}}}}]
MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "web_search", "arguments": {"query": "q"}}]},
    {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": "results"},
]


def _client(llm, handler):
    llm._http = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


def ndjson(*chunks):
    return "\n".join(json.dumps(c) for c in chunks).encode()


def test_ollama_streaming_tool_call_and_payload():
    seen = {}

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        body = ndjson(
            {"message": {"role": "assistant", "content": "", "thinking": "hmm"}},
            {"message": {"role": "assistant", "content": "",
                         "tool_calls": [{"id": "x", "function": {"name": "web_search", "arguments": {"query": "a"}}}]}},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 5},
        )
        return httpx.Response(200, content=body)

    llm = _client(OllamaLLM("qwen3.5:9b", "http://h", num_ctx=4096, think=False), handler)
    progress = []
    resp = llm.chat(MESSAGES, TOOLS, on_progress=lambda n, p: progress.append(p))
    assert [(c.name, c.arguments) for c in resp.tool_calls] == [("web_search", {"query": "a"})]
    assert resp.reasoning == "hmm" and resp.prompt_tokens == 12 and resp.completion_tokens == 5
    assert progress == ["thinking"]
    assert seen["think"] is False and seen["options"]["num_ctx"] == 4096 and seen["stream"] is True
    assistant = seen["messages"][2]
    assert assistant["tool_calls"][0]["function"] == {"name": "web_search", "arguments": {"query": "q"}}
    assert seen["messages"][3] == {"role": "tool", "content": "results", "tool_name": "web_search"}


def test_ollama_retries_without_think_for_models_without_thinking():
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append("think" in payload)
        if "think" in payload:
            return httpx.Response(400, json={"error": "\"m\" does not support thinking"})
        return httpx.Response(200, content=ndjson({"message": {"content": "ok"}, "done": True}))

    llm = _client(OllamaLLM("m", "http://h", think=False), handler)
    assert llm.chat(MESSAGES, TOOLS).content == "ok"
    assert calls == [True, False] and llm.think is None


def test_ollama_helpful_errors():
    llm = _client(OllamaLLM("nope:1b", "http://h"),
                  lambda r: httpx.Response(404, json={"error": "model 'nope:1b' not found"}))
    with pytest.raises(LLMError, match="ollama pull nope:1b"):
        llm.chat(MESSAGES, TOOLS)
    llm = _client(OllamaLLM("old", "http://h"),
                  lambda r: httpx.Response(400, json={"error": "registry/old does not support tools"}))
    with pytest.raises(LLMError, match="does not support tool calling"):
        llm.chat(MESSAGES, TOOLS)


def test_ollama_connection_error_message():
    def handler(request):
        raise httpx.ConnectError("refused")

    llm = _client(OllamaLLM("m", "http://localhost:11434"), handler)
    with pytest.raises(LLMError, match="Is it running"):
        llm.chat(MESSAGES, TOOLS)
    # a remote GPU server gets network advice instead of "ollama serve"
    remote = _client(OpenAICompatLLM("m", "http://gpu.example.com:8000"), handler)
    with pytest.raises(LLMError, match="network or VPN allows the connection"):
        remote.chat(MESSAGES, TOOLS)


def test_ollama_recovers_tool_call_from_text():
    text = '<tool_call>{"name": "web_search", "arguments": {"query": "z"}}</tool_call>'
    llm = _client(OllamaLLM("m", "http://h"),
                  lambda r: httpx.Response(200, content=ndjson({"message": {"content": text}, "done": True})))
    resp = llm.chat(MESSAGES, TOOLS)
    assert resp.recovered_from_text and resp.tool_calls[0].arguments == {"query": "z"}


def sse(*chunks):
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    return ("\n\n".join(lines) + "\n\n").encode()


def test_openai_streaming_accumulates_tool_call_deltas():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        body = sse(
            {"choices": [{"delta": {"reasoning_content": "plan"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c9", "function": {"name": "web_search", "arguments": '{"que'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ry": "b"}'}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    llm = _client(OpenAICompatLLM("qwen", "http://h:8080", think=False), handler)
    resp = llm.chat(MESSAGES, TOOLS)
    assert resp.tool_calls[0].id == "c9" and resp.tool_calls[0].arguments == {"query": "b"}
    assert resp.reasoning == "plan" and resp.prompt_tokens == 7
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assistant = seen["messages"][2]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"query": "q"}
    assert seen["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "results"}


def test_openai_non_streaming_fallback():
    body = {"choices": [{"message": {"content": "<think>x</think>Done."}}], "usage": {"prompt_tokens": 1}}
    llm = _client(OpenAICompatLLM("m", "http://h/v1"),
                  lambda r: httpx.Response(200, json=body))
    resp = llm.chat(MESSAGES, TOOLS)
    assert resp.content == "Done." and resp.reasoning == "x" and resp.tool_calls == []


def test_make_llm():
    assert isinstance(make_llm("ollama", "m"), OllamaLLM)
    assert isinstance(make_llm("llamacpp", "m"), OpenAICompatLLM)
    with pytest.raises(ValueError):
        make_llm("cloud", "m")


def _drop(request):
    raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


def _connect_timeout(request):
    raise httpx.ConnectTimeout("timed out")


def _garbage(request):
    return httpx.Response(200, content=b'{"message": {"content": "a"}}\n<html>502 Bad Gateway</html>\n')


@pytest.mark.parametrize("cls", [OllamaLLM, OpenAICompatLLM])
@pytest.mark.parametrize(("handler", "message"), [
    (_drop, "Lost the connection"),  # server crashed / ran out of memory mid-answer
    (_connect_timeout, "Is it running"),
    (_garbage, "malformed response"),
])
def test_transport_failures_become_llm_errors(cls, handler, message):
    # Anything else escapes the agent loop, which then cannot stop cleanly and auto-save the deck.
    llm = _client(cls("m", "http://localhost:8080"), handler)
    with pytest.raises(LLMError, match=message):
        llm.chat(MESSAGES, TOOLS)
    with pytest.raises(LLMError):
        llm.list_models()


def test_openai_stream_without_tool_call_index():
    # Some servers omit "index"; argument fragments must still join into one call.
    body = sse(
        {"choices": [{"delta": {"tool_calls": [{"id": "a", "function": {"name": "web_search", "arguments": '{"qu'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"function": {"arguments": 'ery": "x"}'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"id": "b", "function": {"name": "web_search", "arguments": '{"query": "y"}'}}]}}]},
    )
    llm = _client(OpenAICompatLLM("m", "http://h"),
                  lambda r: httpx.Response(200, content=body, headers={"content-type": "text/event-stream"}))
    resp = llm.chat(MESSAGES, TOOLS)
    assert [(c.id, c.arguments) for c in resp.tool_calls] == [("a", {"query": "x"}), ("b", {"query": "y"})]
