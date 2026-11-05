"""A stand-in for a vLLM server, for testing deck-agent's server mode without a GPU.

It speaks the OpenAI-compatible API the way vLLM does: Bearer API key, ``GET /v1/models``,
streamed ``POST /v1/chat/completions`` with tool calls, and vLLM's error format for a bad
key, an unknown model, or a server started without tool calling.

Two ways to answer:

* **scripted** (tests): a function decides the next tool call or reply; deterministic.
* **proxy** (manual end-to-end runs): requests are forwarded to a real model behind any
  OpenAI-compatible endpoint, e.g. Ollama's ``http://localhost:11434/v1``.

Run it by hand::

    python tests/mock_vllm.py --port 8001 --api-key test-key --model mock/Qwen3.8-27B \\
        --upstream http://localhost:11434/v1 --upstream-model qwen3.5:4b

then point deck-agent at it with ``DECK_AGENT_SERVER_URL=http://localhost:8001`` and
``DECK_AGENT_SERVER_API_KEY=test-key``.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# A script maps the conversation so far to the next answer: a reply (str) or tool calls [(name, args), ...].
Script = Callable[[list[dict[str, Any]], list[dict[str, Any]]], "str | list[tuple[str, dict[str, Any]]]"]


def _vllm_error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"object": "error", "message": message, "type": kind, "param": None, "code": status},
                        status_code=status)


def _sse(obj: dict[str, Any] | str) -> str:
    return f"data: {obj if isinstance(obj, str) else json.dumps(obj)}\n\n"


def create_mock_vllm(*, api_key: str = "test-key", model: str = "mock/Qwen3.8-27B", script: Script | None = None,
                     upstream: str | None = None, upstream_model: str | None = None,
                     tool_calling: bool = True, max_model_len: int = 32768) -> Starlette:
    requests: list[dict[str, Any]] = []  # what the server received, for assertions

    def authorized(request: Request) -> bool:
        return request.headers.get("authorization") == f"Bearer {api_key}"

    async def models(request: Request) -> Response:
        if not authorized(request):
            return _vllm_error(401, "Unauthorized", "Unauthorized")
        return JSONResponse({"object": "list", "data": [
            {"id": model, "object": "model", "owned_by": "vllm", "max_model_len": max_model_len}]})

    async def chat(request: Request) -> Response:
        if not authorized(request):
            return _vllm_error(401, "Unauthorized", "Unauthorized")
        body = await request.json()
        requests.append({"body": body, "headers": dict(request.headers)})
        if body.get("model") != model:
            return _vllm_error(404, f"The model `{body.get('model')}` does not exist.", "NotFoundError")
        if body.get("tools") and body.get("tool_choice", "auto") == "auto" and not tool_calling:
            return _vllm_error(400, '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser '
                                    "to be set", "BadRequestError")
        if upstream:
            return await _proxy(body)
        answer = script(body["messages"], body.get("tools") or []) if script else "Hello from the mock server."
        return StreamingResponse(_stream(answer), media_type="text/event-stream")

    async def _proxy(body: dict[str, Any]) -> Response:
        forward = {**body, "model": upstream_model or body["model"]}
        if (body.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
            # vLLM honours chat_template_kwargs; Ollama's OpenAI endpoint needs reasoning_effort instead.
            forward["reasoning_effort"] = "none"
        client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0), trust_env=False)
        req = client.build_request("POST", f"{upstream.rstrip('/')}/chat/completions", json=forward)
        resp = await client.send(req, stream=True)

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(relay(), status_code=resp.status_code,
                                 media_type=resp.headers.get("content-type", "text/event-stream"))

    def _stream(answer):
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        yield _sse(chunk({"role": "assistant", "content": ""}))
        if isinstance(answer, str):
            for i in range(0, len(answer), 12):
                yield _sse(chunk({"content": answer[i:i + 12]}))
            finish = "stop"
        else:
            for index, (name, args) in enumerate(answer):
                text = json.dumps(args)
                half = len(text) // 2  # arguments arrive in pieces, as with vLLM's tool parsers
                yield _sse(chunk({"tool_calls": [{"index": index, "id": f"chatcmpl-tool-{uuid.uuid4().hex[:8]}",
                                                  "type": "function", "function": {"name": name, "arguments": ""}}]}))
                yield _sse(chunk({"tool_calls": [{"index": index, "function": {"arguments": text[:half]}}]}))
                yield _sse(chunk({"tool_calls": [{"index": index, "function": {"arguments": text[half:]}}]}))
            finish = "tool_calls"
        yield _sse(chunk({}, finish))
        yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [],
                    "usage": {"prompt_tokens": 3400, "completion_tokens": 120, "total_tokens": 3520}})
        yield _sse("[DONE]")

    app = Starlette(routes=[Route("/v1/models", models), Route("/v1/chat/completions", chat, methods=["POST"])])
    app.state.requests = requests
    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BackgroundServer:
    """Run an ASGI app with uvicorn in a thread (for tests)."""

    def __init__(self, app, port: int | None = None):
        import uvicorn

        self.port = port or free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> BackgroundServer:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--api-key", default="test-key")
    p.add_argument("--model", default="mock/Qwen3.8-27B", help="model name the mock reports and requires")
    p.add_argument("--upstream", help="forward requests to this OpenAI-compatible URL, e.g. http://localhost:11434/v1")
    p.add_argument("--upstream-model", help="model name to use upstream, e.g. qwen3.5:4b")
    p.add_argument("--no-tool-calling", action="store_true", help="behave like vLLM started without tool calling")
    a = p.parse_args()
    app = create_mock_vllm(api_key=a.api_key, model=a.model, upstream=a.upstream, upstream_model=a.upstream_model,
                           tool_calling=not a.no_tool_calling)
    print(f"mock vLLM on http://127.0.0.1:{a.port}  model={a.model}  "
          + (f"-> {a.upstream} ({a.upstream_model})" if a.upstream else "(fixed reply)"))
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
