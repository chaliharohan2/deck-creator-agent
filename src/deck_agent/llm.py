"""Clients for locally hosted LLM servers.

Two protocols are supported, both open source on the server side:

* ``ollama``  - Ollama's native ``/api/chat`` (MIT). Preferred: it lets us set the
  context window (``num_ctx``) and toggle thinking per request.
* ``openai``  - any OpenAI-compatible ``/v1/chat/completions`` server: llama.cpp's
  ``llama-server --jinja`` (MIT), vLLM (Apache-2.0), SGLang (Apache-2.0), LocalAI
  (MIT), or Ollama's compatibility endpoint.

Only ``httpx`` is used; no vendor SDK is required.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from deck_agent.toolcall_parse import extract_tool_calls, parse_arguments, strip_thinking

ProgressFn = Callable[[int, str], None]  # (approx tokens so far, phase: "thinking" | "writing")


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:10]}")
    error: str | None = None  # set when the arguments could not be parsed


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    recovered_from_text: bool = False  # tool calls were parsed out of plain text
    # Timing, for progress reports and `deck-agent doctor`.
    seconds: float = 0.0  # wall clock for the whole request
    first_output_seconds: float | None = None  # wall clock until the server streamed anything back
    load_seconds: float = 0.0  # server-reported (Ollama): loading the model into memory
    prompt_seconds: float = 0.0  # server-reported (Ollama): processing the prompt
    generation_seconds: float = 0.0  # server-reported (Ollama): writing the answer

    @property
    def prompt_rate(self) -> float | None:
        """Prompt tokens processed per second (None when the server does not report it)."""
        return self.prompt_tokens / self.prompt_seconds if self.prompt_seconds > 0 and self.prompt_tokens else None

    @property
    def generation_rate(self) -> float | None:
        if self.generation_seconds > 0 and self.completion_tokens:
            return self.completion_tokens / self.generation_seconds
        if self.first_output_seconds is not None and self.seconds > self.first_output_seconds and self.completion_tokens:
            return self.completion_tokens / (self.seconds - self.first_output_seconds)
        return None


def _finalize(content: str, reasoning: str, calls: list[ToolCall], known_tools: set[str]) -> LLMResponse:
    visible, think = strip_thinking(content)
    reasoning = "\n".join(x for x in (reasoning.strip(), think) if x)
    resp = LLMResponse(content=visible, tool_calls=calls, reasoning=reasoning)
    if not calls and visible:
        parsed, rest = extract_tool_calls(visible, known_tools)
        if parsed:
            resp.tool_calls = [ToolCall(name=p.name, arguments=p.arguments, error=p.error) for p in parsed]
            resp.content = rest
            resp.recovered_from_text = True
    return resp


class BaseLLM:
    backend = "base"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        num_ctx: int = 32768,
        think: bool = False,
        timeout: float = 900.0,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_ctx = num_ctx
        self.think: bool | None = think
        self.timeout = timeout
        self.api_key = api_key
        self.extra_body = extra_body or {}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # trust_env=False: the LLM server is local; corporate proxy variables must not intercept it.
        self._http = httpx.Client(headers=headers, trust_env=False,
                                  timeout=httpx.Timeout(connect=10.0, read=timeout, write=60.0, pool=10.0))

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             on_progress: ProgressFn | None = None) -> LLMResponse:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        self._http.close()

    def _connect_error(self, exc: Exception) -> LLMError:
        return LLMError(
            f"Cannot reach the {self.backend} server at {self.base_url} ({exc.__class__.__name__}). "
            "Is it running? For Ollama: `ollama serve` and `ollama pull " + self.model + "`."
        )

    def _transport_error(self, exc: Exception) -> LLMError:
        """Turn any failure while talking to the server into an LLMError.

        The agent stops cleanly (and auto-saves) on LLMError; any other exception would
        escape the loop and lose the deck built so far.
        """
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return self._connect_error(exc)
        if isinstance(exc, httpx.TimeoutException):
            return LLMError(f"The model did not answer within {self.timeout:.0f}s; try a smaller model or "
                            "raise --timeout.")
        if isinstance(exc, json.JSONDecodeError):
            return LLMError(f"The {self.backend} server sent a malformed response ({exc.msg}). "
                            "Check the server log; a proxy in between can also cause this.")
        return LLMError(f"Lost the connection to the {self.backend} server at {self.base_url} "
                        f"({exc.__class__.__name__}: {exc}). It may have crashed or run out of memory.")


class OllamaLLM(BaseLLM):
    """Ollama native API (``POST /api/chat``) with streaming."""

    backend = "ollama"

    def _messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in m["tool_calls"]
                    ]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, messages, tools, on_progress=None) -> LLMResponse:
        options = {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k, "num_ctx": self.num_ctx}
        options.update(self.extra_body.get("options", {}))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(messages),
            "tools": tools,
            "stream": True,
            "options": options,
            "keep_alive": self.extra_body.get("keep_alive", "30m"),
        }
        if self.think is not None:
            payload["think"] = self.think
        try:
            return self._stream(payload, {t["function"]["name"] for t in tools}, on_progress)
        except LLMError as exc:
            if "does not support thinking" in str(exc) and "think" in payload:
                self.think = None  # model has no thinking switch; stop sending it
                payload.pop("think")
                return self._stream(payload, {t["function"]["name"] for t in tools}, on_progress)
            raise

    def _stream(self, payload, known_tools, on_progress) -> LLMResponse:
        content, reasoning, raw_calls = [], [], []
        n = 0
        prompt_tokens = completion_tokens = 0
        t0, first, done = time.monotonic(), None, {}
        try:
            with self._http.stream("POST", f"{self.base_url}/api/chat", json=payload) as r:
                if r.status_code != 200:
                    body = r.read().decode("utf-8", "replace")
                    raise LLMError(self._explain(r.status_code, body))
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise LLMError(self._explain(500, chunk["error"]))
                    msg = chunk.get("message") or {}
                    if first is None and (msg.get("thinking") or msg.get("content") or msg.get("tool_calls")):
                        first = time.monotonic() - t0
                    if msg.get("thinking"):
                        reasoning.append(msg["thinking"])
                        n += 1
                        if on_progress:
                            on_progress(n, "thinking")
                    if msg.get("content"):
                        content.append(msg["content"])
                        n += 1
                        if on_progress:
                            on_progress(n, "writing")
                    raw_calls.extend(msg.get("tool_calls") or [])
                    if chunk.get("done"):
                        prompt_tokens = chunk.get("prompt_eval_count") or 0
                        completion_tokens = chunk.get("eval_count") or 0
                        done = chunk
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise self._transport_error(exc) from exc
        calls = []
        for rc in raw_calls:
            fn = rc.get("function") or {}
            args, err = parse_arguments(fn.get("arguments"))
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args, error=err,
                                  id=rc.get("id") or f"call_{uuid.uuid4().hex[:10]}"))
        resp = _finalize("".join(content), "".join(reasoning), calls, known_tools)
        resp.prompt_tokens, resp.completion_tokens = prompt_tokens, completion_tokens
        resp.seconds, resp.first_output_seconds = time.monotonic() - t0, first
        ns = 1e9  # Ollama reports durations in nanoseconds
        resp.load_seconds = (done.get("load_duration") or 0) / ns
        resp.prompt_seconds = (done.get("prompt_eval_duration") or 0) / ns
        resp.generation_seconds = (done.get("eval_duration") or 0) / ns
        return resp

    def _explain(self, status: int, body: str) -> str:
        text = body
        try:
            text = json.loads(body).get("error", body)
        except (json.JSONDecodeError, AttributeError):
            pass
        text = str(text)
        if "not found" in text and "model" in text:
            return f"Model '{self.model}' is not installed in Ollama. Run: ollama pull {self.model}"
        if "does not support tools" in text:
            return (f"Model '{self.model}' does not support tool calling in Ollama. Choose a model with the "
                    "'tools' capability, e.g. qwen3.5:9b, gemma4:12b or granite4.2:8b.")
        if "timed out waiting for llama-server" in text or "timed out waiting for llama runner" in text:
            return (f"Ollama could not load '{self.model}' in time: reading the model into memory took longer than "
                    "Ollama's load timeout (5 minutes by default). This happens with a slow or busy disk, too little "
                    "free RAM, or antivirus software scanning the model file. Try again (a second load is faster), "
                    "free some memory, start Ollama with OLLAMA_LOAD_TIMEOUT=15m, or use a smaller model "
                    "(qwen3.5:4b).")
        if "requires more system memory" in text or "out of memory" in text.lower():
            return (f"Not enough free memory to load '{self.model}' with a {self.num_ctx:,}-token context "
                    f"(Ollama: {text[:200]}). Close other apps, use a smaller model (qwen3.5:4b) or lower the "
                    "context with --num-ctx 16384.")
        return f"Ollama error {status}: {text[:500]}"

    def list_models(self) -> list[str]:
        try:
            r = self._http.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise self._transport_error(exc) from exc

    def capabilities(self) -> list[str]:
        r = self._http.post(f"{self.base_url}/api/show", json={"model": self.model})
        if r.status_code != 200:
            return []
        return r.json().get("capabilities") or []


class OpenAICompatLLM(BaseLLM):
    """Any server implementing ``POST /v1/chat/completions`` with tools (streaming)."""

    backend = "openai"

    def _url(self, path: str) -> str:
        base = self.base_url if self.base_url.endswith("/v1") else self.base_url + "/v1"
        return base + path

    def _messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                        for c in m["tool_calls"]
                    ]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": m["content"]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, messages, tools, on_progress=None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(messages),
            "tools": tools,
            "tool_choice": "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,  # non-standard but honoured by llama.cpp, vLLM and SGLang
        }
        if self.think is not None:
            # Qwen3.x / many hybrid-reasoning templates read this flag (vLLM, SGLang, llama.cpp).
            payload["chat_template_kwargs"] = {"enable_thinking": bool(self.think)}
        payload.update(self.extra_body)
        known = {t["function"]["name"] for t in tools}
        t0 = time.monotonic()
        try:
            with self._http.stream("POST", self._url("/chat/completions"), json=payload) as r:
                if r.status_code != 200:
                    body = r.read().decode("utf-8", "replace")
                    raise LLMError(f"Server error {r.status_code}: {body[:500]}")
                if "text/event-stream" not in r.headers.get("content-type", ""):
                    resp = self._parse_full(json.loads(r.read()), known)
                else:
                    resp = self._parse_stream(r, known, on_progress, t0)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise self._transport_error(exc) from exc
        resp.seconds = time.monotonic() - t0
        return resp

    def _parse_stream(self, r: httpx.Response, known: set[str], on_progress, t0: float | None = None) -> LLMResponse:
        content, reasoning = [], []
        acc: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] = {}
        n = 0
        t0 = time.monotonic() if t0 is None else t0
        first = None
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise LLMError(f"Server error: {chunk['error']}")
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if first is None and any(delta.get(k) for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                    first = time.monotonic() - t0
                piece = delta.get("reasoning_content") or delta.get("reasoning")
                if piece:
                    reasoning.append(piece)
                    n += 1
                    if on_progress:
                        on_progress(n, "thinking")
                if delta.get("content"):
                    content.append(delta["content"])
                    n += 1
                    if on_progress:
                        on_progress(n, "writing")
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    slot = acc.setdefault(_delta_index(tc, fn, acc), {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    if fn.get("name") and not slot["name"]:
                        slot["name"] = fn["name"]
                    args = fn.get("arguments")
                    if isinstance(args, dict):
                        slot["arguments"] = json.dumps(args)
                    elif args:
                        slot["arguments"] += args
                    n += 1
                    if on_progress:
                        on_progress(n, "writing")
        calls = []
        for _, slot in sorted(acc.items()):
            args, err = parse_arguments(slot["arguments"])
            calls.append(ToolCall(name=slot["name"], arguments=args, error=err,
                                  id=slot["id"] or f"call_{uuid.uuid4().hex[:10]}"))
        resp = _finalize("".join(content), "".join(reasoning), calls, known)
        resp.prompt_tokens = usage.get("prompt_tokens", 0)
        resp.completion_tokens = usage.get("completion_tokens", 0)
        resp.first_output_seconds = first
        return resp

    def _parse_full(self, data: dict[str, Any], known: set[str]) -> LLMResponse:
        if data.get("error"):
            raise LLMError(f"Server error: {data['error']}")
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args, err = parse_arguments(fn.get("arguments"))
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args, error=err,
                                  id=tc.get("id") or f"call_{uuid.uuid4().hex[:10]}"))
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        resp = _finalize(msg.get("content") or "", reasoning, calls, known)
        usage = data.get("usage") or {}
        resp.prompt_tokens = usage.get("prompt_tokens", 0)
        resp.completion_tokens = usage.get("completion_tokens", 0)
        return resp

    def list_models(self) -> list[str]:
        try:
            r = self._http.get(self._url("/models"))
            r.raise_for_status()
            return [m.get("id", "") for m in r.json().get("data", [])]
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise self._transport_error(exc) from exc


def _delta_index(tc: dict[str, Any], fn: dict[str, Any], acc: dict[int, dict[str, str]]) -> int:
    """Which tool call a streamed delta belongs to.

    Servers normally send ``index``. Some omit it; then a delta that carries a known id
    continues that call, a delta with no id and no name continues the last call, and
    anything else starts a new one.
    """
    if isinstance(tc.get("index"), int):
        return tc["index"]
    call_id = tc.get("id")
    for k, slot in acc.items():
        if call_id and slot["id"] == call_id:
            return k
    if acc and not call_id and not fn.get("name"):
        return max(acc)
    return max(acc) + 1 if acc else 0


DEFAULT_BASE_URLS = {"ollama": "http://localhost:11434", "openai": "http://localhost:8080/v1"}


def make_llm(backend: str, model: str, base_url: str | None = None, **kwargs: Any) -> BaseLLM:
    backend = backend.lower()
    if backend == "ollama":
        return OllamaLLM(model, base_url or DEFAULT_BASE_URLS["ollama"], **kwargs)
    if backend in ("openai", "llamacpp", "llama.cpp", "vllm", "sglang", "localai"):
        return OpenAICompatLLM(model, base_url or DEFAULT_BASE_URLS["openai"], **kwargs)
    raise ValueError(f"unknown backend {backend!r}; use 'ollama' or 'openai'")
