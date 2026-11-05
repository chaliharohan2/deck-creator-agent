"""Speed diagnostics: how long agent steps take on this machine, and why.

On a CPU, a 4B-9B model processes the agent's ~3,500-token prompt (instructions plus tool
definitions) at a few dozen tokens per second and writes at a few tokens per second, so a
step takes a minute or more and nothing is shown while the prompt is being processed. These
helpers turn the timing data the model server reports into plain-language progress messages,
a whole-deck estimate (`deck-agent doctor`) and a check of where Ollama placed the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from deck_agent.llm import LLMResponse

# Calibrated on real runs (qwen3.5 4B/9B, Ollama, 4 vCPU): each step after the first adds
# ~350 new prompt tokens (previous tool result + the model's own call) and writes ~170 tokens.
NEW_PROMPT_TOKENS_PER_STEP = 350
OUTPUT_TOKENS_PER_STEP = 170
STEPS_FOR_TEN_SLIDES = 14  # set_deck_info + 10-11 slides + save_deck + summary


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


@dataclass
class SpeedEstimate:
    prompt_rate: float  # prompt tokens processed per second
    generation_rate: float  # tokens written per second
    load_seconds: float
    first_step_seconds: float
    later_step_seconds: float
    deck_seconds: float  # a ~10-slide deck, without web research

    @property
    def verdict(self) -> str:
        minutes = self.deck_seconds / 60
        if minutes <= 5:
            return "fast"
        if minutes <= 15:
            return "usable"
        if minutes <= 30:
            return "slow"
        return "very slow"

    def summary(self) -> str:
        return (f"prompt {self.prompt_rate:,.0f} tok/s, writing {self.generation_rate:.1f} tok/s -> first step "
                f"~{fmt_duration(self.first_step_seconds)}, then ~{fmt_duration(self.later_step_seconds)} per slide; "
                f"a 10-slide deck ~{fmt_duration(self.deck_seconds)} ({self.verdict})")


def estimate_speed(resp: LLMResponse, prompt_tokens: int) -> SpeedEstimate | None:
    """Estimate agent step times from one realistic request.

    ``prompt_tokens`` is the size of the full agent prompt; it is used when the server does not
    report token counts (some OpenAI-compatible servers) or reused a cached prefix.
    """
    prompt_rate = resp.prompt_rate
    if prompt_rate is None and resp.first_output_seconds:
        busy = resp.first_output_seconds - resp.load_seconds
        prompt_rate = (resp.prompt_tokens or prompt_tokens) / busy if busy > 0 else None
    gen_rate = resp.generation_rate
    if not prompt_rate or not gen_rate:
        return None
    # Loading the model is reported separately: it happens once, then Ollama keeps the model in memory.
    first = prompt_tokens / prompt_rate + OUTPUT_TOKENS_PER_STEP / gen_rate
    later = NEW_PROMPT_TOKENS_PER_STEP / prompt_rate + OUTPUT_TOKENS_PER_STEP / gen_rate
    return SpeedEstimate(prompt_rate, gen_rate, resp.load_seconds, first, later,
                         first + (STEPS_FOR_TEN_SLIDES - 1) * later)


def calibrate_with_text_request(resp: LLMResponse, text_resp: LLMResponse) -> LLMResponse:
    """Fill in prompt/writing times for servers that do not report them (OpenAI-compatible APIs).

    A tool call may arrive in one burst at the end, so the time to first output says little.
    A short plain-text answer streams token by token, which gives the writing speed; the rest of
    the tool request's time is then the time spent reading its prompt.
    """
    window = text_resp.seconds - (text_resp.first_output_seconds or 0)
    if resp.generation_seconds or not text_resp.completion_tokens or window < 0.2:
        return resp
    rate = text_resp.completion_tokens / window
    resp.generation_seconds = resp.completion_tokens / rate if resp.completion_tokens else 0.0
    resp.prompt_seconds = max(resp.seconds - resp.generation_seconds, 0.01)
    return resp


SLOW_ADVICE = (
    "To speed it up: use a smaller model (-m qwen3.5:4b); if you have a GPU, make sure Ollama uses it "
    "(`ollama ps` should say 100% GPU; a model split between GPU and CPU runs at close to CPU speed); close "
    "memory-hungry apps; or lower the context with --num-ctx 16384."
)
SLOW_LOAD_SECONDS = 60
SLOW_LOAD_ADVICE = (
    "Loading the model from disk is slow (slow or busy disk, low free RAM, or antivirus scanning the file). "
    "It only happens on the first request: Ollama keeps the model in memory for 30 minutes after deck-agent uses it. "
    "If a load ever fails with a timeout, start Ollama with OLLAMA_LOAD_TIMEOUT=15m."
)


@dataclass
class Placement:
    text: str
    split: bool  # part of the model is outside accelerator memory


def _ollama_entry(base_url: str, model: str, timeout: float) -> tuple[bool, dict | None]:
    """(server answered, entry for ``model`` in ``GET /api/ps`` or None if it is not loaded)."""
    try:
        with httpx.Client(trust_env=False, timeout=timeout) as c:
            r = c.get(f"{base_url.rstrip('/')}/api/ps")
            r.raise_for_status()
            models = r.json().get("models") or []
    except (httpx.HTTPError, ValueError):
        return False, None
    wanted = {model, f"{model}:latest"}
    return True, next((m for m in models if m.get("name") in wanted or m.get("model") in wanted), None)


def ollama_model_loaded(base_url: str, model: str, timeout: float = 3.0) -> bool | None:
    """Whether Ollama has finished loading ``model`` (None if Ollama could not be asked).

    While a model is still being read into memory, Ollama does not list it in ``/api/ps``.
    """
    answered, entry = _ollama_entry(base_url, model, timeout)
    return (entry is not None) if answered else None


def ollama_placement(base_url: str, model: str, timeout: float = 5.0) -> Placement | None:
    """Where Ollama loaded the model (system RAM, GPU memory or split), from ``GET /api/ps``."""
    _, entry = _ollama_entry(base_url, model, timeout)
    if entry is None:
        return None
    size, vram = entry.get("size") or 0, entry.get("size_vram") or 0
    ctx = entry.get("context_length")
    memory = f"{size / 1e9:.1f} GB" + (f", context {ctx:,} tokens" if ctx else "")
    if size and vram >= size * 0.99:
        return Placement(f"100% in GPU memory ({memory})", False)
    if not vram:
        return Placement(f"system RAM only, i.e. running on the CPU ({memory})", False)
    pct = round(100 * vram / size)
    # Some CPUs' matrix accelerators (Intel AMX) are also reported as "GPU" memory here.
    return Placement(f"{pct}% in GPU/accelerator memory, the rest in system RAM ({memory})", True)


def explain_wait(step: int, seconds: float, tokens: int, phase: str, loading: bool = False,
                 remote: bool = False) -> str:
    """One line describing what the model is doing while we wait for it."""
    if remote and not tokens:
        return (f"step {step}: still waiting for the model server ({fmt_duration(seconds)}). Tool calls are only "
                "shown once complete. If this is much slower than usual, the server may be busy with other "
                "requests or the network may be slow; `deck-agent doctor --mode server` measures it.")
    if loading:
        return (f"step {step}: Ollama is still loading the model into memory ({fmt_duration(seconds)}). The first "
                "load reads the whole model file from disk; with a slow disk, low free RAM or antivirus scanning it "
                "can take minutes, and Ollama gives up after 5 minutes (raise OLLAMA_LOAD_TIMEOUT if it does).")
    if tokens:
        doing = "reasoning (thinking mode is on; turn it off for speed)" if phase == "thinking" else "writing"
        return f"step {step}: model {doing}, {tokens} tokens so far ({fmt_duration(seconds)})"
    if step == 1:
        return (f"step {step}: still waiting for the model ({fmt_duration(seconds)}). It is processing the "
                "~3,500-token agent prompt; on a CPU the first step can take a few minutes and nothing is "
                "shown until it finishes. Later steps are faster.")
    return (f"step {step}: still waiting for the model ({fmt_duration(seconds)}). Tool calls are only shown "
            "once complete; on a CPU each step can take a minute or more.")
