"""The deck-building agent: a tool-calling loop tuned for small local models.

Small models need more structure than frontier models, so the loop adds a few
guard rails:

* invalid tool arguments are returned to the model as precise, fixable errors;
* identical repeated calls are short-circuited instead of executed again;
* if the model stops to *describe* a plan instead of calling tools, it is nudged
  (autonomous mode only);
* old tool results are trimmed when the conversation approaches the context window;
* an unsaved deck is always saved at the end, so a run never loses work.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from deck_agent.config import Settings
from deck_agent.deck.themes import THEMES
from deck_agent.llm import BaseLLM, LLMError, LLMResponse
from deck_agent.tools import DeckSession, ToolRegistry

SYSTEM_PROMPT = """You are DeckSmith, an expert presentation designer and researcher. You create slide decks by calling tools. Today's date is {today}.

# Workflow
1. Work out the topic, audience, tone and slide count (default: about {default_slides} slides).
2. {research_step}
3. Call set_deck_info once (title, subtitle, theme). Themes: {themes}.
4. Add the slides in order, one tool call per slide. A good structure: title -> overview/agenda -> content slides (optionally grouped with section slides) -> summary or recommendations -> closing slide.
5. Call save_deck, then reply with a short summary (2-4 sentences) of the deck.

# Writing good slides
- One idea per slide. Make the title the key message ("Solar costs fell 90% since 2010", not "Solar costs").
- 3-6 bullets per slide, each under 15 words. No paragraphs. Put explanations in `notes` (speaker notes).
- Vary the slide types: two-column for comparisons, table for structured facts, chart for numbers, quote for an impactful statement, section slides between parts.
- Charts need real numbers from the user or from pages you read. Never invent statistics. Values must be plain numbers; put units in y_axis_title.
- Put the URLs of pages you used in the slide's `sources`.
- If a tool returns an error, correct the arguments and call it again.
{mode_rules}"""

RESEARCH_ON = (
    "If the topic needs facts, figures or recent information, research first: call web_search with 1-3 focused "
    "queries, then read_webpage on the 2-3 most useful results. Skip research when the user supplied all the content."
)
RESEARCH_OFF = (
    "Web research is disabled. Use the user's input and your own knowledge; avoid precise statistics you are not sure of."
)
MODE_AUTONOMOUS = "- Work autonomously. Do not ask the user questions; make sensible assumptions and build the whole deck."
MODE_CHAT = (
    "- You are in a conversation. If the request is genuinely unclear, ask one short question; otherwise act.\n"
    "- To change an existing deck use list_slides, update_slide, delete_slide, move_slide or the add_* tools "
    "(with `position`), then call save_deck again."
)

NUDGE = (
    "Continue by calling the tools now: set_deck_info (if not done), then the add_*_slide tools one slide at a time, "
    "then save_deck. Do not describe the plan in text."
)

EventFn = Callable[[str, dict[str, Any]], None]


@dataclass
class AgentResult:
    text: str
    files: dict[str, Path] = field(default_factory=dict)
    steps: int = 0
    tool_calls: int = 0
    errors: int = 0
    stopped_reason: str = "done"


def _signature(name: str, args: dict[str, Any]) -> str:
    return name + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def _short(value: Any, n: int = 80) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


class DeckAgent:
    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        session: DeckSession,
        settings: Settings,
        *,
        interactive: bool = False,
        on_event: EventFn | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.session = session
        self.settings = settings
        self.interactive = interactive
        self.on_event = on_event or (lambda kind, data: None)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt()}]
        self._refresh_schemas()

    def _refresh_schemas(self) -> None:
        # Re-read every step: tools can be switched off mid-request (e.g. exhausted web search).
        self._tool_schemas = self.registry.schemas()
        self._schema_chars = len(json.dumps(self._tool_schemas))

    # ------------------------------------------------------------------ prompt
    def system_prompt(self) -> str:
        web = "web_search" in self.registry
        return SYSTEM_PROMPT.format(
            today=date.today().strftime("%d %B %Y"),
            default_slides=self.settings.default_slides,
            research_step=RESEARCH_ON if web else RESEARCH_OFF,
            themes=", ".join(THEMES),
            mode_rules=MODE_CHAT if self.interactive else MODE_AUTONOMOUS,
        ) + (f"\n- Preferred theme: {self.settings.theme}." if self.settings.theme else "")

    def reset(self) -> None:
        self.session.reset()
        self.messages = [{"role": "system", "content": self.system_prompt()}]

    # ------------------------------------------------------------------ context management
    def _approx_tokens(self) -> int:
        chars = self._schema_chars + sum(
            len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or [])) for m in self.messages
        )
        return int(chars / 3.2)

    def _trim_context(self) -> None:
        budget = int(self.settings.num_ctx * 0.72)
        if self._approx_tokens() <= budget:
            return
        keep_recent = 6
        for limit in (600, 200):
            for m in self.messages[1:-keep_recent]:
                if m["role"] == "tool" and len(m["content"]) > limit:
                    m["content"] = m["content"][:limit] + " …[older tool output trimmed to save context]"
                if self._approx_tokens() <= budget:
                    self.on_event("trimmed", {"tokens": self._approx_tokens()})
                    return
        self.on_event("trimmed", {"tokens": self._approx_tokens()})

    # ------------------------------------------------------------------ main loop
    def run(self, user_message: str) -> AgentResult:
        self.registry.new_request()
        self.messages.append({"role": "user", "content": user_message})
        result = AgentResult(text="")
        nudges = 0
        last_sig, last_output, last_ok, repeats = None, "", True, 0
        start_slides = len(self.session.deck.slides)

        for step in range(1, self.settings.max_steps + 1):
            result.steps = step
            self._refresh_schemas()
            self._trim_context()
            self.on_event("llm_start", {"step": step})
            try:
                response: LLMResponse = self.llm.chat(
                    self.messages, self._tool_schemas,
                    on_progress=lambda n, phase: self.on_event("llm_progress", {"tokens": n, "phase": phase}),
                )
            except LLMError as exc:
                self.on_event("llm_error", {"error": str(exc)})
                result.text = f"Model error: {exc}"
                result.stopped_reason = "llm_error"
                break
            self.on_event("llm_end", {"step": step, "prompt_tokens": response.prompt_tokens,
                                      "completion_tokens": response.completion_tokens,
                                      "recovered": response.recovered_from_text})

            if not response.tool_calls:
                text = response.content.strip()
                built_nothing = len(self.session.deck.slides) <= max(start_slides, 1)
                if not self.interactive and built_nothing and nudges < 2:
                    nudges += 1
                    self.messages.append({"role": "assistant", "content": text})
                    self.messages.append({"role": "user", "content": NUDGE})
                    self.on_event("nudge", {"text": _short(text, 200)})
                    continue
                self.messages.append({"role": "assistant", "content": text})
                result.text = text
                break

            self.messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls],
            })
            for call in response.tool_calls:
                result.tool_calls += 1
                sig = _signature(call.name, call.arguments)
                self.on_event("tool_call", {"name": call.name, "arguments": call.arguments})
                if call.error:
                    ok, output = False, f"Could not parse the arguments for {call.name}: {call.error}"
                elif sig == last_sig:
                    repeats += 1
                    ok = last_ok
                    output = (f"You just made this exact {call.name} call. Its result was: {_short(last_output, 300)} "
                              + ("Do not repeat it; continue with the next step." if last_ok
                                 else "Change the arguments to fix the error instead of repeating the call."))
                else:
                    tool_result = self.registry.execute(call.name, call.arguments)
                    ok, output = tool_result.ok, tool_result.content
                    last_sig, last_output, last_ok, repeats = sig, output, ok, 0
                if not ok:
                    result.errors += 1
                self.on_event("tool_result", {"name": call.name, "ok": ok, "output": output})
                self.messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": output})
            if repeats >= 3:
                result.stopped_reason = "loop"
                result.text = "Stopped: the model kept repeating the same tool call."
                break
        else:
            result.stopped_reason = "max_steps"
            result.text = f"Stopped after {self.settings.max_steps} steps."

        if self.session.dirty and self.session.deck.slides:
            try:
                paths, note = self.session.save()
                self.on_event("autosave", {"files": {k: str(v) for k, v in paths.items()}, "note": note})
            except Exception as exc:  # noqa: BLE001 - report, never lose the run
                self.on_event("autosave_error", {"error": str(exc)})
        result.files = dict(self.session.saved)
        return result
