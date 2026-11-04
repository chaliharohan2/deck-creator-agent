"""Tool registry: JSON-schema generation, argument validation and safe execution."""

from __future__ import annotations

import copy
import json
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

# Fields whose string values are comma-separated lists rather than a single item.
_COMMA_LIST_FIELDS = {"categories", "headers", "formats"}
# Wrapper keys some small models put around the real arguments.
_WRAPPER_KEYS = ("arguments", "parameters", "args", "input", "properties")


class ToolError(Exception):
    """An expected failure whose message is shown to the model so it can recover."""


def _strip_optional(annotation: Any) -> Any:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_list_annotation(annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    if origin in (list, tuple):
        return True
    if origin in (typing.Union, types.UnionType):
        return any(_is_list_annotation(a) for a in typing.get_args(annotation) if a is not type(None))
    return False


def _scalar_to_str(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    return value


def _stringify(annotation: Any, value: Any) -> Any:
    """Convert numbers to strings where the schema expects strings (models often send 2024, not "2024")."""
    annotation = _strip_optional(annotation)
    if annotation is str:
        return _scalar_to_str(value)
    if typing.get_origin(annotation) is list and isinstance(value, list):
        (inner,) = typing.get_args(annotation) or (Any,)
        return [_stringify(inner, v) for v in value]
    return value


class ToolArgs(BaseModel):
    """Base for tool argument models: tolerant of the usual small-model slips."""

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fields = cls.model_fields
        # {"arguments": {...}} wrapping
        if len(data) == 1:
            (key, value), = data.items()
            if key in _WRAPPER_KEYS and key not in fields and isinstance(value, dict):
                data = value
        out = {}
        for key, value in data.items():
            if value is None:
                continue  # fall back to the default
            field = fields.get(key)
            if field is not None and isinstance(value, str) and _is_list_annotation(field.annotation):
                text = value.strip()
                if text.startswith(("[", "{")):
                    try:
                        value = json.loads(text)
                    except json.JSONDecodeError:
                        pass
                if isinstance(value, str):
                    if key in _COMMA_LIST_FIELDS and "\n" not in text:
                        sep = "|" if "|" in text else ","  # "Country | Cost" as well as "Country, Cost"
                        value = [p.strip() for p in text.strip(sep).split(sep) if p.strip()]
                    elif text:
                        value = [line for line in text.splitlines() if line.strip()]
                    else:
                        value = []
            elif field is not None and isinstance(value, str) and value.strip().startswith(("{", "[")) \
                    and field.annotation is not str:
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            if field is not None:
                if "bullets" in key or key == "takeaways":
                    from deck_agent.deck.model import normalize_bullets

                    value = normalize_bullets(value)
                elif key == "sources" and isinstance(value, list):  # "url1, url2" is two sources
                    from deck_agent.deck.model import string_list

                    value = string_list(value)
                value = _stringify(field.annotation, value)
            out[key] = value
        return out


class NoArgs(ToolArgs):
    pass


@dataclass
class ToolResult:
    ok: bool
    content: str


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any], str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": simplify_schema(self.args_model.model_json_schema()),
            },
        }


def simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref``s, collapse ``Optional`` unions and drop titles.

    Small models (and some servers' tool templates) handle flat schemas far
    better than pydantic's default output.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node: Any, in_properties: bool = False) -> Any:
        if isinstance(node, list):
            return [resolve(x) for x in node]
        if not isinstance(node, dict):
            return node
        if in_properties:  # keys are property names (may legitimately be "title")
            return {k: resolve(v) for k, v in node.items()}
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].split("/")[-1]])
            extra = {k: v for k, v in node.items() if k != "$ref"}
            return resolve({**target, **extra})
        if "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            if len(options) == 1:
                rest = {k: v for k, v in node.items() if k != "anyOf"}
                merged = {**resolve(options[0]), **rest}
                if merged.get("default", 0) is None:
                    merged.pop("default")
                return resolve(merged)
        out = {}
        for k, v in node.items():
            if k == "title":
                continue
            out[k] = resolve(v, in_properties=(k == "properties"))
        return out

    result = resolve(schema)
    result.setdefault("type", "object")
    result.setdefault("properties", {})
    return result


def format_validation_error(tool: str, exc: ValidationError) -> str:
    lines = [f"Invalid arguments for {tool}:"]
    for err in exc.errors()[:8]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "arguments"
        msg = str(err.get("msg", "")).removeprefix("Value error, ")
        if err.get("type") == "missing":
            msg = "this required field is missing"
        lines.append(f"- {loc}: {msg}")
    lines.append("Fix these and call the tool again.")
    return "\n".join(lines)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._request_hooks: list[Callable[[], None]] = []
        self._disabled: set[str] = set()

    def on_new_request(self, hook: Callable[[], None]) -> None:
        """Register a callback run at the start of every user request (e.g. to reset budgets)."""
        self._request_hooks.append(hook)

    def new_request(self) -> None:
        self._disabled.clear()
        for hook in self._request_hooks:
            hook()

    def disable(self, name: str) -> None:
        """Hide a tool from the model until the next request.

        Small models often ignore "stop using this tool" messages but reliably stop
        once the tool disappears from the schema list.
        """
        self._disabled.add(name)

    def is_enabled(self, name: str) -> bool:
        return name in self._tools and name not in self._disabled

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def add(self, name: str, description: str, args_model: type[BaseModel] = NoArgs):
        """Decorator form of :meth:`register`."""

        def deco(fn: Callable[[Any], str]) -> Callable[[Any], str]:
            self.register(Tool(name, description, args_model, fn))
            return fn

        return deco

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for n, t in self._tools.items() if n not in self._disabled]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name in self._disabled:
            return ToolResult(False, f"{name} is not available for the rest of this request; continue with the other tools.")
        tool = self._tools.get(name)
        if tool is None:
            close = [n for n in self._tools if name and (name in n or n in name)]
            hint = f" Did you mean {close[0]}?" if close else ""
            return ToolResult(False, f"Unknown tool '{name}'.{hint} Available tools: {', '.join(self._tools)}")
        try:
            args = tool.args_model.model_validate(arguments or {})
        except ValidationError as exc:
            return ToolResult(False, format_validation_error(name, exc))
        try:
            return ToolResult(True, tool.handler(args))
        except ValidationError as exc:  # raised while building a slide model
            return ToolResult(False, format_validation_error(name, exc))
        except (ToolError, ValueError) as exc:
            return ToolResult(False, f"Error in {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - never crash the agent loop on a tool bug
            return ToolResult(False, f"Error in {name}: {exc.__class__.__name__}: {exc}")
