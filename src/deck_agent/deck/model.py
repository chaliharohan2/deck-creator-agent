"""Deck specification: a renderer-independent description of a presentation.

The agent's tools build a :class:`Deck`; renderers turn it into PPTX / HTML / PDF.
A deck can also be saved to and loaded from JSON, so a deck produced by the agent
can be edited by hand and re-rendered without the LLM (``deck-agent render``).

Validators here are deliberately forgiving about the *shape* of input (small
language models often send a bullet list as one newline-separated string, or
numbers as "45%"), but strict about things that would produce a broken slide
(mismatched chart data, oversized tables).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_BULLETS = 10
MAX_TABLE_ROWS = 12
MAX_TABLE_COLS = 7
MAX_CHART_CATEGORIES = 24
MAX_CHART_SERIES = 6

SUB_BULLET_PREFIX = "- "

_LEADING_MARKER = re.compile(r"^\s*(?:[-*•▪◦·]|\d{1,2}[.)])\s+")
_NUMBER_CLEAN = re.compile(r"[,\s$€£¥₹%]")


# --------------------------------------------------------------------------- helpers
def normalize_bullets(value: Any) -> list[str]:
    """Normalize bullet input into a canonical list of strings.

    Canonical form: top-level items are plain text; sub-points start with ``"- "``.

    Accepts a list of strings or a single newline-separated string. If *every*
    item starts with a list marker (``-``, ``*``, ``•``, ``1.``) the markers are
    treated as plain list syntax and stripped; otherwise an item starting with
    ``-``/``*`` or indentation is kept as a sub-point.
    """
    if value is None:
        return []
    if isinstance(value, str):
        items = [line for line in value.splitlines() if line.strip()]
    elif isinstance(value, (list, tuple)):
        items = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):  # {"text": ..., "sub": [...]} style
                text = str(item.get("text") or item.get("bullet") or item.get("title") or "").strip()
                if text:
                    items.append(text)
                for sub in item.get("sub") or item.get("sub_bullets") or item.get("children") or []:
                    items.append(SUB_BULLET_PREFIX + str(sub).strip())
                continue
            items.append(str(item))
    else:
        items = [str(value)]

    items = [i.rstrip() for raw in items for i in _unwrap_json_fragment(raw) if i and i.strip()]
    if not items:
        return []

    all_marked = all(_LEADING_MARKER.match(i) for i in items)
    out: list[str] = []
    for raw in items:
        indent = len(raw) - len(raw.lstrip())
        text = raw.strip()
        marker = _LEADING_MARKER.match(text)
        is_sub = False
        if marker:
            text = text[marker.end():].strip()
            # A marker indented under another item is a sub-point; a bare "- " marker
            # is a sub-point only when the other items are unmarked.
            is_sub = indent >= 2 or not all_marked
        elif indent >= 2:
            is_sub = True
        if not text:
            continue
        if is_sub and out:  # a sub-point needs a parent
            out.append(SUB_BULLET_PREFIX + text)
        else:
            out.append(text)
    return out


def _unwrap_json_fragment(item: str) -> list[str]:
    """Models sometimes send '["a", "b"]' or '"a"' as one bullet; unwrap such fragments."""
    text = item.strip().rstrip(",")
    if text.startswith("[") and text.endswith("]"):
        try:
            value = json.loads(text)
            if isinstance(value, list):
                return [str(v) for v in value if v is not None]
        except json.JSONDecodeError:
            text = text[1:-1].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    indent = item[: len(item) - len(item.lstrip())]
    return [indent + text]


def bullet_level(item: str) -> tuple[int, str]:
    """Return ``(level, text)`` for a canonical bullet string."""
    if item.startswith(SUB_BULLET_PREFIX):
        return 1, item[len(SUB_BULLET_PREFIX):]
    return 0, item


def parse_number(value: Any) -> float:
    """Parse a chart value. Accepts numbers and simple strings like "1,234", "45%", "$12.5"."""
    if isinstance(value, bool):
        raise ValueError("chart values must be numbers, not true/false")
    if isinstance(value, (int, float)):
        number = float(value)
    elif value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a", "na"}):
        return 0.0
    else:
        cleaned = _NUMBER_CLEAN.sub("", str(value).strip()).replace("−", "-")  # Unicode minus sign
        if cleaned.startswith("(") and cleaned.endswith(")"):  # accounting negative
            cleaned = "-" + cleaned[1:-1]
        try:
            number = float(cleaned)
        except ValueError:
            raise ValueError(
                f"chart value {value!r} is not a plain number; use numbers only and put units "
                "(%, $, millions, ...) in the axis title"
            ) from None
    if not math.isfinite(number):  # float("nan") / float("inf") parse, but cannot be drawn
        raise ValueError(f"chart value {value!r} is not a finite number")
    return number


_THOUSANDS = re.compile(r"^\s*[-+(−]?[$€£¥₹]?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*%?\)?\s*$")


def parse_values(v: Any) -> list[float]:
    """Parse a series' values: a list, one number, or a string such as "1, 2, 3" or "1;2;3".

    A comma is a list separator unless the string is one number with thousands
    separators ("1,234"), so "1, 2, 3" gives three values rather than 123.
    """
    if v is None:
        return []
    if not isinstance(v, (list, tuple)):
        v = [v]
    items: list[Any] = []
    for x in v:
        if isinstance(x, str) and re.search(r"[,;\n]", x) and not _THOUSANDS.match(x):
            # "1, 2, 3", "1,234, 5,678" or "1;2;3\n": ";", newlines and ", " always separate;
            # a bare "," separates unless it groups thousands. Empty ends from stray separators are ignored.
            parts = [p for chunk in re.split(r"[;\n]|,\s+", x.strip(" ,;\n"))
                     for p in ([chunk] if _THOUSANDS.match(chunk) else chunk.split(","))]
            items.extend(parts)
        else:
            items.append(x)
    return [parse_number(x) for x in items]


def normalize_series(v: Any) -> Any:
    """Accept the series shapes models produce and return ``[{"name", "values"}, ...]``.

    Handles ``{"Revenue": [1, 2]}``, a bare list of numbers, and items that use
    ``data``/``value``/``y`` instead of ``values`` or ``label`` instead of ``name``.
    """
    if isinstance(v, dict):
        if "values" in v or "data" in v:
            v = [v]
        else:
            return [{"name": k, "values": vals} for k, vals in v.items()]
    if isinstance(v, list) and v and all(isinstance(x, (int, float, str)) for x in v):
        return [{"name": "Series 1", "values": v}]
    if isinstance(v, list):
        fixed = []
        for i, s in enumerate(v):
            if isinstance(s, (list, tuple)):  # [[1, 2], [3, 4]]: bare value arrays
                s = {"name": f"Series {i + 1}", "values": list(s)}
            if isinstance(s, dict) and len(s) == 1:  # [{"Revenue": [1, 2]}]: name -> values
                (key, vals), = s.items()
                if key not in ("values", "data", "value", "y", "points") and isinstance(vals, (list, tuple, str)):
                    s = {"name": key, "values": vals}
            if isinstance(s, dict):
                if "values" not in s:
                    for alt in ("data", "value", "y", "points"):
                        if alt in s:
                            s = {**s, "values": s[alt]}
                            break
                if not s.get("name"):
                    s = {**s, "name": str(s.get("label") or f"Series {i + 1}")}
            fixed.append(s)
        return fixed
    return v


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def string_list(value: Any) -> list[str]:
    """A list of sources/citations from a list or one string ("url1, url2", "url1; url2", lines)."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    out = []
    for v in value:
        if v is None:
            continue
        text = str(v)
        parts = re.split(r"[\n;]|,\s*(?=https?://)", text) if "\n" in text or "http" in text else [text]
        out.extend(p.strip() for p in parts if p.strip())
    return out


def normalize_headers(v: Any) -> list[str]:
    """Table headers from a list or one string ("A | B" or "A, B")."""
    if isinstance(v, str):
        v = [h.strip() for h in v.strip().strip("|").split("|" if "|" in v else ",") if h.strip()]
    elif isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], str) and "|" in v[0]:
        v = [h.strip() for h in v[0].strip().strip("|").split("|") if h.strip()]
    return [_clean_text(h) for h in (v or [])]


_MD_SEPARATOR = re.compile(r"^:?-{2,}:?$")


def normalize_rows(v: Any) -> list[list[str]]:
    """Table rows from lists, dicts ({"Country": "DE", ...}), "a | b" strings or a markdown table."""
    if isinstance(v, str):
        v = [line for line in v.splitlines() if line.strip()]
    rows = []
    for row in v or []:
        if isinstance(row, dict):
            row = list(row.values())
        elif isinstance(row, str):
            row = [c.strip() for c in row.strip().strip("|").split("|")]
        elif not isinstance(row, (list, tuple)):
            row = [row]
        cells = ["" if c is None else _fmt_cell(c) for c in row]
        if cells and all(_MD_SEPARATOR.match(c) for c in cells):  # markdown "|---|---|" line
            continue
        rows.append(cells)
    return rows


# --------------------------------------------------------------------------- slides
class SlideBase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    notes: str = Field("", description="Speaker notes")
    sources: list[str] = Field(default_factory=list, description="Source URLs or citations")

    @field_validator("notes", mode="before")
    @classmethod
    def _notes(cls, v: Any) -> str:
        if isinstance(v, list):
            return "\n".join(str(x) for x in v)
        return _clean_text(v)

    @field_validator("sources", mode="before")
    @classmethod
    def _sources(cls, v: Any) -> list[str]:
        return string_list(v)

    @property
    def heading(self) -> str:
        return getattr(self, "title", "") or getattr(self, "quote", "")[:60]


class TitleSlide(SlideBase):
    kind: Literal["title"] = "title"
    title: str
    subtitle: str = ""
    author: str = ""
    date: str = ""


class SectionSlide(SlideBase):
    kind: Literal["section"] = "section"
    title: str
    subtitle: str = ""


class BulletSlide(SlideBase):
    kind: Literal["bullets"] = "bullets"
    title: str
    bullets: list[str]

    @field_validator("bullets", mode="before")
    @classmethod
    def _bullets(cls, v: Any) -> list[str]:
        return normalize_bullets(v)

    @field_validator("bullets")
    @classmethod
    def _bullets_limits(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one bullet is required")
        if len(v) > MAX_BULLETS:
            raise ValueError(
                f"{len(v)} bullets is too many for one slide (max {MAX_BULLETS}); "
                "split the content across two slides"
            )
        return v


class TwoColumnSlide(SlideBase):
    kind: Literal["two_column"] = "two_column"
    title: str
    left_heading: str = ""
    left_bullets: list[str]
    right_heading: str = ""
    right_bullets: list[str]

    @field_validator("left_bullets", "right_bullets", mode="before")
    @classmethod
    def _bullets(cls, v: Any) -> list[str]:
        return normalize_bullets(v)

    @field_validator("left_bullets", "right_bullets")
    @classmethod
    def _limits(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("each column needs at least one bullet")
        if len(v) > 7:
            raise ValueError(f"{len(v)} bullets is too many for one column (max 7)")
        return v


class ChartSeries(BaseModel):
    name: str = "Series 1"
    values: list[float]

    @field_validator("values", mode="before")
    @classmethod
    def _values(cls, v: Any) -> list[float]:
        return parse_values(v)


ChartType = Literal["column", "bar", "line", "pie", "doughnut", "area", "stacked_column", "stacked_bar"]

_CHART_ALIASES = {
    "column": "column", "columns": "column", "vertical_bar": "column", "bar_vertical": "column",
    "clustered_column": "column", "histogram": "column",
    "bar": "bar", "horizontal_bar": "bar", "hbar": "bar", "bars": "bar", "clustered_bar": "bar",
    "line": "line", "lines": "line", "line_chart": "line", "trend": "line",
    "pie": "pie", "pie_chart": "pie",
    "doughnut": "doughnut", "donut": "doughnut", "ring": "doughnut",
    "area": "area",
    "stacked_column": "stacked_column", "stacked": "stacked_column", "stacked_bar": "stacked_bar",
}


def normalize_chart_type(v: Any) -> str:
    key = re.sub(r"[\s-]+", "_", str(v or "column").strip().lower()).removesuffix("_chart")
    if key not in _CHART_ALIASES:
        raise ValueError(f"unknown chart_type {v!r}; use one of: {', '.join(ChartType.__args__)}")
    return _CHART_ALIASES[key]


class ChartSlide(SlideBase):
    kind: Literal["chart"] = "chart"
    title: str
    chart_type: ChartType = "column"
    categories: list[str]
    series: list[ChartSeries]
    y_axis_title: str = ""
    x_axis_title: str = ""
    bullets: list[str] = Field(default_factory=list, description="Optional key takeaways shown beside the chart")

    @field_validator("chart_type", mode="before")
    @classmethod
    def _chart_type(cls, v: Any) -> str:
        return normalize_chart_type(v)

    @field_validator("categories", mode="before")
    @classmethod
    def _categories(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = [c.strip() for c in re.split(r"[,\n]", v) if c.strip()]
        return [str(c) for c in (v or [])]

    @field_validator("series", mode="before")
    @classmethod
    def _series(cls, v: Any) -> Any:
        return normalize_series(v)

    @field_validator("bullets", mode="before")
    @classmethod
    def _bullets(cls, v: Any) -> list[str]:
        return normalize_bullets(v)

    @model_validator(mode="after")
    def _check_data(self) -> ChartSlide:
        if not self.categories:
            raise ValueError("categories must not be empty")
        if not self.series:
            raise ValueError("series must contain at least one {name, values} entry")
        if len(self.categories) > MAX_CHART_CATEGORIES:
            raise ValueError(f"too many categories ({len(self.categories)}, max {MAX_CHART_CATEGORIES})")
        if len(self.series) > MAX_CHART_SERIES:
            raise ValueError(f"too many series ({len(self.series)}, max {MAX_CHART_SERIES})")
        for s in self.series:
            if len(s.values) != len(self.categories):
                raise ValueError(
                    f"series '{s.name}' has {len(s.values)} values but there are "
                    f"{len(self.categories)} categories; they must match one-to-one"
                )
        if self.chart_type in ("pie", "doughnut"):
            if len(self.series) > 1:
                raise ValueError(f"a {self.chart_type} chart takes exactly one series; use a column chart to compare series")
            if any(v < 0 for v in self.series[0].values):
                raise ValueError(f"a {self.chart_type} chart cannot show negative values")
            if sum(self.series[0].values) <= 0:
                raise ValueError(f"a {self.chart_type} chart needs at least one positive value")
        if len(self.bullets) > 5:
            raise ValueError("at most 5 takeaway bullets fit beside a chart")
        return self


class TableSlide(SlideBase):
    kind: Literal["table"] = "table"
    title: str
    headers: list[str]
    rows: list[list[str]]

    @field_validator("headers", mode="before")
    @classmethod
    def _headers(cls, v: Any) -> list[str]:
        return normalize_headers(v)

    @field_validator("rows", mode="before")
    @classmethod
    def _rows(cls, v: Any) -> list[list[str]]:
        return normalize_rows(v)

    @model_validator(mode="after")
    def _check_shape(self) -> TableSlide:
        if not self.headers:
            raise ValueError("headers must not be empty")
        if len(self.headers) > MAX_TABLE_COLS:
            raise ValueError(f"too many columns ({len(self.headers)}, max {MAX_TABLE_COLS}); split the table")
        if not self.rows:
            raise ValueError("rows must not be empty")
        if len(self.rows) > MAX_TABLE_ROWS:
            raise ValueError(f"too many rows ({len(self.rows)}, max {MAX_TABLE_ROWS}); split across two slides")
        n = len(self.headers)
        for i, row in enumerate(self.rows):
            if len(row) > n:
                raise ValueError(f"row {i + 1} has {len(row)} cells but there are only {n} headers")
            if len(row) < n:
                row.extend([""] * (n - len(row)))
        return self


def _fmt_cell(c: Any) -> str:
    if isinstance(c, float) and c.is_integer():
        return str(int(c))
    return str(c).strip()


class ImageSlide(SlideBase):
    kind: Literal["image"] = "image"
    title: str
    image_path: str = Field(description="Local path of the (already downloaded) image")
    caption: str = ""
    credit: str = ""
    bullets: list[str] = Field(default_factory=list)

    @field_validator("bullets", mode="before")
    @classmethod
    def _bullets(cls, v: Any) -> list[str]:
        return normalize_bullets(v)

    @field_validator("bullets")
    @classmethod
    def _limits(cls, v: list[str]) -> list[str]:
        if len(v) > 6:
            raise ValueError("at most 6 bullets fit beside an image")
        return v


class QuoteSlide(SlideBase):
    kind: Literal["quote"] = "quote"
    quote: str
    attribution: str = ""

    @field_validator("quote", mode="before")
    @classmethod
    def _quote(cls, v: Any) -> str:
        return _clean_text(v).strip("\"'“”")

    @field_validator("quote")
    @classmethod
    def _limit(cls, v: str) -> str:
        if not v:
            raise ValueError("quote must not be empty")
        if len(v) > 300:
            raise ValueError("quote is too long for a slide (max 300 characters); shorten it")
        return v


Slide = Annotated[
    TitleSlide | SectionSlide | BulletSlide | TwoColumnSlide | ChartSlide | TableSlide | ImageSlide | QuoteSlide,
    Field(discriminator="kind"),
]

SLIDE_TYPES: dict[str, type[SlideBase]] = {
    "title": TitleSlide,
    "section": SectionSlide,
    "bullets": BulletSlide,
    "two_column": TwoColumnSlide,
    "chart": ChartSlide,
    "table": TableSlide,
    "image": ImageSlide,
    "quote": QuoteSlide,
}


# --------------------------------------------------------------------------- deck
class Deck(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = "Untitled presentation"
    subtitle: str = ""
    author: str = ""
    date: str = ""
    theme: str = "light"
    slides: list[Slide] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> Deck:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def outline(self, detail: bool = True) -> str:
        """Human/LLM-readable outline of the deck."""
        if not self.slides:
            return "(the deck has no slides yet)"
        lines = [f'Deck: "{self.title}" (theme: {self.theme}, {len(self.slides)} slides)']
        for i, s in enumerate(self.slides, 1):
            lines.append(f"{i}. [{s.kind}] {s.heading}")
            if not detail:
                continue
            if isinstance(s, (TitleSlide, SectionSlide)) and s.subtitle:
                lines.append(f"     subtitle: {s.subtitle}")
            if isinstance(s, (BulletSlide, ImageSlide, ChartSlide)) and s.bullets:
                for b in s.bullets[:6]:
                    lines.append(f"     • {_short(b, 90)}")
            if isinstance(s, TwoColumnSlide):
                lines.append(f"     left: {s.left_heading} ({len(s.left_bullets)} bullets)")
                lines.append(f"     right: {s.right_heading} ({len(s.right_bullets)} bullets)")
            if isinstance(s, ChartSlide):
                lines.append(
                    f"     {s.chart_type} chart, {len(s.categories)} categories, "
                    f"series: {', '.join(x.name for x in s.series)}"
                )
            if isinstance(s, TableSlide):
                lines.append(f"     table {len(s.rows)}x{len(s.headers)}: {', '.join(s.headers)}")
            if isinstance(s, QuoteSlide):
                lines.append(f"     — {s.attribution}")
        return "\n".join(lines)


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"
