"""Small text utilities shared by the renderers."""

from __future__ import annotations

import math
import re
from urllib.parse import urlparse

_BOLD = re.compile(r"(\*\*[^*]+\*\*|__[^_]+__)")
_STRAY_MD = re.compile(r"(?<!\w)[*_`]{1,2}|[*_`]{1,2}(?!\w)")


def parse_inline(text: str) -> list[tuple[str, bool]]:
    """Split text into ``(segment, is_bold)`` runs, honouring ``**bold**`` markdown.

    Other stray markdown characters (single ``*``, backticks) are removed because
    slide text is not rendered as markdown.
    """
    out: list[tuple[str, bool]] = []
    for part in _BOLD.split(text):
        if not part:
            continue
        if (part.startswith("**") and part.endswith("**")) or (part.startswith("__") and part.endswith("__")):
            seg = part[2:-2]
            if seg:
                out.append((seg, True))
        else:
            seg = _STRAY_MD.sub("", part)
            if seg:
                out.append((seg, False))
    return out or [("", False)]


def plain(text: str) -> str:
    return "".join(seg for seg, _ in parse_inline(text))


def short_source(src: str, max_len: int = 48) -> str:
    """Compact display form of a source (domain + short path for URLs)."""
    src = src.strip()
    if re.match(r"^https?://", src):
        u = urlparse(src)
        host = u.netloc.removeprefix("www.")
        path = u.path.rstrip("/")
        label = host + (path if len(host + path) <= max_len else "")
        if not path or len(host + path) > max_len:
            label = host if not path else (host + path)[: max_len - 1] + "…"
        return label
    return src if len(src) <= max_len else src[: max_len - 1] + "…"


def sources_line(sources: list[str], max_items: int = 3, max_chars: int = 150) -> str:
    if not sources:
        return ""
    items = [short_source(s) for s in sources[:max_items]]
    line = "Sources: " + " · ".join(items)
    if len(sources) > max_items:
        line += f" (+{len(sources) - max_items} more in notes)"
    return line if len(line) <= max_chars else line[: max_chars - 1] + "…"


def estimate_lines(text: str, width_in: float, size_pt: float, indent_in: float = 0.0) -> int:
    """Rough number of wrapped lines for ``text`` in a box ``width_in`` wide."""
    avg_char_in = size_pt * 0.52 / 72  # average glyph width for common sans fonts
    usable = max(width_in - indent_in, 0.5)
    per_line = max(int(usable / avg_char_in), 8)
    return max(1, math.ceil(len(plain(text)) / per_line))


def fit_font_size(
    items: list[tuple[int, str]],
    width_in: float,
    height_in: float,
    sizes: tuple[int, ...] = (26, 24, 22, 20, 18, 16, 14, 13, 12),
    line_spacing: float = 1.18,
    para_gap: float = 0.55,
) -> int:
    """Largest font size from ``sizes`` at which the (level, text) items fit the box."""
    for size in sizes:
        total = 0.0
        for i, (level, text) in enumerate(items):
            s = size if level == 0 else max(size - 4, 11)
            lines = estimate_lines(text, width_in, s, indent_in=0.35 + 0.4 * level)
            total += lines * s * line_spacing / 72
            if i:
                total += (s * para_gap if level == 0 else s * 0.25) / 72
        if total <= height_in:
            return size
    return sizes[-1]


def is_numeric(text: str) -> bool:
    """A table cell holding a plain number ("1,200", "-3.5%", "$40"): aligned right."""
    cleaned = re.sub(r"[,%$€£¥₹\s]", "", text).lstrip("+-\u2212")
    if not any(ch.isdigit() for ch in cleaned):  # "nan", "inf", "Infinity" are words here
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def table_column_weights(headers: list[str], rows: list[list[str]]) -> list[int]:
    """Relative column widths: proportional to the longest text in each column (clamped)."""
    return [min(max(max(len(plain(row[c])) for row in [headers, *rows]), 6), 40) for c in range(len(headers))]


def fit_table(
    headers: list[str],
    rows: list[list[str]],
    col_widths_in: list[float],
    height_in: float,
    max_size: int = 18,
    cell_pad_x_in: float = 0.24,
    cell_pad_y_in: float = 0.1,
    min_size: int = 10,
) -> tuple[int, list[float]]:
    """Largest font size (<= ``max_size``) at which the table fits ``height_in``, and its row heights.

    Rows get a roomy height when there is space and shrink to their text otherwise; cells that
    wrap are counted with :func:`estimate_lines`.
    """
    heights: list[float] = []
    for size in range(max_size, min_size - 1, -1):
        text_h = [
            max(estimate_lines(cell, w, size, indent_in=cell_pad_x_in) for cell, w in zip(row, col_widths_in, strict=True))
            * size * 1.2 / 72 + cell_pad_y_in
            for row in [headers, *rows]
        ]
        roomy = max(0.42, size * 2.1 / 72)
        for heights in ([max(roomy, h) for h in text_h], text_h):
            if sum(heights) <= height_in:
                return size, heights
    return min_size, heights


def mix(c1: str, c2: str, t: float) -> str:
    """Blend two hex colours: t=0 -> c1, t=1 -> c2."""
    a = [int(c1[i : i + 2], 16) for i in (0, 2, 4)]
    b = [int(c2[i : i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b, strict=True))


def number_format(values: list[float]) -> tuple[str, int]:
    """Excel-style number format and decimal count suited to the values."""
    if all(float(v).is_integer() for v in values):
        return "#,##0", 0
    if all(abs(v * 10 - round(v * 10)) < 1e-9 for v in values):
        return "#,##0.0", 1
    return "#,##0.00", 2
