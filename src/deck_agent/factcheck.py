"""Cross-check numbers on slides against the text the agent has actually seen.

Small models invent plausible statistics, even when they cite a real source. This module
keeps a corpus of every number that appeared in the user's input, search snippets and pages
the agent read, and checks whether a number written on a slide occurs there, at the
precision the model wrote it and allowing for scaling (528,000 -> 528 thousand -> 0.528
million) and percent <-> fraction conversions.

Limits, stated plainly: this catches numbers that appear *nowhere* in the material, which
is the common failure. It cannot prove a number is used in the right context. With very
number-heavy source pages, a chance match also becomes more likely.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(\s*(?:%|percent|per cent))?")
_SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9, 100.0, 0.01)


def _sig_digits(digits: str) -> int:
    """Significant digits of a written number ("1450" -> 3, "29.3" -> 3, "0.05" -> 1)."""
    if "." in digits:
        return max(1, len(digits.replace(".", "").lstrip("0")))
    return max(1, len(digits.lstrip("0").rstrip("0")))


def _round_sig(x: float, sig: int) -> float:
    if x == 0:
        return 0.0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


@dataclass(frozen=True)
class Figure:
    text: str  # as written, e.g. "528,000" or "13%"
    value: float
    sig: int


def find_figures(text: str) -> list[Figure]:
    """Numbers worth checking: skips years and small bare integers ("top 3", "Q4")."""
    out = []
    for m in _NUM.finditer(text or ""):
        int_part, frac, pct = m.group(1), m.group(2) or "", m.group(3)
        digits = int_part.replace(",", "") + frac
        value = float(digits)
        if not pct and not frac:
            if 1900 <= value <= 2100 or value < 10:
                continue
        out.append(Figure(m.group(0).strip(), value, _sig_digits(digits)))
    return out


def number_figure(value: float) -> Figure:
    """A chart value as a Figure, with the precision the model gave it."""
    text = f"{value:.10g}"
    return Figure(text, float(value), _sig_digits(text.lstrip("-")))


@dataclass
class NumberCorpus:
    values: set[float] = field(default_factory=set)

    def add_text(self, text: str) -> None:
        for m in _NUM.finditer(text or ""):
            self.values.add(float(m.group(1).replace(",", "") + (m.group(2) or "")))

    def __len__(self) -> int:
        return len(self.values)

    def supports(self, fig: Figure) -> bool:
        target = abs(fig.value)
        if target == 0:
            return True
        # A one-digit value ("3", "50%") is too coarse for rounding: 3 would "match" 2.9 or 3.4.
        exact = fig.sig < 2
        for c in self.values:
            for scale in _SCALES:
                x = c * scale
                if not x:
                    continue
                candidate = x if exact else _round_sig(x, fig.sig)
                if abs(candidate - target) <= target * 1e-9:
                    return True
        return False

    def unsupported(self, figures: list[Figure]) -> list[Figure]:
        return [f for f in figures if not self.supports(f)]
