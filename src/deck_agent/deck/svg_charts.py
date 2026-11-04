"""Dependency-free SVG charts for the HTML renderer."""

from __future__ import annotations

import math
from html import escape

from deck_agent.deck.model import ChartSlide
from deck_agent.deck.text import mix
from deck_agent.deck.themes import Theme


def _fmt(v: float, decimals: int) -> str:
    return f"{v:,.{decimals}f}"


def _decimals(values: list[float]) -> int:
    if all(float(v).is_integer() for v in values):
        return 0
    if all(abs(v * 10 - round(v * 10)) < 1e-9 for v in values):
        return 1
    return 2


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    if hi == lo:
        hi = lo + 1
    raw = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v < hi + step * 0.999:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _text(x, y, s, *, size=13, fill="#000", anchor="middle", weight="normal", baseline="auto") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="#{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}" dominant-baseline="{baseline}">{escape(str(s))}</text>'
    )


def _truncate(label: str, n: int) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def render_svg_chart(s: ChartSlide, theme: Theme, width: int, height: int) -> str:
    if s.chart_type in ("pie", "doughnut"):
        body = _pie(s, theme, width, height)
    else:
        body = _cartesian(s, theme, width, height)
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(s.title)}">{body}</svg>'
    )


def _legend(names: list[str], theme: Theme, x0: float, y: float, width: float) -> str:
    out, x = [], x0
    item_w = min(220, width / max(len(names), 1))
    for i, name in enumerate(names):
        color = theme.palette[i % len(theme.palette)]
        out.append(f'<rect x="{x:.1f}" y="{y - 10:.1f}" width="14" height="14" rx="3" fill="#{color}"/>')
        out.append(_text(x + 20, y + 2, _truncate(name, 26), size=14, fill=theme.text, anchor="start"))
        x += item_w
    return "".join(out)


def _cartesian(s: ChartSlide, theme: Theme, W: int, H: int) -> str:
    horizontal = s.chart_type in ("bar", "stacked_bar")
    stacked = s.chart_type.startswith("stacked")
    cats, series = s.categories, s.series
    n, m = len(cats), len(series)
    all_vals = [v for x in series for v in x.values]
    dec = _decimals(all_vals)
    if stacked:
        pos = [sum(max(x.values[i], 0) for x in series) for i in range(n)]
        neg = [sum(min(x.values[i], 0) for x in series) for i in range(n)]
        lo, hi = min(0.0, *neg), max(0.0, *pos)
    else:
        lo, hi = min(0.0, *all_vals), max(0.0, *all_vals)
    ticks = nice_ticks(lo, hi)
    lo, hi = ticks[0], ticks[-1]
    span = (hi - lo) or 1
    grid = mix(theme.background, theme.text, 0.15)
    legend_h = 34 if m > 1 else 0
    longest_cat = max(len(c) for c in cats)
    pad_l = (min(longest_cat, 22) * 8 + 24) if horizontal else 64 + (12 if s.y_axis_title else 0)
    pad_b = (36 if horizontal else 44) + legend_h + (22 if (s.x_axis_title if not horizontal else s.y_axis_title) else 0)
    pad_t, pad_r = 22, 24
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    out: list[str] = []

    def vpos(v: float) -> float:  # value -> pixel along value axis
        frac = (v - lo) / span
        return pad_l + frac * pw if horizontal else pad_t + ph - frac * ph

    # grid + value tick labels
    for t in ticks:
        p = vpos(t)
        if horizontal:
            out.append(f'<line x1="{p:.1f}" y1="{pad_t}" x2="{p:.1f}" y2="{pad_t + ph}" stroke="#{grid}"/>')
            out.append(_text(p, pad_t + ph + 22, _fmt(t, dec if dec else 0), size=13, fill=theme.muted))
        else:
            out.append(f'<line x1="{pad_l}" y1="{p:.1f}" x2="{pad_l + pw}" y2="{p:.1f}" stroke="#{grid}"/>')
            out.append(_text(pad_l - 10, p, _fmt(t, dec if dec else 0), size=13, fill=theme.muted, anchor="end",
                             baseline="middle"))

    band = (ph if horizontal else pw) / n
    zero = vpos(0)
    show_labels = m == 1 and n <= 12 and s.chart_type in ("column", "bar", "line")

    if s.chart_type in ("line", "area"):
        for j, ser in enumerate(series):
            color = theme.palette[j % len(theme.palette)]
            pts = [(pad_l + band * (i + 0.5), vpos(v)) for i, v in enumerate(ser.values)]
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            if s.chart_type == "area":
                poly = f"{pts[0][0]:.1f},{zero:.1f} {path} {pts[-1][0]:.1f},{zero:.1f}"
                out.append(f'<polygon points="{poly}" fill="#{color}" fill-opacity="0.55" stroke="#{color}" stroke-width="2"/>')
            else:
                out.append(f'<polyline points="{path}" fill="none" stroke="#{color}" stroke-width="3.5" '
                           'stroke-linejoin="round" stroke-linecap="round"/>')
                for x, y in pts:
                    out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#{color}"/>')
            if show_labels:
                for (x, y), v in zip(pts, ser.values, strict=True):
                    out.append(_text(x, y - 12, _fmt(v, dec), size=13, fill=theme.text))
    else:
        group = band * 0.72
        bar = group if stacked else group / m
        offsets_pos = [0.0] * n
        offsets_neg = [0.0] * n
        for j, ser in enumerate(series):
            color = theme.palette[j % len(theme.palette)]
            for i, v in enumerate(ser.values):
                start = band * i + (band - group) / 2 + (0 if stacked else j * bar)
                if stacked:
                    base = offsets_pos[i] if v >= 0 else offsets_neg[i]
                    a, b = vpos(base), vpos(base + v)
                    if v >= 0:
                        offsets_pos[i] += v
                    else:
                        offsets_neg[i] += v
                else:
                    a, b = zero, vpos(v)
                lo_px, size_px = min(a, b), abs(b - a)
                if horizontal:
                    y = pad_t + start
                    out.append(f'<rect x="{lo_px:.1f}" y="{y:.1f}" width="{size_px:.1f}" height="{bar:.1f}" rx="2" fill="#{color}"/>')
                    if show_labels:
                        out.append(_text(max(a, b) + 8, y + bar / 2, _fmt(v, dec), size=13, fill=theme.text,
                                         anchor="start", baseline="middle"))
                else:
                    x = pad_l + start
                    out.append(f'<rect x="{x:.1f}" y="{lo_px:.1f}" width="{bar:.1f}" height="{size_px:.1f}" rx="2" fill="#{color}"/>')
                    if show_labels:
                        out.append(_text(x + bar / 2, min(a, b) - 8, _fmt(v, dec), size=13, fill=theme.text))

    # category labels + axis line
    max_chars = max(6, int(band / 8)) if not horizontal else 24
    for i, c in enumerate(cats):
        if horizontal:
            y = pad_t + band * (i + 0.5)
            out.append(_text(pad_l - 10, y, _truncate(c, max_chars), size=14, fill=theme.text, anchor="end",
                             baseline="middle"))
        else:
            x = pad_l + band * (i + 0.5)
            out.append(_text(x, pad_t + ph + 24, _truncate(c, max_chars), size=14 if n <= 8 else 12, fill=theme.text))
    if horizontal:
        out.append(f'<line x1="{zero:.1f}" y1="{pad_t}" x2="{zero:.1f}" y2="{pad_t + ph}" stroke="#{theme.muted}"/>')
    else:
        out.append(f'<line x1="{pad_l}" y1="{zero:.1f}" x2="{pad_l + pw}" y2="{zero:.1f}" stroke="#{theme.muted}"/>')

    # axis titles
    value_title, cat_title = s.y_axis_title, s.x_axis_title
    if horizontal:
        if value_title:
            out.append(_text(pad_l + pw / 2, pad_t + ph + 46, value_title, size=13, fill=theme.muted))
    else:
        if value_title:
            out.append(f'<text transform="translate(16,{pad_t + ph / 2:.1f}) rotate(-90)" font-size="13" '
                       f'fill="#{theme.muted}" text-anchor="middle">{escape(value_title)}</text>')
        if cat_title:
            out.append(_text(pad_l + pw / 2, pad_t + ph + 48, cat_title, size=13, fill=theme.muted))
    if m > 1:
        out.append(_legend([x.name for x in series], theme, pad_l, H - 12, pw))
    return "".join(out)


def _pie(s: ChartSlide, theme: Theme, W: int, H: int) -> str:
    values = s.series[0].values
    total = sum(values) or 1
    legend_w = min(360, W * 0.38)
    r = min((W - legend_w) / 2, H / 2) - 24
    cx, cy = (W - legend_w) / 2, H / 2
    inner = r * 0.55 if s.chart_type == "doughnut" else 0
    out, angle = [], -math.pi / 2
    for i, v in enumerate(values):
        frac = v / total
        if frac <= 0:
            continue
        color = theme.palette[i % len(theme.palette)]
        end = angle + frac * 2 * math.pi
        if frac >= 0.9999:
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="#{color}"/>')
        else:
            large = 1 if frac > 0.5 else 0
            x1, y1 = cx + r * math.cos(angle), cy + r * math.sin(angle)
            x2, y2 = cx + r * math.cos(end), cy + r * math.sin(end)
            out.append(f'<path d="M{cx:.1f},{cy:.1f} L{x1:.1f},{y1:.1f} A{r:.1f},{r:.1f} 0 {large} 1 {x2:.1f},{y2:.1f} Z" '
                       f'fill="#{color}" stroke="#{theme.background}" stroke-width="2"/>')
        mid = (angle + end) / 2
        lr = (r + inner) / 2 if inner else r * 0.66
        if frac >= 0.04:
            out.append(_text(cx + lr * math.cos(mid), cy + lr * math.sin(mid), f"{frac:.0%}", size=16,
                             fill="FFFFFF", weight="bold", baseline="middle"))
        angle = end
    if inner:
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{inner:.1f}" fill="#{theme.background}"/>')
    lx = W - legend_w + 10
    ly = cy - (len(values) * 30) / 2 + 10
    for i, c in enumerate(s.categories):
        color = theme.palette[i % len(theme.palette)]
        y = ly + i * 30
        out.append(f'<rect x="{lx:.1f}" y="{y - 11:.1f}" width="16" height="16" rx="3" fill="#{color}"/>')
        out.append(_text(lx + 26, y + 3, _truncate(c, 30), size=15, fill=theme.text, anchor="start"))
    return "".join(out)
