"""Render a :class:`Deck` to a single self-contained HTML file.

No CDN or JavaScript framework is used, so the file works fully offline:
arrow keys / space / click to navigate, ``N`` toggles speaker notes, ``F`` goes
full screen, and printing (or "Save as PDF" in any browser) produces one slide
per page.
"""

from __future__ import annotations

import base64
import io
import mimetypes
from html import escape
from pathlib import Path

from PIL import Image

from deck_agent.deck.model import (
    BulletSlide,
    ChartSlide,
    Deck,
    ImageSlide,
    QuoteSlide,
    SectionSlide,
    SlideBase,
    TableSlide,
    TitleSlide,
    TwoColumnSlide,
    bullet_level,
)
from deck_agent.deck.svg_charts import render_svg_chart
from deck_agent.deck.text import (
    fit_font_size,
    fit_table,
    is_numeric,
    mix,
    parse_inline,
    short_source,
    sources_line,
    table_column_weights,
)
from deck_agent.deck.themes import Theme, get_theme

PX_PER_IN = 96
SLIDE_W_PX, SLIDE_H_PX = 1280, 720


def _inline(text: str) -> str:
    return "".join(f"<strong>{escape(seg)}</strong>" if bold else escape(seg) for seg, bold in parse_inline(text))


def _pt(size: float) -> str:
    return f"{size * 96 / 72:.1f}px"


def _bullets_html(items: list[str], width_px: float, height_px: float, max_size: int = 26) -> str:
    parsed = [bullet_level(i) for i in items]
    sizes = tuple(s for s in (26, 24, 22, 20, 18, 16, 14, 13, 12) if s <= max_size)
    size = fit_font_size(parsed, width_px / PX_PER_IN, height_px / PX_PER_IN, sizes=sizes)
    out = [f'<ul class="bullets" style="font-size:{_pt(size)}">']
    open_sub = False
    for level, text in parsed:
        if level == 1 and not open_sub:
            out.append('<ul class="sub">')
            open_sub = True
        elif level == 0 and open_sub:
            out.append("</ul>")
            open_sub = False
        out.append(f"<li>{_inline(text)}</li>")
    if open_sub:
        out.append("</ul>")
    out.append("</ul>")
    return "".join(out)


def _data_uri(path: Path) -> str:
    """The image as a data: URI. Raises OSError if the file is missing or is not a readable image."""
    data = path.read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except Exception as exc:  # noqa: BLE001 - Pillow raises various errors for corrupt files
        raise OSError(f"not a readable image: {path}") from exc
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


class HtmlRenderer:
    def __init__(self, deck: Deck, theme: Theme | None = None):
        self.deck = deck
        self.theme = theme or get_theme(deck.theme)

    def render(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def to_html(self) -> str:
        total = len(self.deck.slides)
        sections = []
        for i, slide in enumerate(self.deck.slides, 1):
            inner = getattr(self, f"_slide_{slide.kind}")(slide)
            cover = isinstance(slide, (TitleSlide, SectionSlide))
            chrome = "" if cover else (
                f'<footer><span>{escape(self.deck.title)}</span><span>{i} / {total}</span></footer>'
            )
            src = self._sources(slide)
            notes = self._notes(slide)
            cls = "slide cover" if cover else "slide"
            sections.append(f'<section class="{cls}" data-kind="{slide.kind}">{inner}{src}{chrome}{notes}</section>')
        html = _TEMPLATE.format(
            title=escape(self.deck.title),
            css=self._css(),
            slides="\n".join(sections),
            js=_JS,
        )
        return html

    # ------------------------------------------------------------------ pieces
    def _css(self) -> str:
        t = self.theme
        return _CSS.format(
            bg=t.background, surface=t.surface, title=t.title, text=t.text, muted=t.muted, accent=t.accent,
            accent_text=t.accent_text, cover_bg=t.cover_background, cover_text=t.cover_text,
            cover_sub=mix(t.cover_background, t.cover_text, 0.8), cover_meta=mix(t.cover_background, t.cover_text, 0.6),
            heading_font=t.heading_font, body_font=t.body_font, page_bg=mix(t.background, "000000", 0.82),
        )

    def _sources(self, s: SlideBase) -> str:
        line = sources_line(s.sources)
        credit = getattr(s, "credit", "")
        if credit:
            line = f"Image: {credit}" + (f"   {line}" if line else "")
        return f'<div class="sources">{escape(line)}</div>' if line else ""

    def _notes(self, s: SlideBase) -> str:
        parts = []
        if s.notes:
            parts.append(f"<p>{escape(s.notes)}</p>")
        if s.sources:
            links = "".join(
                f'<li><a href="{escape(x)}">{escape(short_source(x, 80))}</a></li>' if x.startswith("http")
                else f"<li>{escape(x)}</li>" for x in s.sources
            )
            parts.append(f"<p>Sources:</p><ul>{links}</ul>")
        return f'<aside class="notes">{"".join(parts)}</aside>' if parts else ""

    def _head(self, title: str) -> str:
        return f'<header><h2>{_inline(title)}</h2><div class="bar"></div></header>'

    # ------------------------------------------------------------------ slide types
    def _slide_title(self, s: TitleSlide) -> str:
        sub = s.subtitle or self.deck.subtitle
        meta = " · ".join(x for x in (s.author or self.deck.author, s.date or self.deck.date) if x)
        return (
            '<div class="cover-body"><div class="bar big"></div>'
            f"<h1>{_inline(s.title)}</h1>"
            + (f'<p class="subtitle">{_inline(sub)}</p>' if sub else "")
            + "</div>"
            + (f'<p class="meta">{escape(meta)}</p>' if meta else "")
        )

    def _slide_section(self, s: SectionSlide) -> str:
        return (
            f'<div class="cover-body"><div class="bar"></div><h1 class="section">{_inline(s.title)}</h1>'
            + (f'<p class="subtitle">{_inline(s.subtitle)}</p>' if s.subtitle else "")
            + "</div>"
        )

    def _slide_bullets(self, s: BulletSlide) -> str:
        return self._head(s.title) + f'<div class="body">{_bullets_html(s.bullets, 1110, 450)}</div>'

    def _slide_two_column(self, s: TwoColumnSlide) -> str:
        cols = []
        for heading, items in ((s.left_heading, s.left_bullets), (s.right_heading, s.right_bullets)):
            h = f"<h3>{_inline(heading)}</h3>" if heading else ""
            cols.append(f'<div class="card">{h}{_bullets_html(items, 520, 360 if heading else 410, 22)}</div>')
        return self._head(s.title) + f'<div class="body cols">{"".join(cols)}</div>'

    def _slide_chart(self, s: ChartSlide) -> str:
        width = 700 if s.bullets else 1160
        svg = render_svg_chart(s, self.theme, width, 450)
        side = ""
        if s.bullets:
            side = f'<div class="card side"><h3>Key takeaways</h3>{_bullets_html(s.bullets, 360, 330, 18)}</div>'
        return self._head(s.title) + f'<div class="body chart-row">{svg}{side}</div>'

    def _slide_table(self, s: TableSlide) -> str:
        n_rows, n_cols = len(s.rows) + 1, len(s.headers)
        weights = table_column_weights(s.headers, s.rows)
        width_in, height_in = 1164 / PX_PER_IN, 450 / PX_PER_IN  # the slide's content box
        start = 18 if n_rows <= 5 and n_cols <= 4 else 16 if n_rows <= 8 else 14 if n_rows <= 10 else 12
        size, _ = fit_table(s.headers, s.rows, [width_in * w / sum(weights) for w in weights], height_in,
                            max_size=start if n_cols < 6 else min(start, 13), cell_pad_x_in=0.3, cell_pad_y_in=0.2)
        numeric = [all(is_numeric(row[c]) or not row[c] for row in s.rows) and any(row[c] for row in s.rows)
                   for c in range(n_cols)]
        cls = [' class="num"' if numeric[c] else "" for c in range(n_cols)]
        cols = "".join(f'<col style="width:{100 * w / sum(weights):.2f}%">' for w in weights)
        head = "".join(f"<th{cls[c]}>{_inline(h)}</th>" for c, h in enumerate(s.headers))
        rows = "".join("<tr>" + "".join(f"<td{cls[c]}>{_inline(x)}</td>" for c, x in enumerate(row)) + "</tr>"
                       for row in s.rows)
        return self._head(s.title) + (
            f'<div class="body"><table style="font-size:{_pt(size)}"><colgroup>{cols}</colgroup>'
            f"<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    def _slide_image(self, s: ImageSlide) -> str:
        path = Path(s.image_path)
        try:
            img = f'<img src="{_data_uri(path)}" alt="{escape(s.caption or s.title)}">'
        except OSError:  # missing or unreadable file, same placeholder text as the PPTX
            label = "missing image" if not path.exists() else "unreadable image"
            img = f'<div class="missing">[{label}: {escape(path.name)}]</div>'
        cap = f"<figcaption>{_inline(s.caption)}</figcaption>" if s.caption else ""
        fig = f"<figure>{img}{cap}</figure>"
        if s.bullets:
            return self._head(s.title) + (
                f'<div class="body img-row"><div class="img-text">{_bullets_html(s.bullets, 470, 450, 22)}</div>{fig}</div>'
            )
        return self._head(s.title) + f'<div class="body img-only">{fig}</div>'

    def _slide_quote(self, s: QuoteSlide) -> str:
        n = len(s.quote)
        size = 40 if n <= 80 else 34 if n <= 140 else 28 if n <= 220 else 24
        attribution = f'<p class="attribution">— {escape(s.attribution)}</p>' if s.attribution else ""
        return (
            f'<div class="quote-wrap"><div class="qmark">“</div><blockquote style="font-size:{_pt(size)}">'
            f"{_inline(s.quote)}</blockquote>{attribution}</div>"
        )


def render_html(deck: Deck, path: str | Path) -> Path:
    return HtmlRenderer(deck).render(path)


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="deck-creator-agent">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<main class="deck">
{slides}
</main>
<div class="hud"><span id="counter"></span> · ←/→ navigate · N notes · F fullscreen · P print</div>
<script>{js}</script>
</body>
</html>
"""

_CSS = """
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; height: 100%; background: #{page_bg}; overflow: hidden; }}
body {{ font-family: '{body_font}', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; }}
.slide {{ position: absolute; left: 50%; top: 50%; width: 1280px; height: 720px; display: none;
  flex-direction: column; padding: 40px 58px 30px; background: #{bg}; color: #{text};
  transform-origin: center center; box-shadow: 0 10px 40px rgba(0,0,0,.35); overflow: hidden; }}
.slide.active {{ display: flex; }}
.slide.cover {{ background: #{cover_bg}; color: #{cover_text}; justify-content: center; }}
header h2 {{ font-family: '{heading_font}', sans-serif; color: #{title}; font-size: 44px; line-height: 1.05;
  margin: 18px 0 14px; font-weight: 700; }}
.bar {{ width: 86px; height: 7px; background: #{accent}; border-radius: 2px; }}
.bar.big {{ width: 125px; height: 9px; margin-bottom: 26px; }}
.cover-body {{ max-width: 1050px; }}
.cover h1 {{ font-family: '{heading_font}', sans-serif; font-size: 62px; line-height: 1.05; margin: 0 0 22px;
  color: #{cover_text}; }}
.cover h1.section {{ font-size: 54px; margin-top: 20px; }}
.cover .subtitle {{ font-size: 29px; color: #{cover_sub}; margin: 0; line-height: 1.3; }}
.cover .meta {{ position: absolute; bottom: 44px; left: 58px; font-size: 19px; color: #{cover_meta}; margin: 0; }}
.body {{ flex: 1; min-height: 0; margin-top: 22px; }}
ul.bullets {{ margin: 0; padding-left: 1.2em; line-height: 1.25; }}
ul.bullets > li {{ margin-top: .55em; }}
ul.bullets > li:first-child {{ margin-top: 0; }}
ul.bullets li::marker {{ color: #{accent}; }}
ul.sub {{ font-size: .82em; padding-left: 1.2em; margin: .25em 0 0; list-style: '–  '; }}
ul.sub li {{ margin-top: .25em; }}
.card {{ background: #{surface}; border-radius: 14px; padding: 26px 30px; }}
.card h3 {{ margin: 0 0 16px; font-family: '{heading_font}', sans-serif; font-size: 29px; color: #{accent}; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 38px; }}
.chart-row {{ display: flex; gap: 32px; align-items: stretch; }}
.chart-row .side {{ flex: 1; }}
.chart-row .side h3 {{ font-size: 21px; }}
svg.chart {{ font-family: inherit; flex: none; }}
table {{ width: 100%; border-collapse: collapse; table-layout: fixed; overflow-wrap: anywhere; }}
th {{ background: #{accent}; color: #{accent_text}; text-align: left; padding: 10px 14px; }}
td {{ padding: 9px 14px; }}
th.num, td.num {{ text-align: right; }}
tbody tr:nth-child(odd) td {{ background: #{bg}; }}
tbody tr:nth-child(even) td {{ background: #{surface}; }}
figure {{ margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
  height: 100%; min-height: 0; }}
figure img {{ max-width: 100%; max-height: 410px; object-fit: contain; border-radius: 6px; }}
figcaption {{ margin-top: 10px; font-size: 17px; color: #{muted}; font-style: italic; }}
.img-row {{ display: grid; grid-template-columns: 42% 1fr; gap: 30px; }}
.img-only figure img {{ max-height: 440px; }}
.missing {{ color: #{muted}; padding: 80px; background: #{surface}; border-radius: 12px; }}
.quote-wrap {{ margin: auto 60px; position: relative; padding-left: 90px; }}
.qmark {{ position: absolute; left: -10px; top: -70px; font-size: 210px; color: #{accent};
  font-family: Georgia, serif; line-height: 1; }}
blockquote {{ margin: 0; font-style: italic; font-family: '{heading_font}', sans-serif; color: #{title};
  line-height: 1.3; }}
.attribution {{ font-size: 26px; color: #{muted}; margin-top: 28px; }}
.sources {{ position: absolute; left: 58px; right: 58px; bottom: 44px; font-size: 13px; font-style: italic;
  color: #{muted}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.cover .sources {{ color: #{cover_meta}; }}
footer {{ position: absolute; left: 58px; right: 58px; bottom: 18px; display: flex; justify-content: space-between;
  font-size: 13px; color: #{muted}; }}
.notes {{ display: none; }}
body.show-notes .slide.active .notes {{ display: block; position: absolute; right: 0; top: 0; bottom: 0;
  width: 42%; overflow: auto; background: rgba(15,23,42,.95); color: #e2e8f0; padding: 24px; font-size: 18px; }}
.notes a {{ color: #7dd3fc; }}
.hud {{ position: fixed; bottom: 8px; right: 12px; font: 12px sans-serif; color: rgba(255,255,255,.55); }}
@media print {{
  @page {{ size: 1280px 720px; margin: 0; }}
  html, body {{ overflow: visible; height: auto; background: none; }}
  .slide {{ display: flex !important; position: relative; left: 0; top: 0; transform: none !important;
    box-shadow: none; break-after: page; page-break-after: always; }}
  .hud, .notes {{ display: none !important; }}
  * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
}}
"""

_JS = """
(function () {
  const slides = Array.from(document.querySelectorAll('.slide'));
  let i = Math.max(0, Math.min(slides.length - 1, (parseInt(location.hash.slice(1), 10) || 1) - 1));
  function fit() {
    const s = Math.min(window.innerWidth / 1280, window.innerHeight / 720) * 0.96;
    slides.forEach(el => el.style.transform = 'translate(-50%, -50%) scale(' + s + ')');
  }
  function show(n) {
    i = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach((el, k) => el.classList.toggle('active', k === i));
    document.getElementById('counter').textContent = (i + 1) + ' / ' + slides.length;
    history.replaceState(null, '', '#' + (i + 1));
    // replaceState fires no hashchange; tell an embedding page (the deck-agent UI) which slide shows.
    window.dispatchEvent(new CustomEvent('slidechange', { detail: i + 1 }));
  }
  document.addEventListener('keydown', e => {
    if (['ArrowRight', 'ArrowDown', 'PageDown', ' ', 'Enter'].includes(e.key)) { show(i + 1); e.preventDefault(); }
    else if (['ArrowLeft', 'ArrowUp', 'PageUp', 'Backspace'].includes(e.key)) { show(i - 1); e.preventDefault(); }
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(slides.length - 1);
    else if (e.key === 'n' || e.key === 'N') document.body.classList.toggle('show-notes');
    else if (e.key === 'f' || e.key === 'F') { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
    else if (e.key === 'p' || e.key === 'P') window.print();
  });
  document.addEventListener('click', e => { if (!e.target.closest('a')) show(e.clientX > window.innerWidth / 3 ? i + 1 : i - 1); });
  window.addEventListener('resize', fit);
  window.addEventListener('hashchange', () => {
    const n = parseInt(location.hash.slice(1), 10);
    if (n && n - 1 !== i) show(n - 1);
  });
  fit(); show(i);
})();
"""
