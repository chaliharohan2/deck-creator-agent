import re

import pytest
from pptx import Presentation

from deck_agent.deck.model import ChartSlide, Deck, ImageSlide
from deck_agent.deck.render import find_soffice, render_deck, safe_filename
from deck_agent.deck.svg_charts import nice_ticks, render_svg_chart
from deck_agent.deck.text import parse_inline, short_source, sources_line
from deck_agent.deck.themes import THEMES, get_theme
from deck_agent.sample import build_sample_deck


@pytest.mark.parametrize("theme", list(THEMES))
def test_sample_deck_renders_in_every_theme(tmp_path, theme):
    deck = build_sample_deck(theme, tmp_path / "assets")
    files = render_deck(deck, tmp_path, f"s_{theme}", ["pptx", "html"])
    prs = Presentation(files["pptx"])
    assert len(prs.slides) == len(deck.slides)
    html = files["html"].read_text()
    assert html.count('<section class="slide') == len(deck.slides)
    assert files["json"].exists()


def test_pptx_content(tmp_path):
    deck = build_sample_deck("light", tmp_path / "assets")
    path = render_deck(deck, tmp_path, "s", ["pptx"])["pptx"]
    prs = Presentation(path)
    titles = [s.shapes.title.text_frame.text if s.shapes.title else "" for s in prs.slides]
    assert titles[0] == "Local AI Agents in 2026"
    assert "Why run the model locally?" in titles
    charts = [sh.chart for s in prs.slides for sh in s.shapes if sh.has_chart]
    assert len(charts) == 4
    assert any(sh.has_table for s in prs.slides for sh in s.shapes)
    assert any(sh.shape_type == 13 for s in prs.slides for sh in s.shapes)  # picture
    notes = prs.slides[0].notes_slide.notes_text_frame.text
    assert "Welcome everyone" in notes
    chart_slide_notes = prs.slides[3].notes_slide.notes_text_frame.text
    assert "huggingface.co/Qwen" in chart_slide_notes
    # bold markdown became a bold run, without literal asterisks
    bullets_text = [sh for sh in prs.slides[1].shapes if sh.has_text_frame and "Privacy" in sh.text_frame.text][0]
    assert "**" not in bullets_text.text_frame.text
    runs = [r for p in bullets_text.text_frame.paragraphs for r in p.runs]
    assert any(r.text == "Privacy" and r.font.bold for r in runs)
    assert prs.core_properties.title == deck.title


def test_value_axis_starts_at_zero(tmp_path):
    deck = Deck(title="T", slides=[ChartSlide(title="c", chart_type="bar", categories=["a", "b"],
                                              series=[{"name": "s", "values": [70, 80]}])])
    prs = Presentation(render_deck(deck, tmp_path, "c", ["pptx"])["pptx"])
    chart = next(sh.chart for sh in prs.slides[0].shapes if sh.has_chart)
    assert chart.value_axis.minimum_scale == 0


def test_missing_image_renders_placeholder(tmp_path):
    deck = Deck(title="T", slides=[ImageSlide(title="i", image_path=str(tmp_path / "nope.png"))])
    files = render_deck(deck, tmp_path, "m", ["pptx", "html"])
    assert "missing image" in files["html"].read_text()


@pytest.mark.skipif(find_soffice() is None, reason="LibreOffice not installed")
def test_pdf_export(tmp_path):
    deck = build_sample_deck("dark", tmp_path / "assets")
    files = render_deck(deck, tmp_path, "pdfdeck", ["pdf"])
    data = files["pdf"].read_bytes()
    assert data.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page[^s]", data)) == len(deck.slides)


def test_render_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        render_deck(build_sample_deck("light", tmp_path), tmp_path, "x", ["docx"])


def test_safe_filename():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("My Deck: Q3 / 2026.pptx") == "2026"  # path part dropped
    assert safe_filename("Heat pumps in Europe!") == "Heat_pumps_in_Europe"
    assert safe_filename("???") == "deck"


def test_text_helpers():
    assert parse_inline("a **b** c") == [("a ", False), ("b", True), (" c", False)]
    assert parse_inline("`code` and *it*") == [("code and it", False)]
    assert short_source("https://www.example.com/a/b") == "example.com/a/b"
    assert sources_line([]) == ""
    assert "+2 more" in sources_line([f"https://x{i}.com" for i in range(5)])


def test_svg_chart_types():
    theme = get_theme("light")
    for kind in ("column", "bar", "line", "area", "stacked_column", "stacked_bar", "pie", "doughnut"):
        c = ChartSlide(title="t", chart_type=kind, categories=["a", "b", "c"], series=[{"name": "s", "values": [1, 2, 3]}])
        svg = render_svg_chart(c, theme, 800, 400)
        assert svg.startswith("<svg") and svg.endswith("</svg>")
    neg = ChartSlide(title="t", categories=["a", "b"], series=[{"name": "s", "values": [-5, 10]}])
    assert "<rect" in render_svg_chart(neg, theme, 800, 400)


def test_nice_ticks():
    assert nice_ticks(0, 79.9) == [0, 20, 40, 60, 80]
    assert nice_ticks(-5, 10)[0] <= -5
