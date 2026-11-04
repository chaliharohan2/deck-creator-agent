import re

import pytest
from pptx import Presentation

from deck_agent.deck.model import ChartSlide, Deck, ImageSlide, TableSlide
from deck_agent.deck.render import PdfExportError, find_soffice, render_deck, safe_filename
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
    # never a path: separators become word breaks and "."/".." parts are dropped
    for name in ("../../etc/passwd", "/etc/passwd", "..\\..\\x", "..", "./a"):
        out = safe_filename(name)
        assert "/" not in out and "\\" not in out and not out.startswith(".") and out != ".."
    assert safe_filename("../../etc/passwd") == "etc_passwd"
    # a title with a slash keeps all its words (used to become "2026" / "Benefits")
    assert safe_filename("My Deck: Q3 / 2026.pptx") == "My_Deck_Q3_2026"
    assert safe_filename("Heat Pumps: Costs / Benefits") == "Heat_Pumps_Costs_Benefits"
    assert safe_filename("Heat pumps in Europe!") == "Heat_pumps_in_Europe"
    assert safe_filename("notes.deck.json") == "notes"
    assert safe_filename("???") == "deck" and safe_filename("..") == "deck"


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


def test_svg_tick_labels_match_the_ticks():
    c = ChartSlide(title="t", categories=["a", "b"], series=[{"name": "s", "values": [3, 12]}])
    svg = render_svg_chart(c, get_theme("light"), 800, 400)
    labels = re.findall(r'text-anchor="end"[^>]*>([^<]+)<', svg)
    assert labels == ["0.0", "2.5", "5.0", "7.5", "10.0", "12.5"]  # not "2" and "8" at 2.5 and 7.5


def _table_bottom(path):
    from pptx.util import Emu

    shape = next(sh for sh in Presentation(path).slides[0].shapes if sh.has_table)
    return Emu(shape.top + sum(r.height for r in shape.table.rows)).inches


@pytest.mark.parametrize("rows", [
    [[f"Row {i}", "Some longer descriptive text for this cell", f"{i * 1000:,}"] for i in range(12)],
    [[f"Option {i}", "A long explanation that has to wrap onto several lines in its column " * 2, "1.5%"]
     for i in range(8)],
])
def test_tables_stay_above_the_sources_line(tmp_path, rows):
    deck = Deck(title="T", slides=[TableSlide(title="Table", headers=["Name", "Detail", "Value"], rows=rows,
                                              sources=["https://example.org"])])
    assert _table_bottom(render_deck(deck, tmp_path, "t", ["pptx"])["pptx"]) <= 6.65  # sources at 6.68in


def test_numeric_columns_are_right_aligned(tmp_path):
    deck = Deck(title="T", slides=[TableSlide(title="t", headers=["Country", "Units"],
                                              rows=[["France", "528,000"], ["Italy", "423,000"]])])
    files = render_deck(deck, tmp_path, "n", ["pptx", "html"])
    table = next(sh for sh in Presentation(files["pptx"]).slides[0].shapes if sh.has_table).table
    assert table.cell(0, 1).text_frame.paragraphs[0].alignment == 3  # PP_ALIGN.RIGHT, header included
    assert '<th class="num">Units</th>' in files["html"].read_text()


def test_unreadable_image_renders_placeholder(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    deck = Deck(title="T", slides=[ImageSlide(title="i", image_path=str(bad)),
                                   ImageSlide(title="j", image_path=str(tmp_path))])  # a directory
    files = render_deck(deck, tmp_path, "img", ["pptx", "html"])
    texts = [sh.text_frame.text for s in Presentation(files["pptx"]).slides for sh in s.shapes if sh.has_text_frame]
    assert any("unreadable image: broken.png" in t for t in texts)
    html = files["html"].read_text()
    assert "unreadable image: broken.png" in html and "<img" not in html


def test_html_deck_reports_slide_changes(tmp_path):
    html = render_deck(build_sample_deck("light", tmp_path), tmp_path, "h", ["html"])["html"].read_text()
    assert "new CustomEvent('slidechange'" in html  # lets the UI's slide strip follow the preview


def test_failed_pdf_export_never_returns_a_stale_file(tmp_path, monkeypatch):
    fake = tmp_path / "soffice"
    fake.write_text("#!/bin/sh\nexit 0\n")  # "succeeds" without writing anything, as LibreOffice sometimes does
    fake.chmod(0o755)
    monkeypatch.setenv("DECK_AGENT_SOFFICE", str(fake))
    (tmp_path / "old.pdf").write_bytes(b"%PDF- stale")
    deck = Deck(title="T", slides=[ChartSlide(title="c", categories=["a"], series=[[1]])])
    with pytest.raises(PdfExportError):
        render_deck(deck, tmp_path, "old", ["pdf"])
    assert not (tmp_path / "old.pdf").exists()
