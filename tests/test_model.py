import pytest
from pydantic import ValidationError

from deck_agent.deck.model import (
    BulletSlide,
    ChartSlide,
    Deck,
    QuoteSlide,
    TableSlide,
    TitleSlide,
    normalize_bullets,
    parse_number,
    parse_values,
    string_list,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["a", "b"], ["a", "b"]),
        ("a\nb\n\n", ["a", "b"]),
        (["- a", "- b"], ["a", "b"]),  # all-marked list: markers are just list syntax
        (["• a", "• b"], ["a", "b"]),
        (["1. a", "2) b"], ["a", "b"]),
        (["a", "- sub", "b"], ["a", "- sub", "b"]),  # mixed: "- " marks a sub-point
        (["- a", "  - sub", "- b"], ["a", "- sub", "b"]),  # indented marker is a sub-point
        ([{"text": "a", "sub": ["x"]}], ["a", "- x"]),
        (["- orphan"], ["orphan"]),
        ('["a"]\n["b", "c"]', ["a", "b", "c"]),  # JSON fragments sent as one string
        (['"quoted"', "'single'"], ["quoted", "single"]),
        (None, []),
    ],
)
def test_normalize_bullets(raw, expected):
    assert normalize_bullets(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(3, 3.0), ("1,234", 1234.0), ("45%", 45.0), ("$12.5", 12.5), ("(3)", -3.0), (None, 0.0), ("n/a", 0.0)],
)
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize("bad", ["3.2B", "lots", True])
def test_parse_number_rejects_ambiguous(bad):
    with pytest.raises(ValueError):
        parse_number(bad)


def test_bullet_slide_limits():
    with pytest.raises(ValidationError, match="at least one bullet"):
        BulletSlide(title="t", bullets=[])
    with pytest.raises(ValidationError, match="too many"):
        BulletSlide(title="t", bullets=[str(i) for i in range(11)])


def test_chart_validation_and_aliases():
    c = ChartSlide(title="t", chart_type="Horizontal Bar", categories="A, B", series={"S": ["1", "2"]})
    assert c.chart_type == "bar"
    assert c.categories == ["A", "B"]
    assert c.series[0].values == [1.0, 2.0]

    with pytest.raises(ValidationError, match="must match one-to-one"):
        ChartSlide(title="t", categories=["a", "b"], series=[{"name": "s", "values": [1]}])
    with pytest.raises(ValidationError, match="exactly one series"):
        ChartSlide(title="t", chart_type="pie", categories=["a"],
                   series=[{"name": "s", "values": [1]}, {"name": "u", "values": [2]}])
    with pytest.raises(ValidationError, match="unknown chart_type"):
        ChartSlide(title="t", chart_type="radar", categories=["a"], series=[[1]])


def test_chart_series_shapes():
    bare = ChartSlide(title="t", categories=["a", "b"], series=[1, 2])
    assert bare.series[0].name == "Series 1"
    alt = ChartSlide(title="t", categories=["a"], series=[{"label": "L", "data": [5]}])
    assert (alt.series[0].name, alt.series[0].values) == ("L", [5.0])


def test_table_pads_rows_and_rejects_wide_rows():
    t = TableSlide(title="t", headers=["a", "b", "c"], rows=[["1"], ["1", "2", "3"]])
    assert t.rows[0] == ["1", "", ""]
    t2 = TableSlide(title="t", headers="x | y", rows=["1 | 2"])
    assert t2.headers == ["x", "y"] and t2.rows == [["1", "2"]]
    with pytest.raises(ValidationError, match="only 2 headers"):
        TableSlide(title="t", headers=["a", "b"], rows=[["1", "2", "3"]])


def test_quote_strips_quotes_and_limits_length():
    assert QuoteSlide(quote='"Hello"').quote == "Hello"
    with pytest.raises(ValidationError):
        QuoteSlide(quote="x" * 301)


def test_deck_json_roundtrip(tmp_path):
    deck = Deck(title="T", slides=[TitleSlide(title="T"), BulletSlide(title="B", bullets=["x"], sources="https://a.b")])
    path = deck.save_json(tmp_path / "d.deck.json")
    loaded = Deck.load_json(path)
    assert loaded == deck
    assert loaded.slides[1].sources == ["https://a.b"]
    assert "2. [bullets] B" in loaded.outline()


@pytest.mark.parametrize("bad", ["nan", "inf", "-Infinity", float("nan"), float("inf")])
def test_parse_number_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        parse_number(bad)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,234", [1234.0]),                 # thousands separator, one value
        ("1, 2, 3", [1.0, 2.0, 3.0]),        # used to become [123.0]
        ("1,2,3", [1.0, 2.0, 3.0]),
        ("1,234, 5,678", [1234.0, 5678.0]),
        ("1;2;3;", [1.0, 2.0, 3.0]),
        ("12.5\n13", [12.5, 13.0]),
        (["1", "", "3"], [1.0, 0.0, 3.0]),   # an explicit gap stays a gap
        ("−5", [-5.0]),                 # Unicode minus sign
        (7, [7.0]),
    ],
)
def test_parse_values(raw, expected):
    assert parse_values(raw) == expected


def test_series_as_list_of_name_value_objects():
    c = ChartSlide(title="t", categories=["a", "b"], series=[{"Revenue": [1, 2]}, {"Cost": "3, 4"}])
    assert [(s.name, s.values) for s in c.series] == [("Revenue", [1.0, 2.0]), ("Cost", [3.0, 4.0])]


def test_string_list():
    assert string_list("https://a.com, https://b.com") == ["https://a.com", "https://b.com"]
    assert string_list(["EHPA 2025 report", "https://a.com;https://b.com"]) == [
        "EHPA 2025 report", "https://a.com", "https://b.com"]
    assert string_list("Smith, J. (2024)") == ["Smith, J. (2024)"]  # a plain citation keeps its commas
