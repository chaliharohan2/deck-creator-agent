from deck_agent.factcheck import NumberCorpus, find_figures, number_figure

NOTES = """Heat pump sales rose 13% in 2025, reaching 2.9 million units in 21 countries.
Total stock: 29.3 million. Units sold: France 528,000, Italy 423,000. Norway 650 per 1,000 households."""


def corpus():
    c = NumberCorpus()
    c.add_text(NOTES)
    return c


def test_supported_values_including_scaling_and_rounding():
    c = corpus()
    for v in (528, 528000, 0.528, 423, 2.9, 2_900_000, 29.3, 29, 13, 0.13, 650, 0):
        assert c.supports(number_figure(v)), v


def test_invented_values_are_not_supported():
    c = corpus()
    for v in (1450, 950, 35, 1.8, 531):
        assert not c.supports(number_figure(v)), v


def test_find_figures_skips_years_and_small_integers():
    texts = [f.text for f in find_figures("In 2025 the top 3 markets sold 528,000 units (13%); Q4 rose 4.5%")]
    assert texts == ["528,000", "13%", "4.5%"]


def test_precision_matters():
    c = NumberCorpus()
    c.add_text("revenue was 1,449.7 million")
    assert c.supports(number_figure(1450))   # 3 significant digits: 1449.7 rounds to 1450
    assert c.supports(number_figure(1.45))   # billions
    assert not c.supports(number_figure(1449.2))
