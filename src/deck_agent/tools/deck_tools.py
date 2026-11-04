"""Deck-building tools exposed to the agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from deck_agent.deck.model import (
    SLIDE_TYPES,
    BulletSlide,
    ChartSeries,
    ChartSlide,
    Deck,
    ImageSlide,
    QuoteSlide,
    SectionSlide,
    TableSlide,
    TitleSlide,
    TwoColumnSlide,
    normalize_chart_type,
    normalize_series,
    parse_number,
)
from deck_agent.deck.render import PdfExportError, render_deck, safe_filename
from deck_agent.deck.themes import THEMES
from deck_agent.factcheck import NumberCorpus, find_figures, number_figure
from deck_agent.tools.base import NoArgs, ToolArgs, ToolError, ToolRegistry

ThemeName = Literal["light", "dark", "corporate", "warm", "forest", "minimal"]
assert set(ThemeName.__args__) == set(THEMES), "keep ThemeName in sync with THEMES"

LONG_BULLET_WORDS = 22


@dataclass
class DeckSession:
    """Mutable state shared by the deck tools during an agent session."""

    output_dir: Path = Path("decks")
    formats: list[str] = field(default_factory=lambda: ["pptx"])
    default_theme: str | None = None
    allow_private_urls: bool = False
    deck: Deck = field(default_factory=Deck)
    dirty: bool = False
    saved: dict[str, Path] = field(default_factory=dict)
    basename: str | None = None
    known_urls: set[str] = field(default_factory=set)  # URLs the agent actually saw (search, pages, user)
    facts: NumberCorpus = field(default_factory=NumberCorpus)  # numbers the agent actually saw
    verify_numbers: bool = True

    def __post_init__(self) -> None:
        if self.default_theme:
            self.deck.theme = self.default_theme

    @property
    def assets_dir(self) -> Path:
        return self.output_dir / "assets"

    def reset(self) -> None:
        self.deck = Deck(theme=self.default_theme or "light")
        self.dirty = False
        self.saved = {}
        self.basename = None

    def note_source(self, text: str) -> None:
        """Record material the agent genuinely saw: its URLs and its numbers."""
        self.note_urls(text)
        self.facts.add_text(text)

    def note_urls(self, text: str) -> None:
        """Remember URLs that genuinely appeared (search results, pages read, the user's message)."""
        for url in _URL.findall(text or ""):
            self.known_urls.add(_norm_url(url.rstrip(".,;:!?")))

    def is_known_url(self, url: str) -> bool:
        cited = _norm_url(url)
        for known in self.known_urls:
            if cited == known or known.startswith(cited + "/"):  # exact page, or a parent page of it
                return True
            if "/" not in known and cited.split("/", 1)[0] == known:  # user vouched for a whole site
                return True
        return False

    def insert(self, slide: Any, position: int | None) -> int:
        slides = self.deck.slides
        for number, existing in enumerate(slides, 1):
            what = _duplicate(existing, slide)
            if what:
                raise ToolError(f"slide {number} already has this {what}. To reorder slides use move_slide; to change "
                                "a slide use update_slide. Do not add a copy.")
        if position is None or position > len(slides):
            slides.append(slide)
            index = len(slides)
        else:
            index = max(1, position)
            slides.insert(index - 1, slide)
        self.dirty = True
        return index

    def get(self, number: int) -> Any:
        if not 1 <= number <= len(self.deck.slides):
            raise ToolError(f"slide {number} does not exist; the deck has {len(self.deck.slides)} slides")
        return self.deck.slides[number - 1]

    def save(self, filename: str | None = None, formats: list[str] | None = None) -> tuple[dict[str, Path], str]:
        if not self.deck.slides:
            raise ToolError("the deck has no slides yet; add slides before saving")
        if filename:
            self.basename = safe_filename(filename)
        if not self.basename:
            self.basename = safe_filename(self.deck.title, default="deck")
        # The user's configured formats are a floor; the model may add formats but not drop them.
        fmts = list(dict.fromkeys([*self.formats, *(formats or [])]))
        note = ""
        try:
            paths = render_deck(self.deck, self.output_dir, self.basename, fmts)
        except PdfExportError as exc:
            paths = render_deck(self.deck, self.output_dir, self.basename, [f for f in fmts if f != "pdf"] or ["pptx"])
            note = f" (PDF skipped: {exc})"
        self.saved = paths
        self.dirty = False
        return paths, note


# --------------------------------------------------------------------------- argument models
_POSITION = Field(None, description="Optional 1-based slide number to insert at. Omit to append at the end.")
_NOTES = Field("", description="Speaker notes: the detail the presenter will say (2-5 sentences).")
_SOURCES = Field(default_factory=list, description="URLs or citations for facts on this slide.")


class DeckInfoArgs(ToolArgs):
    title: str = Field(description="Presentation title")
    subtitle: str = Field("", description="One-line subtitle")
    author: str = Field("", description="Presenter name, only if the user gave one")
    theme: ThemeName | None = Field(None, description="Visual theme")


class BulletArgs(ToolArgs):
    title: str = Field(description="Slide title stating the key message")
    bullets: list[str] = Field(description="3-6 short bullet points. Prefix an item with '- ' to make it a sub-point.")
    notes: str = _NOTES
    sources: list[str] = _SOURCES
    position: int | None = _POSITION


class TwoColumnArgs(ToolArgs):
    title: str = Field(description="Slide title")
    left_heading: str = Field(description="Heading of the left column")
    left_bullets: list[str] = Field(description="2-5 bullets for the left column")
    right_heading: str = Field(description="Heading of the right column")
    right_bullets: list[str] = Field(description="2-5 bullets for the right column")
    notes: str = _NOTES
    sources: list[str] = _SOURCES
    position: int | None = _POSITION


class SeriesArg(ToolArgs):
    name: str = Field(description="Series name shown in the legend")
    values: list[float] = Field(description="Plain numbers, one per category")

    @field_validator("values", mode="before")
    @classmethod
    def _values(cls, v: Any) -> list[float]:
        if isinstance(v, (int, float, str)):
            v = [v]
        return [parse_number(x) for x in (v or [])]


class ChartArgs(ToolArgs):
    title: str = Field(description="Slide title stating what the data shows")
    chart_type: Literal["column", "bar", "line", "pie", "doughnut", "area", "stacked_column", "stacked_bar"] = Field(
        "column", description="column/bar compare items, line/area show trends over time, pie/doughnut show shares of a whole"
    )
    categories: list[str] = Field(description="Category labels (x-axis), e.g. years or product names")
    series: list[SeriesArg] = Field(description="One or more data series, each with one value per category")
    y_axis_title: str = Field("", description="Value axis label including units, e.g. 'Revenue (USD bn)'")
    x_axis_title: str = Field("", description="Category axis label")
    takeaways: list[str] = Field(default_factory=list, description="Optional 1-4 short insights shown beside the chart")
    notes: str = _NOTES
    sources: list[str] = _SOURCES
    position: int | None = _POSITION

    @field_validator("series", mode="before")
    @classmethod
    def _series(cls, v: Any) -> Any:
        return normalize_series(v)

    @field_validator("chart_type", mode="before")
    @classmethod
    def _chart_type(cls, v: Any) -> Any:
        return normalize_chart_type(v)


class TableArgs(ToolArgs):
    title: str = Field(description="Slide title")
    headers: list[str] = Field(description="Column headers (2-6 columns)")
    rows: list[list[str]] = Field(description="Table rows; each row is a list of cell texts, same length as headers")
    notes: str = _NOTES
    sources: list[str] = _SOURCES
    position: int | None = _POSITION


class ImageArgs(ToolArgs):
    title: str = Field(description="Slide title")
    image: str = Field(description="Image URL (from search_images) or a local file path")
    caption: str = Field("", description="Short caption under the image")
    credit: str = Field("", description="Image credit / source page")
    bullets: list[str] = Field(default_factory=list, description="Optional 0-5 bullets shown beside the image")
    notes: str = _NOTES
    sources: list[str] = _SOURCES
    position: int | None = _POSITION


class SectionArgs(ToolArgs):
    title: str = Field(description="Section name, e.g. 'Market Analysis'")
    subtitle: str = Field("", description="Optional one-line description")
    notes: str = _NOTES
    position: int | None = _POSITION


class QuoteArgs(ToolArgs):
    quote: str = Field(description="The quote or key statement (max ~200 characters)")
    attribution: str = Field("", description="Who said it")
    notes: str = _NOTES
    sources: list[str] = _SOURCES
    position: int | None = _POSITION


class UpdateArgs(ToolArgs):
    slide_number: int = Field(description="1-based number of the slide to change")
    title: str | None = Field(None, description="New title")
    subtitle: str | None = Field(None, description="New subtitle (title/section slides)")
    bullets: list[str] | None = Field(None, description="Replacement bullet list")
    notes: str | None = Field(None, description="Replacement speaker notes")
    sources: list[str] | None = Field(None, description="Replacement sources")


class SlideNumberArgs(ToolArgs):
    slide_number: int = Field(description="1-based slide number")


class MoveArgs(ToolArgs):
    slide_number: int = Field(description="1-based number of the slide to move")
    new_position: int = Field(description="1-based position to move it to")


class SaveArgs(ToolArgs):
    filename: str = Field("", description="File name without extension (defaults to the deck title)")
    formats: list[Literal["pptx", "pdf", "html"]] = Field(
        default_factory=list, description="Output formats; omit to use the configured defaults"
    )


# --------------------------------------------------------------------------- helpers
def _bullet_warnings(bullets: list[str]) -> str:
    notes = []
    long_items = [i + 1 for i, b in enumerate(bullets) if len(b.split()) > LONG_BULLET_WORDS]
    if long_items:
        notes.append(f"bullet(s) {long_items} are long; shorter bullets read better (detail belongs in notes)")
    top_level = [b for b in bullets if not b.startswith("- ")]
    if len(top_level) > 6:
        notes.append("more than 6 bullets; consider splitting the slide")
    if len(top_level) == 1:
        notes.append("the slide has a single bullet; add 2-4 supporting points with update_slide, "
                     "or fold it into another slide")
    return (" Tip: " + "; ".join(notes) + ".") if notes else ""


_URL = re.compile(r"https?://[^\s<>\"'()\[\]{}]+")


def _norm_url(url: str) -> str:
    url = re.sub(r"^https?://", "", url.strip().lower()).split("#", 1)[0]
    url = url.removeprefix("www.")
    return url.rstrip("/")


_FIGURE = re.compile(
    r"\d[\d,.]*\s?(%|percent|per ?cent|million|billion|trillion|bn\b|mn\b|[kmb]\b)|[$€£¥₹]\s?\d|\d\s?(gw|mw|kwh|twh|t\b|tonnes?)",
    re.IGNORECASE,
)


def _unsourced_figures_warning(texts: list[str], sources: list[str]) -> str:
    """Small models invent statistics. Flag specific figures that come with no source."""
    if sources or not any(_FIGURE.search(t or "") for t in texts):
        return ""
    return (" Warning: this slide states specific figures but has no `sources`. If the numbers did not come "
            "from the user or from a page you read, rewrite the slide without precise figures (update_slide); "
            "otherwise add the source URL.")


def _body(slide: Any) -> tuple:
    if isinstance(slide, TwoColumnSlide):
        return (*slide.left_bullets, *slide.right_bullets)
    if isinstance(slide, ChartSlide):
        return (*slide.categories, *(v for x in slide.series for v in x.values))
    if isinstance(slide, TableSlide):
        return tuple(c for row in slide.rows for c in row)
    if isinstance(slide, QuoteSlide):
        return (slide.quote,)
    return tuple(getattr(slide, "bullets", ()) or ())


def _duplicate(existing: Any, new: Any) -> str:
    if existing.kind != new.kind or new.kind == "title":
        return ""
    norm = lambda t: " ".join(str(t).lower().split())  # noqa: E731
    if norm(existing.heading) == norm(new.heading):
        return "title"
    if _body(new) and _body(new) == _body(existing):
        return "content"
    return ""


def _check_numbers(session: DeckSession, texts: list[str]) -> str:
    if not session.verify_numbers:
        return ""
    bad = session.facts.unsupported([f for t in texts for f in find_figures(t or "")])
    if not bad:
        return ""
    return (f" Warning: these figures do not appear in the user's input or in any search result or page you read: "
            f"{', '.join(dict.fromkeys(f.text for f in bad[:6]))}. Remove or correct them with update_slide unless "
            "the user gave them to you.")


def _vet_sources(session: DeckSession, slide: Any, texts: list[str]) -> str:
    """Drop cited URLs the agent never saw, then flag figures that are left without a source.

    Small models invent plausible-looking citations; a fabricated source is worse than none.
    """
    invented = [s for s in slide.sources if _URL.match(s) and not session.is_known_url(s)]
    note = ""
    if invented:
        slide.sources = [s for s in slide.sources if s not in invented]
        note = (f" Removed source(s) {', '.join(invented[:3])}: they did not appear in your search results, the pages "
                "you read or the user's message. Only cite pages you actually used, and remove figures you cannot "
                "source (update_slide).")
    return note + _check_numbers(session, texts) + _unsourced_figures_warning(texts, slide.sources)


def _added(session: DeckSession, index: int, slide: Any, extra: str = "") -> str:
    return (f"Added slide {index} ({slide.kind}): \"{slide.heading}\". "
            f"The deck now has {len(session.deck.slides)} slides.{extra}")


def _ensure_title_slide(session: DeckSession) -> None:
    deck = session.deck
    if deck.slides and isinstance(deck.slides[0], TitleSlide):
        return
    if deck.title and deck.title != Deck.model_fields["title"].default:
        deck.slides.insert(0, TitleSlide(title=deck.title, subtitle=deck.subtitle, author=deck.author, date=deck.date))


# --------------------------------------------------------------------------- registration
def register_deck_tools(registry: ToolRegistry, session: DeckSession) -> None:
    @registry.add(
        "set_deck_info",
        "Set the presentation title, subtitle, author and theme, and create/update the title slide. Call this first.",
        DeckInfoArgs,
    )
    def set_deck_info(a: DeckInfoArgs) -> str:
        deck = session.deck
        deck.title, deck.subtitle, deck.author = a.title.strip(), a.subtitle.strip(), a.author.strip()
        if a.theme:
            deck.theme = a.theme
        deck.date = deck.date or date.today().strftime("%B %Y")
        title_slide = TitleSlide(title=deck.title, subtitle=deck.subtitle, author=deck.author, date=deck.date)
        if deck.slides and isinstance(deck.slides[0], TitleSlide):
            title_slide.notes = deck.slides[0].notes
            deck.slides[0] = title_slide
            action = "Updated"
        else:
            deck.slides.insert(0, title_slide)
            action = "Created"
        session.dirty = True
        return f"{action} title slide. Title: \"{deck.title}\", theme: {deck.theme}. Now add the content slides."

    @registry.add("add_bullet_slide", "Add a slide with a title and bullet points.", BulletArgs)
    def add_bullet_slide(a: BulletArgs) -> str:
        _ensure_title_slide(session)
        slide = BulletSlide(title=a.title, bullets=a.bullets, notes=a.notes, sources=a.sources)
        extra = _bullet_warnings(slide.bullets) + _vet_sources(session, slide, [slide.title, *slide.bullets])
        return _added(session, session.insert(slide, a.position), slide, extra)

    @registry.add("add_two_column_slide", "Add a slide comparing two things side by side (pros/cons, before/after, A vs B).",
                  TwoColumnArgs)
    def add_two_column_slide(a: TwoColumnArgs) -> str:
        _ensure_title_slide(session)
        slide = TwoColumnSlide(title=a.title, left_heading=a.left_heading, left_bullets=a.left_bullets,
                               right_heading=a.right_heading, right_bullets=a.right_bullets,
                               notes=a.notes, sources=a.sources)
        extra = _vet_sources(session, slide, [slide.title, *slide.left_bullets, *slide.right_bullets])
        return _added(session, session.insert(slide, a.position), slide, extra)

    @registry.add("add_chart_slide",
                  "Add a native, editable chart. Use only real numbers from the user or from pages you read.", ChartArgs)
    def add_chart_slide(a: ChartArgs) -> str:
        _ensure_title_slide(session)
        slide = ChartSlide(
            title=a.title, chart_type=a.chart_type, categories=a.categories,
            series=[ChartSeries(name=s.name, values=s.values) for s in a.series],
            y_axis_title=a.y_axis_title, x_axis_title=a.x_axis_title, bullets=a.takeaways,
            notes=a.notes, sources=a.sources,
        )
        if session.verify_numbers:
            bad = [(cat, v) for x in slide.series for cat, v in zip(slide.categories, x.values, strict=True)
                   if not session.facts.supports(number_figure(v))]
            if bad and not len(session.facts):
                raise ToolError("chart not added: there is no source material with numbers in this conversation, so "
                                "the chart cannot be backed by data. Use a bullet slide without precise figures instead.")
            if bad:
                listed = ", ".join(f"{cat}={v:g}" for cat, v in bad[:8])
                raise ToolError(f"chart not added: these values do not appear in the user's input or in any search "
                                f"result or page you read: {listed}. Charts may only use numbers from your sources. "
                                "Add the chart again with only the categories whose values you found.")
        extra = _vet_sources(session, slide, [*slide.bullets])
        if not slide.sources:
            extra += (" Warning: this chart has no `sources`. Charts must use real numbers from the user or from "
                      "pages you read; add the source URL, or delete the slide if the numbers were estimated.")
        return _added(session, session.insert(slide, a.position), slide, extra)

    @registry.add("add_table_slide", "Add a slide with a table (structured comparisons, specs, timelines).", TableArgs)
    def add_table_slide(a: TableArgs) -> str:
        _ensure_title_slide(session)
        slide = TableSlide(title=a.title, headers=a.headers, rows=a.rows, notes=a.notes, sources=a.sources)
        cells = [slide.title, *slide.headers, *(c for row in slide.rows for c in row)]
        return _added(session, session.insert(slide, a.position), slide, _vet_sources(session, slide, cells))

    @registry.add("add_image_slide",
                  "Add a slide with an image (URL from search_images, or a local file) and optional bullets.", ImageArgs)
    def add_image_slide(a: ImageArgs) -> str:
        _ensure_title_slide(session)
        from deck_agent.tools.web import save_image

        path = save_image(a.image, session.assets_dir, allow_private=session.allow_private_urls)
        slide = ImageSlide(title=a.title, image_path=str(path), caption=a.caption, credit=a.credit,
                           bullets=a.bullets, notes=a.notes, sources=a.sources)
        extra = _vet_sources(session, slide, [slide.title, *slide.bullets])
        return _added(session, session.insert(slide, a.position), slide, extra)

    @registry.add("add_section_slide", "Add a section divider slide to separate parts of the talk.", SectionArgs)
    def add_section_slide(a: SectionArgs) -> str:
        _ensure_title_slide(session)
        slide = SectionSlide(title=a.title, subtitle=a.subtitle, notes=a.notes)
        return _added(session, session.insert(slide, a.position), slide)

    @registry.add("add_quote_slide", "Add a slide with one large quote or key statement.", QuoteArgs)
    def add_quote_slide(a: QuoteArgs) -> str:
        _ensure_title_slide(session)
        slide = QuoteSlide(quote=a.quote, attribution=a.attribution, notes=a.notes, sources=a.sources)
        extra = _vet_sources(session, slide, [slide.quote])
        if slide.attribution and not slide.sources:
            extra = (" Warning: a quotation attributed to someone needs the URL of the page where you read it in "
                     "`sources`. If you wrote this statement yourself, remove the attribution (update_slide cannot "
                     "do this: delete the slide and add it again without `attribution`).")
        return _added(session, session.insert(slide, a.position), slide, extra)

    @registry.add("list_slides", "Show the current outline of the deck with slide numbers.", NoArgs)
    def list_slides(_: NoArgs) -> str:
        return session.deck.outline()

    @registry.add("update_slide", "Change fields (title, subtitle, bullets, notes, sources) of an existing slide.",
                  UpdateArgs)
    def update_slide(a: UpdateArgs) -> str:
        slide = session.get(a.slide_number)
        changes = {k: v for k, v in a.model_dump(exclude={"slide_number"}).items() if v is not None}
        if not changes:
            raise ToolError("nothing to change; pass at least one of title, subtitle, bullets, notes, sources")
        data = slide.model_dump()
        fields = set(type(slide).model_fields)
        if "bullets" in changes and "bullets" not in fields:
            raise ToolError(f"slide {a.slide_number} is a {slide.kind} slide and has no bullets")
        unknown = [k for k in changes if k not in fields]
        if unknown:
            raise ToolError(f"slide {a.slide_number} ({slide.kind}) has no field(s) {unknown}")
        data.update(changes)
        new_slide = SLIDE_TYPES[slide.kind].model_validate(data)
        texts = [getattr(new_slide, "title", ""), *getattr(new_slide, "bullets", [])]
        extra = _vet_sources(session, new_slide, texts)
        session.deck.slides[a.slide_number - 1] = new_slide
        if isinstance(new_slide, TitleSlide) and a.slide_number == 1:
            session.deck.title, session.deck.subtitle = new_slide.title, new_slide.subtitle
        session.dirty = True
        return f"Updated slide {a.slide_number} ({', '.join(changes)}).{extra}"

    @registry.add("delete_slide", "Delete a slide by its number.", SlideNumberArgs)
    def delete_slide(a: SlideNumberArgs) -> str:
        slide = session.get(a.slide_number)
        session.deck.slides.pop(a.slide_number - 1)
        session.dirty = True
        return f"Deleted slide {a.slide_number} (\"{slide.heading}\"). The deck now has {len(session.deck.slides)} slides."

    @registry.add("move_slide", "Move a slide to a new position.", MoveArgs)
    def move_slide(a: MoveArgs) -> str:
        slide = session.get(a.slide_number)
        slides = session.deck.slides
        slides.pop(a.slide_number - 1)
        target = min(max(1, a.new_position), len(slides) + 1)
        slides.insert(target - 1, slide)
        session.dirty = True
        return f"Moved \"{slide.heading}\" to position {target}."

    @registry.add("save_deck", "Render and save the deck to files. Call this when the deck is complete.", SaveArgs)
    def save_deck(a: SaveArgs) -> str:
        paths, note = session.save(a.filename or None, list(a.formats) or None)
        files = ", ".join(str(p) for k, p in paths.items() if k != "json")
        return f"Saved {len(session.deck.slides)} slides to: {files}{note}"

