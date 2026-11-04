"""A built-in sample deck that exercises every slide type (no LLM needed)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from deck_agent.deck.model import (
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
)


def _make_placeholder_image(path: Path) -> Path:
    """Draw a simple illustrative image so the sample works offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1200, 800
    img = Image.new("RGB", (w, h), (20, 33, 61))
    draw = ImageDraw.Draw(img)
    for y in range(h):  # vertical gradient
        t = y / h
        draw.line([(0, y), (w, y)], fill=(int(20 + 27 * t), int(33 + 58 * t), int(61 + 173 * t)))
    for i, (x, bar_h) in enumerate([(150, 260), (330, 380), (510, 300), (690, 470), (870, 560)]):
        draw.rounded_rectangle([x, h - 120 - bar_h, x + 120, h - 120], radius=12,
                               fill=(56 + i * 30, 189 - i * 10, 248 - i * 20))
    draw.line([(100, h - 118), (1100, h - 118)], fill=(226, 232, 240), width=4)
    img.save(path)
    return path


def build_sample_deck(theme: str = "light", asset_dir: str | Path = "decks/assets") -> Deck:
    image = _make_placeholder_image(Path(asset_dir) / "sample_chart_art.png")
    return Deck(
        title="Local AI Agents in 2026",
        subtitle="What small open models can do on a laptop",
        author="deck-creator-agent",
        date="November 2026",
        theme=theme,
        slides=[
            TitleSlide(title="Local AI Agents in 2026", subtitle="What small open models can do on a laptop",
                       author="deck-creator-agent", date="November 2026",
                       notes="Welcome everyone. This sample deck shows every slide type the agent can produce."),
            BulletSlide(
                title="Why run the model locally?",
                bullets=[
                    "**Privacy**: prompts and documents never leave your machine",
                    "**Cost**: no per-token fees or subscriptions",
                    "- Hardware you already own is enough for 4B–9B models",
                    "**Control**: pin model versions and tune sampling",
                    "**Offline**: works without internet (except web research)",
                ],
                notes="Emphasise privacy first; it is the most common reason teams go local.",
            ),
            SectionSlide(title="Model landscape", subtitle="Open-weight models that fit in 8–20 GB"),
            ChartSlide(
                title="Agentic tool-use scores (TAU2-Bench, vendor-reported)",
                chart_type="bar",
                categories=["Qwen3.5-4B", "Qwen3.5-9B", "Gemma 4 12B", "Gemma 4 26B-A4B", "Gemma 4 31B"],
                series=[ChartSeries(name="TAU2-Bench (%)", values=[79.9, 79.1, 69.0, 68.2, 76.9])],
                y_axis_title="Score (%)",
                bullets=["Qwen3.5 small models lead at < 10B", "Gemma 4 31B is strongest Gemma", "Always test on your own tasks"],
                sources=["https://huggingface.co/Qwen/Qwen3.5-9B", "https://huggingface.co/google/gemma-4-12b-it"],
            ),
            TwoColumnSlide(
                title="CPU-only vs. small GPU",
                left_heading="CPU only (16–32 GB RAM)",
                left_bullets=["Qwen3.5-9B Q4_K_M (~6.6 GB)", "Qwen3.6-35B-A3B for 32 GB+ RAM", "Expect 3–15 tokens/s"],
                right_heading="GPU with 12–20 GB VRAM",
                right_bullets=["Qwen3.8-27B Q4 (~18 GB)", "Gemma 4 26B-A4B (~17 GB)", "Expect 30–80 tokens/s"],
            ),
            ChartSlide(
                title="Download size by quantization (Qwen3.5-9B)",
                chart_type="column",
                categories=["Q4_K_M", "Q8_0", "BF16"],
                series=[ChartSeries(name="Size (GB)", values=[6.6, 10, 19])],
                y_axis_title="GB",
                sources=["https://ollama.com/library/qwen3.5"],
            ),
            ChartSlide(
                title="Where the time goes in a typical run",
                chart_type="doughnut",
                categories=["Web research", "Writing slides", "Rendering", "Other"],
                series=[ChartSeries(name="Share", values=[45, 40, 5, 10])],
            ),
            ChartSlide(
                title="Illustrative adoption trend",
                chart_type="line",
                categories=["2022", "2023", "2024", "2025", "2026"],
                series=[ChartSeries(name="Local", values=[5, 12, 24, 38, 55]),
                        ChartSeries(name="Cloud", values=[60, 70, 76, 80, 82])],
                y_axis_title="Teams using (%)",
                notes="Illustrative numbers only: not real survey data.",
            ),
            TableSlide(
                title="Recommended models by hardware",
                headers=["Hardware", "Model", "Ollama tag", "Size"],
                rows=[
                    ["CPU, 8–16 GB RAM", "Qwen3.5-4B", "qwen3.5:4b", "3.3 GB"],
                    ["CPU, 16 GB RAM", "Qwen3.5-9B", "qwen3.5:9b", "6.6 GB"],
                    ["CPU, 32 GB+ RAM", "Qwen3.6-35B-A3B", "qwen3.6:35b", "23 GB"],
                    ["GPU 12–16 GB", "Gemma 4 12B", "gemma4:12b", "8 GB"],
                    ["GPU 20–24 GB", "Qwen3.8-27B", "qwen3.8:27b", "18 GB"],
                ],
            ),
            ImageSlide(
                title="Images with supporting points",
                image_path=str(image),
                caption="Generated placeholder artwork",
                bullets=["Images can come from Creative Commons search", "Or from your own local files", "Credits go in the footer"],
                credit="deck-creator-agent sample",
            ),
            QuoteSlide(quote="The best model is the one that runs where your data lives.", attribution="Anonymous engineer"),
            SectionSlide(title="Thank you", subtitle="Questions?"),
        ],
    )
