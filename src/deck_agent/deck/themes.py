"""Colour/font themes shared by all renderers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    description: str
    background: str  # content-slide background
    surface: str  # cards, table banding
    title: str  # slide-title colour
    text: str  # body text
    muted: str  # footers, captions, axis labels
    accent: str  # bars, bullets, highlights
    accent_text: str  # text drawn on top of `accent`
    cover_background: str  # title/section slide background
    cover_text: str  # title/section slide text
    palette: tuple[str, ...]  # chart series colours
    heading_font: str = "Calibri"
    body_font: str = "Calibri"


THEMES: dict[str, Theme] = {
    t.name: t
    for t in [
        Theme(
            name="light",
            description="Clean white slides, navy titles, blue accent (default)",
            background="FFFFFF", surface="F1F4F9", title="14213D", text="273449", muted="6B7688",
            accent="2F5BEA", accent_text="FFFFFF", cover_background="14213D", cover_text="FFFFFF",
            palette=("2F5BEA", "14B8A6", "F59E0B", "EF4444", "8B5CF6", "64748B"),
        ),
        Theme(
            name="dark",
            description="Dark slate background with cyan accent",
            background="0F172A", surface="1E293B", title="F8FAFC", text="E2E8F0", muted="94A3B8",
            accent="38BDF8", accent_text="0F172A", cover_background="020617", cover_text="F8FAFC",
            palette=("38BDF8", "A78BFA", "34D399", "FBBF24", "F87171", "94A3B8"),
        ),
        Theme(
            name="corporate",
            description="Conservative navy and grey, suited to business reviews",
            background="FFFFFF", surface="EEF2F6", title="0B3C5D", text="1F2D3D", muted="6C7A89",
            accent="1D70B8", accent_text="FFFFFF", cover_background="0B3C5D", cover_text="FFFFFF",
            palette=("1D70B8", "0B3C5D", "5BA4CF", "F2A900", "7A8B99", "C0392B"),
            heading_font="Arial", body_font="Arial",
        ),
        Theme(
            name="warm",
            description="Cream background with terracotta accent",
            background="FFFBF5", surface="F6EDE3", title="3D2C29", text="4A3B36", muted="8A7A72",
            accent="E07A5F", accent_text="FFFFFF", cover_background="3D405B", cover_text="FFFBF5",
            palette=("E07A5F", "3D405B", "81B29A", "F2CC8F", "B5838D", "6D6875"),
            heading_font="Georgia", body_font="Calibri",
        ),
        Theme(
            name="forest",
            description="Soft green palette, good for sustainability/science topics",
            background="F7FAF7", surface="E6EFE8", title="1F3B2D", text="2C3E33", muted="6F8577",
            accent="2D6A4F", accent_text="FFFFFF", cover_background="1B4332", cover_text="F7FAF7",
            palette=("2D6A4F", "52B788", "B7950B", "40916C", "95D5B2", "74796D"),
        ),
        Theme(
            name="minimal",
            description="Black and white, high contrast, lots of whitespace",
            background="FFFFFF", surface="F4F4F4", title="111111", text="222222", muted="777777",
            accent="111111", accent_text="FFFFFF", cover_background="FFFFFF", cover_text="111111",
            palette=("111111", "6B6B6B", "A3A3A3", "D4D4D4", "404040", "8C8C8C"),
            heading_font="Helvetica", body_font="Helvetica",
        ),
    ]
}

DEFAULT_THEME = "light"


def get_theme(name: str | None) -> Theme:
    if not name:
        return THEMES[DEFAULT_THEME]
    key = name.strip().lower()
    if key not in THEMES:
        raise ValueError(f"unknown theme {name!r}; available: {', '.join(THEMES)}")
    return THEMES[key]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
