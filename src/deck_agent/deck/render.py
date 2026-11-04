"""Render a deck to one or more output formats."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from deck_agent.deck.html_renderer import render_html
from deck_agent.deck.model import Deck
from deck_agent.deck.pptx_renderer import render_pptx

FORMATS = ("pptx", "pdf", "html")


class PdfExportError(RuntimeError):
    pass


def find_soffice() -> str | None:
    """Locate the LibreOffice binary (used for PPTX -> PDF)."""
    env = os.environ.get("DECK_AGENT_SOFFICE")
    if env and Path(env).exists():
        return env
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in (
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def export_pdf(pptx_path: Path, timeout: int = 240) -> Path:
    """Convert a .pptx to .pdf with LibreOffice running headless (MPL-2.0)."""
    soffice = find_soffice()
    if not soffice:
        raise PdfExportError(
            "PDF export needs LibreOffice (free, open source). Install it (e.g. `sudo apt install "
            "libreoffice-impress`, `brew install --cask libreoffice`) or open the HTML output in a "
            "browser and use Print -> Save as PDF."
        )
    out_dir = pptx_path.parent
    pdf = pptx_path.with_suffix(".pdf")
    pdf.unlink(missing_ok=True)  # LibreOffice can fail with exit code 0; never return a stale PDF
    with tempfile.TemporaryDirectory(prefix="deck-agent-lo-") as profile:
        cmd = [
            soffice,
            f"-env:UserInstallation={Path(profile).as_uri()}",
            "--headless", "--norestore", "--convert-to", "pdf", "--outdir", str(out_dir), str(pptx_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise PdfExportError(f"LibreOffice timed out after {timeout}s") from exc
    if proc.returncode != 0 or not pdf.exists():
        raise PdfExportError(f"LibreOffice failed (exit {proc.returncode}): {proc.stderr.strip()[:400]}")
    return pdf


def safe_filename(name: str, default: str = "deck") -> str:
    """A file name (without extension) from a deck title or a model-chosen name.

    Path separators become word breaks ("Costs / Benefits" keeps both words) and "." / ".."
    path parts are dropped, so the result can never point outside the output folder.
    """
    name = re.sub(r"(\.deck)?\.(pptx|pdf|html|json)$", "", name.strip(), flags=re.I)
    parts = [p for p in re.split(r"[/\\]+", name) if p.strip() not in ("", ".", "..")]
    name = re.sub(r"[^\w\-. ]+", "", " ".join(parts))
    name = re.sub(r"\s+", "_", name.strip()).strip("._-")
    return name[:80].rstrip("._-") or default


def render_deck(deck: Deck, out_dir: str | Path, basename: str, formats: list[str] | tuple[str, ...] = ("pptx",)) -> dict[str, Path]:
    """Render ``deck`` into ``out_dir`` and return ``{format: path}``.

    A ``<basename>.deck.json`` spec is always written as well so the deck can be
    edited by hand and re-rendered with ``deck-agent render``. PDF failures are
    reported via :class:`PdfExportError` after the other formats are written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = safe_filename(basename)
    fmts = [f.lower().lstrip(".") for f in formats] or ["pptx"]
    unknown = [f for f in fmts if f not in FORMATS]
    if unknown:
        raise ValueError(f"unknown format(s) {unknown}; choose from {FORMATS}")
    results: dict[str, Path] = {"json": deck.save_json(out_dir / f"{base}.deck.json")}
    if "pptx" in fmts or "pdf" in fmts:
        results["pptx"] = render_pptx(deck, out_dir / f"{base}.pptx")
    if "html" in fmts:
        results["html"] = render_html(deck, out_dir / f"{base}.html")
    if "pdf" in fmts:
        results["pdf"] = export_pdf(results["pptx"])
    return results
