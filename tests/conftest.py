from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def png(tmp_path: Path) -> Path:
    path = tmp_path / "pic.png"
    Image.new("RGB", (640, 400), (30, 120, 200)).save(path)
    return path
