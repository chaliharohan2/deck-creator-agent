"""Minimal ``.env`` loader (no dependency): ``KEY=value`` lines into ``os.environ``.

Variables already set in the real environment win over the file, so a CI job or a shell
``export`` can always override what is written in ``.env``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
# Inside double quotes: \" \\ \n \t \r, as in python-dotenv. Other backslashes stay (C:\Users\me).
_DOUBLE_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)
_ESCAPE = re.compile(r"\\([\\\"'ntr])")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r"}


def _value(raw: str) -> str:
    """The value part of a line: quoted (anything after the closing quote, e.g. a comment, is
    ignored) or unquoted (an inline comment needs whitespace before the ``#``)."""
    if raw.startswith('"'):
        m = _DOUBLE_QUOTED.match(raw)
        if m:  # decode escapes in one pass, so "\\n" stays a backslash followed by n
            return _ESCAPE.sub(lambda e: _ESCAPES.get(e.group(1), e.group(1)), m.group(1))
    elif raw.startswith("'"):
        end = raw.find("'", 1)
        if end > 0:
            return raw[1:end]  # single quotes are literal
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines. Supports comments, ``export``, and single/double quotes."""
    values: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():  # -sig: Notepad's BOM
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue  # not a KEY=value line; ignore rather than fail
        values[m.group(1)] = _value(m.group(2).strip())
    return values


def find_env_file(explicit: str | Path | None = None) -> Path | None:
    """``--env-file`` if given, else ``$DECK_AGENT_ENV_FILE``, else ``./.env`` if it exists."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("DECK_AGENT_ENV_FILE")
    if env:
        return Path(env).expanduser()
    local = Path.cwd() / ".env"
    return local if local.is_file() else None


def load_env_file(path: str | Path | None) -> list[str]:
    """Load ``path`` into ``os.environ`` without overriding existing variables.

    Returns the names that were set from the file (values are never returned or logged).
    """
    if path is None:
        return []
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"env file not found: {path}")
    loaded = []
    for key, value in parse_env_file(path).items():
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
