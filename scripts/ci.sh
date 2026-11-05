#!/usr/bin/env bash
# shellcheck disable=SC2119,SC2120  # pytest args reach steps via the dynamic "step_$step" "$@" dispatch
# Every CI step, defined once. GitHub Actions runs exactly these commands, and so can you:
#
#   scripts/ci.sh setup          # fresh virtualenv (.ci-venv) with the latest dependencies
#   scripts/ci.sh lint           # ruff + JavaScript syntax check
#   scripts/ci.sh test           # the test suite (no LibreOffice/browser needed)
#   scripts/ci.sh pdf            # PDF tests; fails if LibreOffice is missing
#   scripts/ci.sh browser        # web UI in Chromium; fails if Playwright/Chromium is missing
#   scripts/ci.sh integration    # pdf + browser together; fails if either tool is missing
#   scripts/ci.sh package        # build sdist + wheel, install the wheel in a clean venv, smoke-test it
#   scripts/ci.sh all            # setup + lint + test + package (+ pdf / browser when available)
#
# Why a script: CI that "passes in one environment and fails on GitHub" usually runs different
# commands, against differently installed dependencies, with tools that exist in only one place.
# Here the commands are the same everywhere, `setup` always starts from a clean virtualenv, and
# jobs that need LibreOffice or a browser fail loudly instead of silently skipping.
#
# Environment: PYTHON (interpreter for the venv, default python3), CI_VENV (default .ci-venv),
# EXTRAS (extra pip extras for setup, e.g. "browser").
set -euo pipefail

cd "$(dirname "$0")/.."
VENV="${CI_VENV:-.ci-venv}"
PY="$VENV/bin/python"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

need_venv() {
  if [[ ! -x "$PY" ]]; then
    echo "No CI virtualenv at $VENV: run 'scripts/ci.sh setup' first." >&2
    exit 1
  fi
}

step_setup() {
  local python="${PYTHON:-python3}" extras="dev${EXTRAS:+,$EXTRAS}"
  say "setup: fresh virtualenv $VENV from $($python --version 2>&1), extras [$extras]"
  rm -rf "$VENV"
  "$python" -m venv "$VENV"
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -e ".[$extras]"
  "$PY" -m pip list --format=freeze | grep -iE '^(deck-creator-agent|python-pptx|pydantic|httpx|starlette|uvicorn|ruff|pytest|playwright)=' || true
}

step_lint() {
  need_venv
  say "lint: ruff"
  "$PY" -m ruff check src tests
  say "lint: JavaScript syntax"
  if ! command -v node >/dev/null; then
    echo "node is required for the JavaScript syntax check (it is preinstalled on GitHub runners)." >&2
    exit 1
  fi
  node --check src/deck_agent/web/static/app.js
}

step_test() {
  need_venv
  say "test: unit and integration tests"
  "$PY" -m pytest -m "not browser" "$@"
}

step_pdf() {
  need_venv
  say "pdf: tests that need LibreOffice"
  DECK_AGENT_CI_REQUIRE=pdf "$PY" -m pytest -m pdf "$@"
}

step_browser() {
  need_venv
  say "browser: the web UI in Chromium"
  DECK_AGENT_CI_REQUIRE=browser "$PY" -m pytest -m browser "$@"
}

step_integration() {
  need_venv
  say "integration: LibreOffice and browser tests (both tools required)"
  DECK_AGENT_CI_REQUIRE=pdf,browser "$PY" -m pytest -m "pdf or browser" "$@"
}

step_package() {
  need_venv
  say "package: build sdist and wheel"
  rm -rf dist
  "$PY" -m build --outdir dist . >/dev/null
  "$PY" -m twine check --strict dist/*
  ls -l dist

  # Install the wheel into a clean venv OUTSIDE the source tree. An editable install hides files
  # missing from the package (the web UI's HTML/CSS/JS, for example); this does not.
  local tmp port pid status=0
  tmp="$(mktemp -d)"
  say "package: install the wheel in a clean virtualenv ($tmp)"
  "$PY" -m venv "$tmp/venv"
  "$tmp/venv/bin/python" -m pip install --quiet dist/*.whl
  pushd "$tmp" >/dev/null
  ./venv/bin/deck-agent --version
  ./venv/bin/deck-agent themes >/dev/null
  ./venv/bin/deck-agent sample -o out -f pptx html
  ./venv/bin/deck-agent render out/sample_light.deck.json -f html -o rerender >/dev/null
  test -s out/sample_light.pptx && test -s rerender/sample_light.html

  say "package: start the installed web UI and fetch its pages"
  port="$(./venv/bin/python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
  ./venv/bin/deck-agent ui --deck out/sample_light.deck.json --port "$port" --no-web >ui.log 2>&1 &
  pid=$!
  for _ in $(seq 1 50); do
    curl -fsS "http://127.0.0.1:$port/api/state" >/dev/null 2>&1 && break
    sleep 0.2
  done
  {
    curl -fsS "http://127.0.0.1:$port/" | grep -q "Deck Agent" &&
    curl -fsS "http://127.0.0.1:$port/static/app.js" | grep -q "function updateDeck" &&
    curl -fsS "http://127.0.0.1:$port/static/style.css" | grep -q ".preview-bar" &&
    curl -fsS "http://127.0.0.1:$port/api/state" | grep -q '"count":12' &&
    curl -fsS "http://127.0.0.1:$port/api/preview.html" | grep -q 'class="slide'
  } || status=$?
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  if [[ $status -ne 0 ]]; then
    echo "The installed web UI did not serve its pages. Server log:" >&2
    cat ui.log >&2
    exit 1
  fi
  echo "installed package OK: CLI, rendering and web UI (static files included)"
  popd >/dev/null
  rm -rf "$tmp"
}

step_all() {
  step_setup
  step_lint
  step_test
  step_package
  if command -v soffice >/dev/null || command -v libreoffice >/dev/null; then step_pdf; else
    say "pdf: skipped here (LibreOffice not installed); CI runs it"; fi
  if "$PY" -c "import playwright" 2>/dev/null; then step_browser; else
    say "browser: skipped here (run 'EXTRAS=browser scripts/ci.sh setup' to include it); CI runs it"; fi
}

step="${1:-all}"
shift || true
case "$step" in
  setup | lint | test | pdf | browser | integration | package | all) "step_$step" "$@" ;;
  *)
    echo "usage: scripts/ci.sh {setup|lint|test|pdf|browser|integration|package|all}" >&2
    exit 2
    ;;
esac
