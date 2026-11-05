"""Browser smoke test: the real web UI in Chromium, against a scripted model.

Unit tests cannot see JavaScript bugs (a broken event handler, a state variable clobbered by
another, a preview that never redraws). This drives the page the way a person does and fails on
any JavaScript error. Needs `pip install -e '.[browser]'` and `python -m playwright install chromium`.
"""

from __future__ import annotations

import io

import pytest
from mock_vllm import BackgroundServer
from pptx import Presentation

from deck_agent.config import Settings
from deck_agent.llm import LLMResponse, ToolCall
from deck_agent.web.server import create_app

pytestmark = pytest.mark.browser

SCRIPT = [
    [("set_deck_info", {"title": "Green Tea", "theme": "forest"})],
    [("add_bullet_slide", {"title": "Why green tea", "bullets": ["Rich in antioxidants", "Less caffeine"]})],
    [("add_chart_slide", {"title": "Cups sold", "chart_type": "column", "categories": ["2024", "2025"],
                          "series": [{"name": "cups", "values": [120, 180]}]})],
    "Your 3-slide deck is ready.",
]


class ScriptedLLM:
    base_url = "http://localhost:11434"

    def __init__(self):
        self.script = [list(s) if isinstance(s, list) else s for s in SCRIPT]

    def chat(self, messages, tools, on_progress=None):
        step = self.script.pop(0) if self.script else "Done."
        if isinstance(step, str):
            return LLMResponse(content=step)
        return LLMResponse(tool_calls=[ToolCall(name=n, arguments=a) for n, a in step])

    def list_models(self):
        return ["fake:9b"]

    def close(self):
        pass


@pytest.fixture
def ui(tmp_path):
    settings = Settings(output_dir=tmp_path / "decks", web=False, model="fake:9b")
    with BackgroundServer(create_app(settings, llm_factory=lambda s: ScriptedLLM())) as srv:
        yield srv


@pytest.fixture
def page(ui):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
        pg = context.new_page()
        pg.js_errors = []
        pg.on("pageerror", lambda e: pg.js_errors.append(str(e)))
        pg.on("console", lambda m: pg.js_errors.append(m.text) if m.type == "error" else None)
        pg.goto(ui.url)
        pg.wait_for_selector("#model option", state="attached")
        yield pg
        browser.close()


def build_deck(page):
    page.fill("#input", "A short deck about green tea. Cups sold: 120 in 2024 and 180 in 2025.")
    page.click("#send")
    page.wait_for_function("document.querySelectorAll('#outline button').length === 3", timeout=20_000)
    page.wait_for_function("document.querySelector('#status').hidden === true", timeout=20_000)


def preview_slides(page) -> int:
    return page.evaluate("document.querySelector('#frame').contentDocument.querySelectorAll('section.slide').length")


def test_build_preview_navigate_and_download(page):
    # Initial state: CPU mode active, GPU server unavailable (not configured).
    assert page.get_attribute('#modeSwitch .tab[data-mode="cpu"]', "aria-checked") == "true"
    assert page.is_disabled('#modeSwitch .tab[data-mode="server"]')
    assert "Slides will appear here" in page.frame_locator("#frame").locator("body").inner_text()

    build_deck(page)
    assert page.locator(".msg.assistant").inner_text().startswith("Your 3-slide deck is ready.")
    assert page.inner_text("#deckTitle") == "Green Tea" and page.inner_text("#deckCount") == "3 slides"
    assert page.locator(".activity .step.ok").count() == 3

    # The live preview shows the deck (not a placeholder or an overlay).
    page.wait_for_function("document.querySelector('#frame').contentDocument.querySelectorAll('section.slide').length === 3")
    assert page.is_hidden("#overlay") and page.is_visible("#frame")

    # Slide strip navigation moves the preview.
    page.click('#outline button[data-n="2"]')
    page.wait_for_function("document.querySelector('#frame').contentWindow.location.hash === '#2'")

    # Theme switch redraws without the model.
    page.select_option("#theme", "dark")
    page.wait_for_function("document.querySelector('#theme').value === 'dark'")
    page.wait_for_function("document.querySelector('#frame').contentDocument.querySelectorAll('section.slide').length === 3")

    # Download the PowerPoint file.
    with page.expect_download() as info:
        page.click("#dl-pptx")
    data = open(info.value.path(), "rb").read()
    assert info.value.suggested_filename == "Green_Tea.pptx"
    assert len(Presentation(io.BytesIO(data)).slides) == 3

    # A reload restores the conversation and the deck.
    page.reload()
    page.wait_for_function("document.querySelectorAll('#outline button').length === 3")
    assert page.locator(".msg.user").count() == 1 and page.locator(".msg.assistant").count() == 1
    assert page.js_errors == []


@pytest.mark.pdf
def test_powerpoint_render_tab(page):
    build_deck(page)
    page.click('.preview-bar .tab[data-mode="pdf"]')
    page.wait_for_function("document.querySelector('#overlay').hidden && document.querySelector('#pageImg').naturalWidth > 0",
                           timeout=120_000)
    assert page.inner_text("#pageNum") == "1 / 3"
    page.click("#pageNext")
    page.wait_for_function("document.querySelector('#pageNum').textContent === '2 / 3'")
    page.click('.preview-bar .tab[data-mode="html"]')
    page.wait_for_function("!document.querySelector('#frame').hidden")
    page.wait_for_function("document.querySelector('#frame').contentDocument.querySelectorAll('section.slide').length === 3")
    assert preview_slides(page) == 3 and page.js_errors == []
