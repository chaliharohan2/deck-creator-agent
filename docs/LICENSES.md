# Open-source audit

Requirement: every library must be open source, and nothing paid or "free tier" may be used. This page records how each part of the agent meets that, and **what open source cannot fully provide**.

## Python dependencies (runtime, including transitive)

All are OSI-approved licences; none requires an account, API key or payment.

| Package | Licence | Used for |
|---|---|---|
| python-pptx | MIT | Writing .pptx files (native charts, tables, notes) |
| pydantic / pydantic-core | MIT | Deck schema and tool-argument validation |
| httpx, httpcore, h11, anyio | BSD-3 / MIT | Talking to the local model server and fetching pages |
| ddgs, primp | MIT | Key-less metasearch (DuckDuckGo, Bing, Brave, Mojeek, Wikipedia, ...) |
| (no library) | – | Wikipedia search fallback via the official MediaWiki API, called with `httpx` |
| trafilatura, courlan, htmldate | Apache-2.0 | Extracting the main text of web pages |
| jusText, lxml, lxml_html_clean | BSD | HTML parsing |
| pypdf | BSD-3 | Reading PDF search results |
| Pillow | MIT-CMU | Image validation and conversion |
| XlsxWriter | BSD-2 | Chart data embedded in .pptx (used by python-pptx) |
| rich, markdown-it-py, mdurl, Pygments | MIT / BSD-2 | Terminal UI |
| starlette, uvicorn | BSD-3 | Local web UI server (front end is hand-written HTML/CSS/JS, no external assets) |
| pypdfium2 (PDFium) | BSD-3 / Apache-2.0 | Turns the LibreOffice PDF into slide images for the UI's "PowerPoint render" tab |
| click, dateparser, python-dateutil, regex, pytz, tzlocal, tzdata, babel, charset-normalizer, idna, urllib3, certifi, tld, typing-extensions, annotated-types, typing-inspection | MIT / BSD / Apache-2.0 / PSF / MPL-2.0 | Transitive utilities |

Notes on weak copyleft: `certifi` is MPL-2.0 and `tld` is tri-licensed MPL-1.1 / GPL-2.0 / LGPL-2.1. Both are file-level licences: you can use them unmodified in any project, including commercial or closed-source ones.

Regenerate this list at any time:

```bash
pip install pip-licenses && pip-licenses --from=mixed --order=license
```

## External programs (optional, installed separately)

| Program | Licence | Role |
|---|---|---|
| Ollama | MIT | Default local model server |
| llama.cpp (`llama-server`) | MIT | Alternative model server (OpenAI-compatible) |
| vLLM / SGLang | Apache-2.0 | Alternative GPU model servers |
| LibreOffice | MPL-2.0 | PPTX → PDF export (headless) |
| SearXNG | AGPL-3.0 | Optional self-hosted search engine. AGPL only matters if you modify it *and* offer it as a network service to others. |

## Models

The recommended models (Qwen3.5 / 3.6 / 3.8, Gemma 4, Granite 4.2) are all **Apache-2.0** open-weight models (see [MODELS.md](MODELS.md)). "Open weights" means the trained weights are freely downloadable and usable commercially; the training data and pipeline are not fully published, which is normal for current LLMs.

## What cannot be done purely with open source

1. **Web search depends on someone's search index.** The search *clients* here (ddgs, SearXNG) are open source and need no key or payment, but they work by requesting the public result pages of search engines (DuckDuckGo, Bing, Brave, ...) and parsing them. `ddgs` also presents browser-like request headers to those engines. Consequences:
   * results depend on those engines' availability, and **heavy or shared-IP use gets rate-limited**. During development, a cloud sandbox was throttled (HTTP 202/403/429) after a few dozen queries;
   * using engines this way may not match each engine's terms of service. Check them if you deploy this for an organisation;
   * when an engine changes its pages, the scraper may break until `ddgs` is updated (`pip install -U ddgs`).

   The agent mitigates this. When metasearch returns nothing it queries **Wikipedia's official MediaWiki API**, which is free, needs no key or account, and is not scraped. The agent identifies itself with an honest User-Agent (`deck-creator-agent/0.1 (+repo URL)`), as Wikimedia's policy requires. After two empty searches it stops researching and builds from what it has, without inventing figures. Pages fetched by `read_webpage` use the same honest User-Agent, so sites that block bots will refuse them and the agent picks another source.

   No open-source project provides a fresh, full-web index you can run yourself. YaCy (GPL) is a peer-to-peer engine you can self-host, but its coverage is far below the engines above. The official APIs of Brave, Bing, Google, Tavily, Exa and others are paid or freemium, so they are excluded. **Workarounds:** self-host SearXNG for more control and engine choice, or run with `--no-web` and supply your own source material (`--request-file notes.md`).

2. **Image licensing cannot be guaranteed automatically.** `search_images` asks the search engine for Creative-Commons-licensed images (`license_image=Share` by default), but that label comes from the engine's metadata. Check the source page before publishing a deck. Image credits are added to slides and notes automatically.

3. **PowerPoint itself is proprietary.** The agent writes standard `.pptx` files without needing PowerPoint. They open in PowerPoint, Keynote, Google Slides and LibreOffice Impress (open source). PDF conversion uses LibreOffice. Its layout is very close to PowerPoint's but not pixel-identical, mainly through font substitution (e.g. Calibri → Carlito).

4. **Theme fonts.** Themes reference common fonts (Calibri, Arial, Georgia, Helvetica). Fonts are not bundled or distributed; the viewer's installed fonts (or metric-compatible open fonts such as Carlito / Liberation) are used.

Nothing in the project uses a paid library, a freemium API, or a service that needs an account.
