# deck-creator-agent

An AI agent that turns a request like *"10 slides on heat pumps for our city council"* into a finished presentation. It researches the topic on the web, plans the deck, writes the slides (charts, tables, comparisons, speaker notes, sources), and saves **PowerPoint, PDF and HTML** files.

* **Runs a local model.** Default `qwen3.5:9b` through [Ollama](https://ollama.com); works on a CPU or a 10–20 GB GPU. See [docs/MODELS.md](docs/MODELS.md).
* **Built entirely on open source.** No API keys, accounts, paid or freemium services. See [docs/LICENSES.md](docs/LICENSES.md).
* **Tuned for small models.** Forgiving argument parsing, precise error feedback, loop guards, research budgets and an auto-save, so a 4B–9B model can complete a whole deck.

## Quick start

```bash
# 1. A local model server (Ollama, MIT licence) and a model
#    https://ollama.com/download
ollama pull qwen3.5:9b          # 6.6 GB; use qwen3.5:4b (3.3 GB) on a small machine

# 2. The agent
git clone https://github.com/chaliharohan2/deck-creator-agent.git
cd deck-creator-agent
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .

# 3. Optional: LibreOffice for PDF export (open source)
#    Debian/Ubuntu: sudo apt install libreoffice-impress   macOS: brew install --cask libreoffice

# 4. Check everything, then build a deck
deck-agent doctor
deck-agent run "Create a 10-slide deck on the adoption of heat pumps in Europe for a city council. Include a chart." -f pptx pdf html
```

Files are written to `./decks/`: `<title>.pptx`, `.pdf`, `.html` and an editable `<title>.deck.json`.

Try the renderer without any model: `deck-agent sample -f pptx html pdf`.

## Which model should I use?

| Hardware | Model | `-m` |
|---|---|---|
| CPU, 8–16 GB RAM | Qwen3.5-4B | `qwen3.5:4b` |
| **CPU, 16 GB RAM (default)** | **Qwen3.5-9B** | **`qwen3.5:9b`** |
| CPU, 32 GB+ RAM | Qwen3.6-35B-A3B (MoE, 3B active) | `qwen3.6:35b` |
| GPU 10–16 GB | Gemma 4 12B | `gemma4:12b` |
| GPU 20–24 GB | Qwen3.8-27B | `qwen3.8:27b` |

All Apache-2.0. Benchmarks, sources, alternatives (Gemma 4 26B-A4B, Granite 4.2) and measured CPU speeds are in **[docs/MODELS.md](docs/MODELS.md)**.

## Usage

```bash
# One request → one deck (autonomous)
deck-agent run "8 slides comparing solar and wind for a high-school class" --theme warm
deck-agent run "Investor update for Q3" --request-file notes.md --no-web    # use your own material, offline
deck-agent run "8-slide briefing for a city council, with a chart" \
    --request-file examples/heat_pumps_notes.md --no-web                   # try it with the bundled example
deck-agent run "..." -m qwen3.6:35b --think        # bigger model + reasoning mode (best on a GPU)

# Conversation: build, then refine
deck-agent chat
  you> a 12-slide training deck on phishing awareness for new employees
  you> make slide 5 a two-column do/don't comparison
  you> add a quote slide before the summary, then save as pdf too
  # chat commands: /slides  /save pptx pdf html  /new  /quit

# Re-render a saved deck (after hand-editing the JSON, or to switch theme), no model needed
deck-agent render decks/Heat_Pumps.deck.json -f pptx pdf --theme dark

deck-agent themes      # light, dark, corporate, warm, forest, minimal
deck-agent doctor      # checks model server, tool calling, web search, LibreOffice
```

The formats you pass with `-f` are always produced (the model can add formats, not remove them).

Common options: `-m/--model`, `-f/--formats pptx pdf html`, `-t/--theme`, `-o/--out-dir`, `--slides N`, `--no-web`, `--no-fact-check`, `--think`, `--num-ctx`, `-v/--verbose` (shows every tool result). Run `deck-agent run -h` for all options.

## What the agent can do (its tools)

| Research | Building | Editing |
|---|---|---|
| `web_search`: key-less metasearch, Wikipedia fallback | `set_deck_info`: title, subtitle, author, theme | `list_slides` |
| `read_webpage`: main text of an HTML page or PDF | `add_bullet_slide`: bullets, sub-points, **bold** | `update_slide` |
| `search_images`: Creative-Commons-filtered images | `add_two_column_slide`: comparisons | `delete_slide` |
| | `add_chart_slide`: native, editable PowerPoint charts: column, bar, line, area, pie, doughnut, stacked | `move_slide` |
| | `add_table_slide`, `add_image_slide`, `add_quote_slide`, `add_section_slide` | `save_deck` |

Every slide can carry **speaker notes** and **sources**. Sources appear in a footer line and in full in the notes.

## How it works

```
 you ──► deck-agent (Python) ──────────────► local model server (Ollama / llama.cpp / vLLM)
            │  agent loop: prompt → tool calls → results → … → save
            ├── web tools ──► ddgs metasearch → Wikipedia API fallback → trafilatura / pypdf
            ├── deck tools ─► Deck spec (pydantic) ─┬─► python-pptx ─► .pptx ─► LibreOffice ─► .pdf
            │                                       └─► HTML + inline SVG charts ─► .html (offline)
            └── guard rails: arg coercion · error feedback · loop/research limits · context trimming · auto-save
```

| Path | Purpose |
|---|---|
| `src/deck_agent/agent.py` | Agent loop and system prompt |
| `src/deck_agent/llm.py` | Ollama-native and OpenAI-compatible streaming clients (plain `httpx`, no vendor SDK) |
| `src/deck_agent/toolcall_parse.py` | Recovers tool calls a server returned as text (Hermes JSON, Qwen XML) |
| `src/deck_agent/tools/` | Tool registry, deck tools, web tools |
| `src/deck_agent/deck/` | Deck schema, themes, PPTX / HTML / PDF renderers |

### Why it works with small models

* **Simple, flat tool schemas.** One tool per slide type with flat parameters instead of one deeply nested "deck" object.
* **Forgiving inputs, strict outputs.** Bullets sent as one string, `"45%"` instead of `45`, `2024` instead of `"2024"`, and `{"Revenue": [...]}` instead of a series list are all accepted. Mismatched chart data or oversized tables come back as precise, fixable errors.
* **Guard rails.** Repeated identical calls are short-circuited, research is capped per request (`DECK_AGENT_MAX_SEARCHES`, default 4), and exhausted tools are removed from the tool list (small models ignore "stop searching" but not a missing tool). A model that only *describes* a plan is nudged to call tools.
* **Fact and citation checks.** Small models invent statistics *and* plausible-looking source URLs. Every number and URL the agent sees (your input, search snippets, pages it read) is recorded:
  * a **chart** whose values appear in none of that material is **rejected**, with the unsupported values named, so the model has to chart only real data;
  * figures in text slides that appear nowhere get a warning back to the model;
  * a cited URL the agent never saw is **removed** from the slide.

  This catches numbers and links that were made up outright, the common failure. It cannot prove a real number is used in the right context, so still review a deck before presenting it.
* **Never lose work.** Old tool output is trimmed near the context limit, and an unsaved deck is always saved at the end.
* **Thinking off by default.** Reasoning mode multiplies CPU time. Turn it on with `--think` on a GPU.

## Tested end to end

The agent was run against real local models on a 4-vCPU, CPU-only machine (Ollama 0.40.14):

| Run | Result |
|---|---|
| `qwen3.5:4b`, web research on | 8 slides with chart, table, two-column and quote in 9.5 min. The first searches found real data (EHPA: 2.9 M heat pumps sold in 2025, +13%). The search engines then throttled the machine's shared IP. With research cut off, the model **invented figures and source URLs**, which is what led to the fact and citation checks above. |
| `qwen3.5:9b`, `--request-file examples/heat_pumps_notes.md --no-web` | 10 slides in 15 min, 0 tool errors; every figure in the text slides matched the notes. The chart mixed two real and two **invented** values while citing the real source, which led to the number cross-check. It also added a duplicate slide, so duplicates are now refused. |
| same, with all guard rails | 11 slides in 13 min. The model charted an invented value (Germany = 1,700); the chart was **rejected**, and the model re-added it with only the real figures. It deleted an unsourced attributed quote after the warning. All figures and sources were correct. One wrong name slipped through ("Norfolk" for Norway, with the right number). Number checks can't catch that, so **review a deck before presenting it**. Afterwards the prompt was tightened on slide count and the `author` field. |

These runs are why the guard rails exist. They are covered by unit tests with a scripted fake model (`tests/test_agent.py`, `tests/test_tools.py`, `tests/test_factcheck.py`).

## Configuration

Every option can also be set with an environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `DECK_AGENT_BACKEND` | `ollama` | `ollama` or `openai` (llama.cpp server, vLLM, SGLang, LocalAI) |
| `DECK_AGENT_BASE_URL` | per backend | e.g. `http://localhost:11434`, `http://localhost:8080/v1` |
| `DECK_AGENT_MODEL` | `qwen3.5:9b` | model name / tag |
| `DECK_AGENT_NUM_CTX` | `32768` | context window (Ollama) |
| `DECK_AGENT_THINK` | `false` | reasoning mode |
| `DECK_AGENT_FORMATS` | `pptx` | comma-separated: `pptx,pdf,html` |
| `DECK_AGENT_OUTPUT_DIR` | `decks` | output folder |
| `DECK_AGENT_THEME` | – | preferred theme |
| `DECK_AGENT_WEB` | `true` | enable web tools |
| `DECK_AGENT_SEARCH_BACKEND` | `ddgs` | `ddgs`, `searxng` or `wikipedia` |
| `DECK_AGENT_WIKIPEDIA_FALLBACK` | `true` | use Wikipedia's API when the main search returns nothing |
| `DECK_AGENT_SEARXNG_URL` | `http://localhost:8888` | your SearXNG instance |
| `DECK_AGENT_MAX_SEARCHES` / `_MAX_READS` | `4` / `4` | research budget per request |
| `DECK_AGENT_IMAGE_LICENSE` | `Share` | image licence filter: `any`, `Public`, `Share`, `ShareCommercially`, `Modify`, `ModifyCommercially` |
| `DECK_AGENT_VERIFY_NUMBERS` | `true` | cross-check slide numbers against sources (`--no-fact-check` to allow illustrative numbers) |
| `DECK_AGENT_ALLOW_PRIVATE_URLS` | `false` | allow reading `localhost`/LAN pages (off by default to block prompt-injection tricks) |

`HTTPS_PROXY` and `SSL_CERT_FILE` are honoured for web research, which helps behind corporate proxies.

### Other model servers

```bash
# llama.cpp (MIT)
llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M --jinja -c 32768 --port 8080
deck-agent run "..." --backend openai --base-url http://localhost:8080/v1 -m qwen3.5-9b
```

More options, including vLLM, are in [docs/MODELS.md](docs/MODELS.md#running-without-ollama).

### Web search options and limits

* **Default: `ddgs` (MIT).** Key-less metasearch over DuckDuckGo, Bing, Brave, Mojeek and others, with Wikipedia's official API as an automatic fallback.
* **Self-hosted: SearXNG (AGPL-3.0).** More engines and more control:
  ```bash
  docker run -d --name searxng -p 8888:8080 -v "$PWD/searxng:/etc/searxng" searxng/searxng
  # enable JSON output: add "- json" under `search: formats:` in ./searxng/settings.yml, then restart
  deck-agent run "..." --search-backend searxng
  ```
* **Offline:** `--no-web`, with your own material via `--request-file`.

Key-less search depends on public search engines, which **rate-limit heavy use**. When that happens the agent falls back to Wikipedia and then builds from what it has; it doesn't invent figures. See [docs/LICENSES.md](docs/LICENSES.md#what-cannot-be-done-purely-with-open-source).

## What open source can't fully do

Everything here is open source. Two limitations are inherent and you should know about them:

1. **Web search needs someone's index.** The open-source clients query public engines, so availability depends on them. No open-source, self-hostable index of the whole web is fresh enough to replace them.
2. **Image licences are best-effort.** The Creative Commons filter relies on search-engine metadata, so check images before publishing.

Details: [docs/LICENSES.md](docs/LICENSES.md).

## Development

```bash
pip install -e ".[dev]"
pytest            # ~100 tests; the agent loop and LLM clients use fakes/mocks, so no model is needed
ruff check src tests
```

