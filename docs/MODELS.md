# Choosing a local model (researched November 2026)

The agent works with any local model that supports **tool calling**. This page summarises the open-weight small language models (SLMs) available as of **4 November 2026** that run on a CPU or on a GPU with 10–20 GB of memory, and which one to pick.

Licensing: every model recommended here is released under **Apache-2.0**: free for commercial use, with no usage caps.

## TL;DR: what to install

| Your hardware | Recommended model | Ollama tag | Download | Why |
|---|---|---|---|---|
| CPU only, 8–16 GB RAM | Qwen3.5-4B | `qwen3.5:4b` | 3.3 GB | Fastest model that still drives the tools reliably. |
| **CPU only, 16 GB RAM (default)** | **Qwen3.5-9B** | **`qwen3.5:9b`** | **6.6 GB** | Best tool-calling score under 10B (BFCL-V4 66.1, TAU2 79.1). |
| CPU only, 32 GB+ RAM | Qwen3.6-35B-A3B (MoE) | `qwen3.6:35b` | ~23 GB | Only 3B parameters active per token, so it runs at small-model speed with much larger-model quality. |
| GPU 10–16 GB | Gemma 4 12B, or Qwen3.5-9B at Q8 | `gemma4:12b` / `qwen3.5:9b-q8_0` | 8 GB / 10 GB | Gemma 4 12B: TAU2 69.0, 256K context. Qwen 9B at Q8 for less quantization loss. |
| GPU 16–20 GB | Gemma 4 26B-A4B (MoE) | `gemma4:26b` | 16–19 GB | 3.8B active params, so it is fast; TAU2 68.2. |
| **GPU 20–24 GB** | **Qwen3.8-27B** | **`qwen3.8:27b`** | **18 GB** | Newest and strongest open model that fits (released 14 Aug 2026). |
| Alternative (text-only) | IBM Granite 4.2 8B | `granite4.2:8b` | 5.3 GB | Apache-2.0, tool calling + low/no-thinking modes (Aug 2026). |

```bash
ollama pull qwen3.5:9b                         # default
deck-agent run "..." -m qwen3.6:35b            # use another model
```

All sizes above are Ollama's default **Q4_K_M** quantization unless stated. Q4_K_M keeps tool-calling accuracy close to full precision for these models. Avoid going below ~3-bit (e.g. `Q2_K`) for agent work: malformed tool calls increase sharply.

## The candidates in detail

### Qwen3.5 small models (Alibaba, 2 March 2026): the default
* Sizes: 0.8B, 2B, 4B, 9B (dense, hybrid Gated-DeltaNet + attention), plus vision input.
* Licence: Apache-2.0. Context: 262K tokens natively.
* Vendor-reported agent scores ([model card][qwen35-9b]):

  | | BFCL-V4 (function calling) | TAU2-Bench (agentic tool use) |
  |---|---|---|
  | Qwen3.5-9B | **66.1** | 79.1 |
  | Qwen3.5-4B | 50.3 | **79.9** |

* Artificial Analysis rated the 9B roughly twice as capable as the next sub-10B models when it launched ([article][aa]).
* Thinking mode is **on by default**. The agent turns it off (`think: false`) because on a CPU it adds thousands of tokens per step. Enable it with `--think` on a GPU if plans look shallow.
* Recommended sampling (non-thinking): temperature 0.7, top_p 0.8, top_k 20. The agent defaults to temperature 0.6, top_p 0.95, top_k 20, which kept JSON arguments stable in testing.

### Qwen3.6-35B-A3B (16 April 2026) and Qwen3.6-27B (22 April 2026)
* 35B-A3B is a mixture-of-experts model: 35B total, **3B active** per token ([model card][qwen36]). Because only 3B weights are used per token, it is the best quality you can get on a **CPU with 32 GB+ RAM** at a usable speed.
* Agentic scores (vendor): TAU3-Bench 67.2, MCP-Atlas 62.8, MCPMark 37.0.
* Apache-2.0.

### Qwen3.8-27B (14 August 2026): the strongest model that fits ~20 GB
* Dense 27B, vision input, 262K context, Apache-2.0 ([model card][qwen38]).
* Large agentic gains over Qwen3.6-27B (vendor): OSWorld-Verified 84.3 vs 63.9, WebArena-Verified 64.8 vs 48.8, SWE-bench Pro 61.7 vs 53.5.
* Q4_K_M is ~18 GB, so it fits a 20–24 GB GPU (or a 16 GB GPU with partial CPU offload, slower).
* Only the 27B and a 2.4T-A95B flagship were open-sourced in the 3.8 generation ([release log][qwen-gh]). There is **no small (<10B) Qwen3.8** yet.

### Gemma 4 (Google, April 2026; 12B added later)
* Sizes: E2B, E4B, 12B (dense), 26B-A4B (MoE, 3.8B active), 31B (dense) ([model card][gemma4]).
* **Licence changed to Apache-2.0** with Gemma 4 (earlier Gemma versions used a custom licence).
* Native function calling and a system role. Tau2 (vendor): 31B 76.9, 12B 69.0, 26B-A4B 68.2, E4B 42.2, E2B 24.5.
* The E2B/E4B edge models are too weak at multi-step tool use for this agent. Use 12B or larger.

### IBM Granite 4.2 (25 August 2026)
* 3B, 8B, 30B, Apache-2.0, native tool calling, with thinking / low-effort / non-thinking modes.
* IBM-reported BFCL-V4 is about 50–52 (3B/8B) and 61 (30B). Secondary sources disagree on exact numbers ([guide][granite]).
* Text-only, and a good alternative if you prefer a Western-vendor model with enterprise documentation.

### Others considered
* **Ministral 3** (Mistral, Dec 2025; 3B/8B/14B, Apache-2.0): solid, but superseded by the 2026 models above for tool use ([source][ministral]).
* **Mistral Small 4** (Mar 2026) is 119B total parameters, far too large here.
* **Phi-4-mini, SmolLM3, Llama 3.2 3B**: run anywhere, but noticeably weaker at multi-step tool calling than Qwen3.5-4B.
* **Qwen 4** had been announced but **not released** as of late October 2026. Re-check if you read this later.

> Benchmarks above are mostly **vendor-reported**. Treat them as a guide and try two models on your own topics; `deck-agent doctor -m <model>` runs a quick tool-calling check.

## Measured speed on a small CPU box

These measurements come from the development sandbox (4 vCPU, 15 GB RAM, no GPU), Ollama 0.40.14:

| Model | Prompt processing | Generation | One full 8-slide deck incl. web research |
|---|---|---|---|
| `qwen3.5:4b` (Q4_K_M) | ~68 tok/s | ~6.6 tok/s | see `README.md` → "Tested end to end" |

A modern 8-core laptop is roughly 2× faster, and any recent GPU is 10–30× faster. Ollama caches the shared prompt prefix between steps, so only new text is re-processed. That makes multi-step agent runs practical even on a CPU.

## Running without Ollama

Any OpenAI-compatible server that supports tools works (`--backend openai`). All are open source:

```bash
# llama.cpp (MIT): --jinja enables the model's own tool-call template
llama-server -hf unsloth/Qwen3.5-9B-GGUF:Q4_K_M --jinja -c 32768 --port 8080
deck-agent run "..." --backend openai --base-url http://localhost:8080/v1 -m qwen3.5-9b

# vLLM (Apache-2.0), GPU
vllm serve Qwen/Qwen3.8-27B --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 --max-model-len 32768
deck-agent run "..." --backend openai --base-url http://localhost:8000/v1 -m Qwen/Qwen3.8-27B
```

If a server returns tool calls as plain text (missing or outdated parser), the agent recovers them from the text, recognising Hermes-JSON and Qwen XML formats.

[qwen35-9b]: https://huggingface.co/Qwen/Qwen3.5-9B
[qwen36]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
[qwen38]: https://huggingface.co/Qwen/Qwen3.8-27B
[qwen-gh]: https://github.com/QwenLM/Qwen3.8
[gemma4]: https://huggingface.co/google/gemma-4-12b-it
[aa]: https://artificialanalysis.ai/articles/qwen3-5-small-models
[granite]: https://aicybr.com/blog/ibm-granite-4-2-reasoning-models-local-ai
[ministral]: https://www.eweek.com/news/mistral-3-launch/

Ollama tag sizes: [qwen3.5](https://ollama.com/library/qwen3.5/tags), [qwen3.6](https://ollama.com/library/qwen3.6), [qwen3.8](https://ollama.com/library/qwen3.8), [gemma4](https://ollama.com/library/gemma4), [granite4.2](https://ollama.com/library/granite4.2).
