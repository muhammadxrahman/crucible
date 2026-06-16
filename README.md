# Crucible

**A private, local AI server for Apple Silicon.** Run chat, vision, embedding, and reranker
models on your own Mac — with an OpenAI-compatible API, a built-in chat web app, and a
local document-RAG pipeline. No cloud, no API keys, nothing leaves your machine.

Crucible runs **native on the host** (not in Docker) so it can use the Mac's GPU through
Apple's MLX framework. It is the orchestration layer — gateway, model manager, batching,
RAG, vision, observability, and UI — built on top of `mlx-lm`, `mlx-vlm`, and
`mlx-embeddings`.

---

## Contents

- [What it can do](#what-it-can-do)
- [Requirements](#requirements)
- [Get started](#get-started)
- [Running and stopping the server](#running-and-stopping-the-server)
- [Use it: the web app](#use-it-the-web-app)
- [Use it: the API](#use-it-the-api)
- [Command-line reference](#command-line-reference)
- [Configuration](#configuration)
- [Observability](#observability)
- [Run automatically on login](#run-automatically-on-login)
- [Hardware and scaling](#hardware-and-scaling)
- [How it works](#how-it-works)
- [Development](#development)

---

## What it can do

- **Chat** — stream responses from local text models, OpenAI-compatible. Concurrent requests
  are folded into one decode batch (continuous batching) and shared prompt prefixes are
  cached, so multi-turn chat and shared system prompts are fast.
- **Vision** — send an image (URL or base64) with a question; image requests route
  automatically to a vision-language model (VLM) for OCR, document Q&A, and screenshot
  understanding.
- **RAG over your documents** — drop in PDFs, Markdown, or text; Crucible chunks, embeds, and
  indexes them locally, then answers questions with **citations** using two-stage retrieval
  (dense search + cross-encoder rerank). No network access during retrieval or generation.
- **Embeddings & reranking** — `/v1/embeddings` for dense vectors and `/v1/rerank` for
  cross-encoder relevance scoring, usable by your own apps.
- **Multiple models at once** — serve text, vision, embedding, and reranker models from one
  process. The model manager loads on demand, evicts least-recently-used models under a
  memory ceiling, and honors pins and idle timeouts.
- **OpenAI drop-in** — point any OpenAI client (the `openai` Python/JS SDK, Cursor,
  Continue.dev, etc.) at `http://127.0.0.1:8000/v1` by changing only the base URL.
- **Built-in web app** — a clean, ChatGPT-style chat at `/` with a model switcher, image and
  document attachments, a "Grounded" toggle for cited answers, and a side panel for model
  management and live throughput.
- **Live metrics, no extra services** — an in-app dashboard at `/observability` plus a
  standard Prometheus `/metrics` endpoint. Prometheus/Grafana are optional, never required.
- **Reliable generation on any model** — chat-sane sampling defaults with a repetition
  penalty and a runaway-loop guard, so even small/quantized models don't collapse into
  repetition. Everything is overridable per request.
- **Tooling** — a benchmark harness (prefill vs decode throughput, scaling with concurrency),
  hardware-profile auto-detection, and native autostart on login via `launchd`.

---

## Requirements

- An **Apple Silicon Mac** (M1 or newer) running macOS. Crucible uses Metal via MLX; it does
  not run on Intel Macs or in a Linux VM/container.
- **[uv](https://docs.astral.sh/uv/)** — the Python toolchain manager. Install it with:
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  uv manages an isolated Python 3.12 for the project; you do not need to install Python
  yourself.
- **Node.js 18+** — only if you want to (re)build the web UI from source. A prebuilt UI is
  served automatically once you run the build step once.
- **Memory** — 16GB works with the small-model profile; the default config targets a 64GB
  Mac. See [Hardware and scaling](#hardware-and-scaling).

---

## Get started

### Fastest first run (a tiny model, ~0.4GB)

This downloads a small 0.5B model so you can see everything working in a minute or two.

```bash
git clone https://github.com/muhammadxrahman/crucible.git && cd crucible
uv sync                                   # create the venv + install dependencies
bash scripts/build-ui.sh                  # build the web app (one time; skip if API-only)
uv run mlxd serve -c config/dev.yaml      # start the server on the tiny model
```

The server runs **in that terminal** — you'll see log lines, and it keeps running until you
press **Ctrl-C**. It **opens the web app at http://127.0.0.1:8000/ in your browser
automatically** once it's ready (pass `--no-open` to disable that).

> The tiny model is fast but not smart (it will, for example, miscount letters). Use it to
> confirm the setup; use the full models below for real answers.
>
> If you skipped the UI build step, the browser page at `/` won't load — that's expected.
> Either run `bash scripts/build-ui.sh`, or just use the [API](#use-it-the-api).

### Full setup (the default 30B + vision + RAG)

The default registry (`config/models.yaml`) serves a 30B MoE chat model, a vision model, an
embedding model, and a reranker — roughly 35–40GB of downloads. Pull them first so you can
watch the progress, then serve from cache:

```bash
uv sync
bash scripts/build-ui.sh
uv run mlxd validate                      # sanity-check the config + show the active profile
uv run mlxd pull                          # download all model weights, with progress, be mindful of your download speeds
uv run mlxd serve                         # start the server (loads from the local cache)
```

Open **http://127.0.0.1:8000/**.

Tips:
- A multi-gigabyte download can take a while; `mlxd pull` shows progress and is resumable.
  For faster downloads: `HF_HUB_ENABLE_HF_TRANSFER=1 uv run mlxd pull`.
- `serve` never aborts a slow download, and if one model fails to load it is skipped with a
  warning (the server still starts and that model loads on first use).
- On a 16GB/24GB Mac, edit `config/models.yaml` to use smaller models; the active hardware
  profile is selected automatically (see [Configuration](#configuration)).

---

## Running and stopping the server

There are two ways to run Crucible. **Pick one** — don't run both, or they'll fight over the
port.

**1. Manually (recommended for normal use).** Run it in a terminal:

```bash
uv run mlxd serve                  # or: -c config/dev.yaml
```

It stays in the foreground and opens the web app in your browser once it's ready (use
`--no-open` to skip that). To stop it, either:

- **Press `Ctrl-C`** in that terminal, or
- click **⏻ Shut down server** at the bottom of the web app's side panel (a graceful
  shutdown, equivalent to Ctrl-C — handy when you don't have the terminal in front of you).
  You can also `POST /admin/shutdown`.

> Note: the shutdown button does not stop the **login service** (option 2) — its `KeepAlive`
> restarts the server. Use `mlxd service uninstall` for that.

**2. As a background login service (optional).** `mlxd service install` registers a `launchd`
agent that starts the server on login **and automatically restarts it** if it exits. This is
convenient, but it changes how you stop it:

```bash
uv run mlxd service uninstall      # the ONLY way to stop the service
```

> **Important:** while the login service is installed, `kill` (or killing the PID from
> `lsof`) **will not stop the server** — `launchd` immediately relaunches it. You must run
> `mlxd service uninstall` (or `launchctl unload ~/Library/LaunchAgents/com.crucible.mlxd.plist`).
> See [Run automatically on login](#run-automatically-on-login).

**Useful checks:**

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN   # what is using the port
uv run mlxd service status         # is the login service active?
uv run mlxd serve --port 8080      # serve on a different port
```

---

## Use it: the web app

Open `http://127.0.0.1:8000/` after starting the server.

- **Chat** — type and get streaming responses. Pick the model from the dropdown.
- **Attach an image** (📎) — the message is sent to the vision model; ask about screenshots,
  photos, or documents.
- **Attach a document** (📎, `.pdf`/`.txt`/`.md`) — it's indexed into the local knowledge
  base. Turn on **Grounded** and your question is answered from your documents with
  clickable `[1] [2]` citations.
- **Thinking** toggle — for reasoning models (Qwen3), turn it on to see the model's
  step-by-step reasoning in a collapsible "💭 Reasoning" block. Off by default (direct
  answers). Generation is unlimited by default, so reasoning isn't cut off.
- **Side panel** — each model shows its real name (the model `path`, not just `primary`).
  Load/unload/pin models, see the active hardware profile and a memory bar, and watch live
  prefill/decode throughput and time-to-first-token.
- **Add a downloaded model** (＋ in the side panel) — pick any MLX model already in your local
  Hugging Face cache, choose its type, name it, and load it without editing config. It loads in
  the background (the panel stays responsive) and is saved to `config/models.yaml`, so it's
  there after a restart.

Capability-aware: if your config has no vision or embedding model, the UI hides the
corresponding controls automatically.

For UI development with hot reload: `npm --prefix web run dev` (proxies API calls to the
running server).

---

## Use it: the API

The server speaks the OpenAI API. The `model` field is the `served_name` from your config
(`primary`, `vision`, `embed`, `rerank` in the default config; only `primary` in `dev.yaml`).

### Chat (any OpenAI client)

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="primary",
    messages=[{"role": "user", "content": "Explain Apple Silicon unified memory briefly."}],
)
print(resp.choices[0].message.content)
```

Streaming with curl:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"primary","stream":true,
       "messages":[{"role":"user","content":"Write a haiku about the GPU."}]}'
```

### Vision (image + text)

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "vision",
  "messages": [{"role":"user","content":[
    {"type":"text","text":"What does this say?"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,<...>"}}
  ]}]
}'
```

HTTP image URLs work too. Image-bearing requests route to a VLM automatically.

### Embeddings and reranking

```bash
curl http://127.0.0.1:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"embed","input":["unified memory","a banana"]}'

curl http://127.0.0.1:8000/v1/rerank -H 'content-type: application/json' \
  -d '{"model":"rerank","query":"apple silicon memory",
       "documents":["bananas have potassium","the CPU and GPU share memory"]}'
```

### RAG: index documents and ask grounded questions

```bash
# Upload files from the browser/host (multipart):
curl -F files=@notes.pdf -F files=@spec.md http://127.0.0.1:8000/rag/upload

# Or ingest a folder by path (server-side):
curl http://127.0.0.1:8000/rag/ingest -H 'content-type: application/json' \
  -d '{"paths":"./docs"}'

# Ask, and get an answer plus the source chunks used:
curl http://127.0.0.1:8000/rag/query -H 'content-type: application/json' \
  -d '{"query":"What is the build boundary?"}'

curl http://127.0.0.1:8000/rag/documents      # list indexed documents
```

### Other endpoints

- `GET /v1/models` — list models, their type, residency, and memory.
- `POST /admin/models/{load,unload,pin}` — manage residency (also exposed via `mlxd models`).
- `GET /healthz` — status, active profile, resident models.
- `GET /metrics`, `GET /metrics/summary`, `GET /observability` — see [Observability](#observability).

The full contract is in [`docs/api.md`](docs/api.md).

---

## Command-line reference

```
uv run mlxd serve [-c config.yaml]      # start the gateway + web UI + RAG + vision
uv run mlxd pull  [served_name...]      # pre-download model weights into the local cache
uv run mlxd validate [-c config.yaml]   # validate the config and show the active profile
uv run mlxd profile                     # show detected hardware and the chosen profile
uv run mlxd models list                 # list models + residency (server must be running)
uv run mlxd models load|unload|pin <served_name>
uv run mlxd bench benchmarks/specs/tiny.yaml   # benchmark prefill/decode vs concurrency
uv run mlxd service install|status|uninstall   # run on login via launchd (see below)
```

---

## Configuration

All models live in a YAML registry; source code never hardcodes a model path.

- **`config/models.yaml`** — the default/production registry (30B chat + vision + embed +
  rerank), tuned for a 64GB Mac.
- **`config/dev.yaml`** — a single tiny model for fast iteration.

A model entry:

```yaml
models:
  - path: mlx-community/Qwen3-30B-A3B-4bit   # a Hugging Face repo (prefer mlx-community 4-bit)
    type: lm                                  # lm | vlm | embedding | rerank
    served_name: primary                      # the name clients use in the `model` field
    pin: true                                 # never evicted
    # context_length, ttl_seconds, quant are optional
```

Other config sections:

- **`server`** — `host`/`port`, `batching` (continuous batching on/off), and `sampling`
  (chat-sane defaults: `temperature`, `top_p`, `repetition_penalty`, `loop_guard`,
  `max_tokens`). Any field is overridable per request.
- **`profiles`** — per-memory-tier limits (`air16` … `max128`): which models to load,
  single-vs-multi resident, default context length, and whether vision is enabled. The
  profile is **auto-detected** from your Mac's memory; override with `profile:` or
  `mlxd serve --profile`.
- **`rag`** — embedding/reranker/generator roles, `rerank` on/off, chunk size, and how many
  chunks to retrieve and cite.

See [`docs/models.md`](docs/models.md) for the model selection matrix and profiles.

---

## Observability

The dashboard is **in-app and self-contained** — open `http://127.0.0.1:8000/observability`
while the server runs to watch prefill/decode throughput (reported separately), time-to-
first-token, queue depth, batch size, cache hit rate, and resident memory.

`GET /metrics` exposes standard Prometheus text, so you *can* attach an external
Prometheus + Grafana for long-term history (`ops/docker-compose.yml`) — but nothing here
requires it, and the engine itself never runs in Docker.

---

## Run automatically on login

Install a native `launchd` agent so the server starts when you log in (host-native, not a
container):

```bash
uv run mlxd service install      # writes + loads a LaunchAgent (RunAtLoad, KeepAlive)
uv run mlxd service status       # check whether it is loaded
uv run mlxd service uninstall    # stop it and remove the agent
```

Because the agent uses `KeepAlive`, the server **restarts automatically** if it crashes or
is killed — which also means a plain `kill` won't stop it. To actually stop it, run
`mlxd service uninstall`. Logs are written under `.crucible/logs/`.

---

## Hardware and scaling

- **Native, for the GPU.** Docker on macOS runs in a Linux VM with no Metal access, which
  forces CPU execution. Crucible runs on the host; Docker is only ever used for the optional
  monitoring stack in `ops/`.
- **MoE-first.** Decode speed is limited by memory bandwidth, not compute. A 30B
  Mixture-of-Experts model (≈3B active) reads few bytes per token and feels like a cloud API
  (~50–90 tok/s on an M5 Pro), while a dense 70B is bandwidth-bound and sluggish. Prefer MoE
  models; prefill stays fast on the Neural Accelerators regardless.
- **Scales 16GB → 128GB.** The same build runs across Macs by auto-selecting a hardware
  profile; below 32GB it serves a single resident model and a smaller default. See
  [`docs/hardware.md`](docs/hardware.md).

---

## How it works

```
  Web UI / API clients (OpenAI SDK, curl, Cursor, …)
                     │  HTTP
        ┌────────────▼─────────────┐
        │  Gateway (FastAPI)       │  /v1/* OpenAI surface, /rag/*, /metrics, /
        ├──────────────────────────┤
        │  Orchestration           │  model manager (LRU, pin, TTL, memory budget),
        │                          │  request scheduler + continuous batching, RAG pipeline
        ├──────────────────────────┤
        │  Backends                │  mlx-lm (text) · mlx-vlm (vision) · mlx-embeddings
        ├──────────────────────────┤
        │  MLX → Metal → GPU       │  unified memory, zero-copy
        └──────────────────────────┘
```

Architecture details and the build boundary are in [`docs/architecture.md`](docs/architecture.md).

---

## Development

```bash
uv sync                         # install deps (incl. dev tools)
./scripts/install-hooks.sh      # install the pre-push regression gate (ruff + pytest)
uv run pytest                   # run the test suite
./scripts/check.sh              # the full gate: ruff format + lint + pytest
```

Tests are GPU-free (they use deterministic fakes); end-to-end checks against real models live
in `scripts/smoke_*.py`. Because the engine is Mac/Metal-specific, the gate runs locally
before pushing.

### Project layout

```
src/crucible/   server (gateway), manager (orchestration), backends (text/vision/embed/
                rerank + image parsing + loop guard), batching, rag, observability,
                benchmark, cli, client, service
config/         models.yaml (default) and dev.yaml; hardware profiles + sampling defaults
web/            the Vite + React chat app (built to web/dist, served at /)
docs/           architecture, hardware, models, api, ui, roadmap
benchmarks/     benchmark specs and generated reports
ops/            optional external Prometheus/Grafana (not required)
scripts/        check.sh (gate), build-ui.sh, install-hooks.sh, smoke_*.py
```

### Status

Crucible is built in milestones; text, vision, embeddings, reranking, RAG, the model
manager, observability, packaging, and the web UI are in place. LoRA fine-tuning and
adapter serving are the remaining planned work. An Anthropic `/v1/messages` surface is on
the roadmap but not yet implemented (the API is OpenAI-compatible today). See
[`docs/roadmap.md`](docs/roadmap.md).
