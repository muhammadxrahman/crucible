# Crucible

A self-hosted, OpenAI- and Anthropic-compatible inference platform that runs text,
vision, embedding, and reranker models on Apple Silicon through MLX. Everything runs
locally: no cloud, no API keys.

The inference engine runs **native on the host**, not in Docker. Docker on macOS runs
in a Linux VM with no Metal access, which forces CPU execution and defeats the platform.
The default install adds no external daemons; Docker is only ever used for an optional
monitoring stack (see `ops/`). See `docs/hardware.md`.

## Status

Milestone **M8** (web UI). A clean, chat-centric web app served at `/`: streaming chat with
a model switcher, inline image attachments (vision) and document upload with a Grounded
toggle for cited answers, and a side panel for model load/unload/pin, memory, and live
throughput. Lean stack (React + Vite, no Tailwind/shadcn); the HTTP API is the durable
contract. Plus M7 (packaging), M6 (vision), M5 (RAG), M4 (observability + benchmarking), M3
(continuous batching + prefix KV-cache), M2 (model manager), M1 (gateway). See
`docs/roadmap.md`.

## Web UI

```
bash scripts/build-ui.sh        # build web/dist (Vite)
uv run mlxd serve               # serves the UI at http://127.0.0.1:8000/
```

Chat with a model picker, attach an image (routed to the VLM), drag in a document and
toggle **Grounded** for cited RAG answers, and manage models / watch live throughput in the
side panel. For UI development: `npm --prefix web run dev` (proxies the API to the gateway).

## Observability

The dashboard is in-app and self-contained — open `http://127.0.0.1:8000/observability`
once the server is running. `GET /metrics` exposes standard Prometheus text, so an external
Prometheus + Grafana can scrape it if long-term retention is wanted, but neither is required
and nothing here uses Docker.

## Quickstart (clone to first cited answer)

```
uv sync                                # create the venv, install deps (Python 3.12)
uv run mlxd validate                   # check the registry and active profile
uv run mlxd serve                      # start the server (downloads models on first run)
# in another shell, open the in-app dashboard:
open http://127.0.0.1:8000/observability
# ingest local docs and ask a grounded question:
curl -s localhost:8000/rag/ingest  -H 'content-type: application/json' -d '{"paths":"./docs"}'
curl -s localhost:8000/rag/query   -H 'content-type: application/json' -d '{"query":"What is the build boundary?"}'
```

For fast local iteration on a tiny model: `uv run mlxd serve -c config/dev.yaml`.

## Run on login (native, no Docker)

```
uv run mlxd service install     # install + load a launchd LaunchAgent (RunAtLoad)
uv run mlxd service status      # check it is loaded
uv run mlxd service uninstall   # remove it
```

## Commands

```
uv run mlxd profile                    # detected hardware and active profile
uv run mlxd validate                   # validate the registry, no server start
uv run mlxd serve [-c config.yaml]     # start the OpenAI/RAG/vision gateway
uv run mlxd models list                # list models and residency (server must be up)
uv run mlxd models load|unload|pin <name>
uv run mlxd bench benchmarks/specs/tiny.yaml
uv run mlxd service install|status|uninstall
./scripts/check.sh                     # full pre-push gate (ruff + pytest)
```

Development setup also installs the pre-push regression gate: `./scripts/install-hooks.sh`.

## Layout

```
src/crucible/   gateway (server), orchestration (manager), backends (text/vision/embed/
                rerank), batching, rag, observability, benchmark, cli, client, service
config/         models.yaml (production) and dev.yaml; hardware profiles
docs/           architecture, hardware, models, api, ui, roadmap
tests/          pytest acceptance and regression suite
benchmarks/     benchmark harness specs and reports
ops/            optional external Prometheus/Grafana (not required)
web/            the Vite + React UI (M8)
scripts/        check.sh (regression gate), install-hooks.sh, smoke_*.py
```
