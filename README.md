# Crucible

A self-hosted, OpenAI- and Anthropic-compatible inference platform that runs text,
vision, embedding, and reranker models on Apple Silicon through MLX. Everything runs
locally: no cloud, no API keys.

The inference engine runs **native on the host**, not in Docker. Docker on macOS runs
in a Linux VM with no Metal access, which forces CPU execution and defeats the platform.
Docker is used only for stateless CPU side-services (Prometheus, Grafana). See
`docs/hardware.md`.

## Status

Milestone **M6** (vision). Image-plus-text requests via the OpenAI vision shape
(`image_url`, HTTP or base64): image-bearing requests route to a VLM (`mlx-vlm`),
text-only to the text model, with an image cache reused across turns. Plus M5 (RAG:
embeddings, vector store, two-stage retrieval, grounded citations), M4 (observability +
benchmarking), M3 (continuous batching + prefix KV-cache), M2 (model manager), and M1
(OpenAI-compatible gateway). See `docs/roadmap.md`.

## Observability

The dashboard is in-app and self-contained — open `http://127.0.0.1:8000/observability`
once the server is running. `GET /metrics` exposes standard Prometheus text, so an external
Prometheus + Grafana can scrape it if long-term retention is wanted, but neither is required
and nothing here uses Docker.

## Setup

```
uv sync                      # create the venv and install deps (Python 3.12)
./scripts/install-hooks.sh   # install the pre-push regression gate
```

## Commands

```
uv run mlxd profile          # show detected hardware and the active profile
uv run mlxd serve -c config/dev.yaml   # start the gateway on a tiny model (fast)
uv run mlxd serve            # start the gateway on the production registry
uv run mlxd models           # list, load, unload, pin models (M2+)
uv run mlxd bench benchmarks/specs/tiny.yaml   # run the benchmark harness (M4+)
uv run pytest                # run tests
./scripts/check.sh           # run the full pre-push gate (ruff + pytest)
```

## Layout

```
src/crucible/   the package (config, hardware, cli; gateway and orchestration land later)
config/         models.yaml and hardware profiles
docs/           architecture, hardware, models, api, ui, roadmap, conventions
tests/          pytest acceptance and regression suite
benchmarks/     benchmark harness and reports
web/            the Vite + React UI (M9)
scripts/        check.sh (regression gate), install-hooks.sh, smoke_generate.py
```
