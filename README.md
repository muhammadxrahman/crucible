# Crucible

A self-hosted, OpenAI- and Anthropic-compatible inference platform that runs text,
vision, embedding, and reranker models on Apple Silicon through MLX. Everything runs
locally: no cloud, no API keys.

The inference engine runs **native on the host**, not in Docker. Docker on macOS runs
in a Linux VM with no Metal access, which forces CPU execution and defeats the platform.
Docker is used only for stateless CPU side-services (Prometheus, Grafana). See
`docs/hardware.md`.

## Status

Milestone **M4** (observability and benchmarking). Native Prometheus `/metrics`, an in-app
dashboard at `/observability` (no Docker, no external daemons), and a benchmark harness
(`mlxd bench`) that reports prefill and decode throughput separately and throughput vs
concurrency. Builds on M3 (continuous batching + prefix KV-cache over `mlx-lm`'s
`BatchGenerator`), M2 (model manager: routing, LRU eviction, pin/TTL), and M1
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
