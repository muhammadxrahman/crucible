# Crucible

A self-hosted, OpenAI and Anthropic compatible inference platform that runs text, vision, embedding, and reranker models on Apple Silicon through MLX. It provides continuous batching, a RAG pipeline, hot-swappable LoRA adapters, a capability-aware web UI, and Prometheus observability. Everything runs locally: no cloud, no API keys.

Primary target: MacBook Pro M5 Pro, 20-core GPU, 64GB unified memory, 307GB/s bandwidth. The same codebase scales down to 16GB Macs through hardware profiles.

This file is the entry point. Keep it short. Detailed, stable reference lives in `docs/` and is linked below. Read the relevant doc before working in that area.

## Hard constraints (do not violate)

These are the rules that are expensive to get wrong. Treat them as invariants.

- The inference engine runs NATIVE on the host. Docker on macOS runs in a Linux VM with no Metal or Neural Accelerator access, so containerizing the engine forces CPU execution and defeats the platform. The end goal is a self-contained installable app, so the default install adds no external daemons: observability ships in-app (native `/metrics` exposition plus an `/observability` dashboard). External Prometheus/Grafana are an optional add-on, never required, and Docker is only ever acceptable for such optional CPU side-services. See `docs/hardware.md`.
- MLX is the backend. It is the only path to the M5 Neural Accelerators (via Metal 4 TensorOps). Build on `mlx-lm`, `mlx-vlm`, `mlx-embeddings`. A `llama.cpp` adapter is optional, behind the same interface, as a benchmark baseline. See `docs/architecture.md`.
- Decode is memory-bandwidth-bound; prefill is compute-bound. Prefer Mixture-of-Experts models, which read few active parameters per token. Report prefill and decode throughput as separate metrics. See `docs/hardware.md`.
- Do not assume 64GB. Detect `hw.memsize` at startup and select a hardware profile. Below 32GB, serve a single resident model. See `docs/models.md`.
- Default to 4-bit quantization. Use 8-bit for models under 3B, where 4-bit degrades quality.
- Bind to `127.0.0.1` by default. An open inference endpoint is a resource-abuse risk.
- The web UI is a client, not a backdoor. It calls the same public API as any other client and holds no special privileges. See `docs/ui.md`.
- Never hardcode model paths in source. Models are declared in `config/models.yaml`. See `docs/models.md`.
- Respect the build boundary. Build the orchestration layer (gateway, model manager, scheduler, batching, RAG, adapter routing, UI, observability). Do not reimplement MLX kernels, the transformer forward pass, or quantization. See `docs/architecture.md`.

## What to build vs reuse

Build: the API gateway, the model manager, the request scheduler and batching glue, the RAG pipeline, the vision routing, the LoRA adapter manager, the web UI, the observability and benchmark layers.

Reuse: `mlx`, `mlx-lm`, `mlx-vlm`, `mlx-embeddings` as the model runtime. Study, but do not vendor wholesale, the reference servers listed in `docs/architecture.md`.

## Build order

Work milestones in order. Do not start a milestone before the previous one passes its acceptance criteria. The current milestone is tracked below. Full milestone definitions with acceptance criteria are in `docs/roadmap.md`.

Current milestone: M5 (RAG layer)

## Commands

```
uv run mlxd serve            # start the server (gateway + UI + orchestration)
uv run mlxd models           # list, load, unload, pin models
uv run mlxd bench <spec>     # run the benchmark harness
uv run mlxd profile          # show detected hardware and active profile
pytest                       # run tests
```

## Conventions (summary)

Python via `uv`. FastAPI, async. Pydantic for config. Frontend in Vite, React, Tailwind, shadcn/ui. Concise code with minimal comments, preferring clear names. Type-annotate public functions and config models. Source never hardcodes a model path or memory limit; both come from config and the active profile. Tests with pytest, and each milestone ships acceptance tests before the next begins. Bind to `127.0.0.1` by default. Report prefill and decode throughput separately, never blended.

## Documentation index

- `docs/architecture.md`: system layers, components, data flow, the build boundary, reference servers.
- `docs/hardware.md`: Apple Silicon constraints, the bandwidth and MoE rationale, memory budgeting, the Docker-no-GPU rule.
- `docs/models.md`: the model registry format, the selection matrix, quantization policy, hardware profiles, scaling to smaller Macs.
- `docs/api.md`: the HTTP API contract (endpoints, request and response shapes).
- `docs/ui.md`: the web UI structure, views, and capability-aware rendering.
- `docs/roadmap.md`: milestones, build lists, and acceptance criteria.
