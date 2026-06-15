# Architecture

## Overview

Three bands. The orchestration layer in the middle is the product. The MLX stack below it is a dependency. The serving and client layer above it is standard web infrastructure.

```
        +-------------------------------+   +-----------------------------+
        |  Web UI (control plane)       |   |  API clients                |
        |  chat, vision, RAG, models,   |   |  Claude Code, Cursor,       |
        |  fine-tune jobs, live metrics |   |  Continue.dev, curl, ext.   |
        +---------------+---------------+   +--------------+--------------+
                        |                                  |
                        +----------------+-----------------+
                                         |
        +--------------------------------v----------------------------------+
        |  API Gateway (FastAPI, async)                                     |
        |  OpenAI:   /v1/chat/completions  /v1/completions                  |
        |            /v1/embeddings  /v1/rerank  /v1/models                 |
        |  Anthropic:/v1/messages                                           |
        |  Ops:      /metrics (Prometheus)  /healthz                        |
        +---------------------------------+---------------------------------+
                                          |
        +---------------------------------v---------------------------------+
        |  Orchestration                                                    |
        |  Model Manager (LRU, pin, TTL, memory accounting, profiles)       |
        |  Request Scheduler  -> Continuous Batching                        |
        |  Adapter Manager (LoRA hot-swap, per-request routing)             |
        +------+-----------------+------------------+----------------+------+
               |                 |                  |                |
        +------v-----+   +-------v------+   +--------v------+   +-----v------+
        |  Text      |   |  Vision      |   |  Embeddings   |   |  RAG       |
        |  mlx-lm    |   |  mlx-vlm     |   |  mlx-embed.   |   |  pipeline  |
        +------+-----+   +-------+------+   +--------+------+   +-----+------+
               |                 |                  |                |
        +------v-----------------v------------------v----------------v------+
        |  MLX runtime  ->  Metal 4 / TensorOps  ->  Neural Accelerators    |
        |  unified memory, zero-copy                                        |
        +-------------------------------------------------------------------+
                  ^ optional: custom Metal kernel (KV-cache / matmul)
```

## Components

### API gateway
FastAPI, async, single process. Exposes the OpenAI-compatible surface, the Anthropic messages surface, the ops endpoints, and serves the built web UI bundle at `/`. Maps inbound requests onto orchestration calls and streams responses over SSE. The full endpoint contract is in `api.md`.

### Model manager
Owns the lifecycle of every loaded model. Responsibilities: load and unload, track resident memory, evict by LRU, honor pins and per-model TTL, and enforce a memory ceiling so the unified memory pool is never oversubscribed. The ceiling and the single-vs-multi-resident behavior come from the active hardware profile, so the manager runs unchanged across memory tiers. Routes each request to the model named in its payload. Memory introspection uses MLX (`mx.get_active_memory`, `mx.set_memory_limit`, and related).

### Request scheduler and continuous batching
Admits requests through a queue and folds new requests into the running decode batch instead of serializing them (in-flight batching). The batching engine is the `mlx-lm` batch generation primitive (`BatchGenerator`); the scheduler is built around it. A prefix or paged KV cache lets shared prompt prefixes (system prompts, RAG context) skip recomputation.

### Adapter manager
Loads LoRA adapters and serves them in one of two modes: fused into a standalone model, or loaded dynamically and selected per request by an `adapter` field. Lets one base model serve several specializations.

### Capability backends
Thin wrappers over the MLX family: `mlx-lm` for text, `mlx-vlm` for vision, `mlx-embeddings` for embeddings and reranking. Each presents a uniform interface to the orchestration layer so the manager treats all model types alike.

### RAG pipeline
Ingestion (load, chunk, embed, upsert), a vector store, and two-stage retrieval (dense vector search followed by cross-encoder rerank). Feeds retrieved, reranked context into a grounded-generation path that returns citations. Detail in `models.md` and `roadmap.md` (Milestone 5).

### Web UI
A Vite and React single-page app, built to static assets and served by the gateway. It is a client of the public API with no privileged path. Capability-aware: it reads `/v1/models` and the active hardware profile and renders only the views the current hardware supports. Detail in `ui.md`.

### Observability
A Prometheus metrics endpoint and a Grafana dashboard. Prometheus and Grafana run as CPU side-services (Docker is acceptable for these). A benchmark harness sweeps model, quantization, context length, and batch size, and emits a report.

## Data flow: a chat request

1. A client posts to `/v1/chat/completions` with a `model` field and messages.
2. The gateway validates and forwards to the scheduler.
3. The model manager ensures the target model is resident, loading or evicting as the ceiling requires.
4. The scheduler admits the request into the current batch; the KV cache reuses any shared prefix.
5. Tokens stream back through the gateway over SSE.
6. Metrics (TTFT, prefill and decode throughput, queue depth, batch size, cache hits) update on `/metrics`.

## The build boundary

Build the orchestration layer and everything above it. Reuse the MLX stack and everything below it.

Do not reimplement: Metal kernels, the array framework, the transformer forward pass, quantization routines. These live in `mlx` and `mlx-lm` and outperform any hand-written replacement on this hardware.

The one place a custom low-level component is justified, and only as an optional depth track, is a custom Metal compute kernel through the MLX custom-kernel API (for example a fused KV-cache update), benchmarked against the stock operation, or treating the KV-cache and scheduler in Milestone 3 as a first-class systems problem. Neither is required for a working platform. A custom C++ CPU kernel is not appropriate here, because CPU inference is the slow path the platform avoids.

## Reference servers (study, do not vendor)

These projects solve adjacent problems. Read their architecture for patterns. Copying them wholesale removes the point of the project.

- `mlx-openai-server` (cubist38): FastAPI, OpenAI-compatible, multi-model YAML, continuous batching, the seed-vs-batching tradeoff. Closest reference for the gateway and config shape.
- `vllm-mlx` (waybarrios): vLLM-style scheduler, paged KV cache, prefix cache, SSD tiering, OpenAI plus Anthropic plus rerank, MCP tool calling. Reference for the scheduler and the multi-surface API.
- `oMLX`: native macOS menu-bar app, two-tier (memory plus SSD) KV cache, model LRU with pin and TTL. Reference for the manager and native packaging.
- `Rapid-MLX`, `vMLX`: speculative decoding, prompt caching, prefix-cache benchmarks. Reference for decode-speed techniques and honest benchmark methodology.
- `mlx-vlm` (Blaizzy): vision feature caching, TurboQuant KV scheme, a unified VLM abstraction. This is the vision dependency.
- `mlx-embeddings` (Blaizzy), `embed-rerank`: embedding and cross-encoder rerank on MLX. These are the retrieval dependencies.
