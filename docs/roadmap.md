# Roadmap

Build milestones in order. Each is independently demoable. Do not start a milestone before the previous one passes its acceptance criteria. Update the current-milestone line in `CLAUDE.md` as work proceeds.

Options and tradeoffs for each milestone are condensed here; the personal design notes carry the longer rationale. Note that milestones are subject to change as new practical findings and limitations are discovered.

## M0: foundations and environment
Goal: text generation from one model through the CLI, plus the repo skeleton.

Build:
- Python env with `uv`. Install `mlx`, `mlx-lm`.
- Verify Metal is active and the GPU is visible to MLX.
- Pull one MoE model from `mlx-community`; confirm with `mlx_lm.generate`.
- Repo scaffold: `src/`, `config/`, `tests/`, `benchmarks/`, `web/`, `CLAUDE.md`, `docs/`, `.claude/`, `pyproject.toml`.
- Config loader with a Pydantic schema for the registry and profiles.
- Hardware detection: read `hw.memsize`, compute the model budget, select a default profile.

Acceptance: `mlx_lm.generate` produces tokens; `pytest` runs; config validates and rejects a malformed registry; `mlxd profile` prints detected memory and the chosen profile.

## M1: text inference server (the spine)
Goal: an OpenAI-compatible `/v1/chat/completions` with streaming, serving one model.

Build:
- FastAPI app, async.
- `/v1/chat/completions` and `/v1/completions` with SSE streaming.
- Map OpenAI fields onto `mlx-lm` generation (`stream_generate`, sampling params, stop sequences).
- Load the configured model at startup. `/healthz`.

Acceptance: a streamed chat completion over curl; the official `openai` Python client works by changing only `base_url`; Claude Code or Cursor pointed at the server holds a real conversation.

Tradeoff: build a thin FastAPI layer on `mlx-lm` primitives rather than wrapping `mlx_lm.server` (which constrains M3) or forking a full server (which skips the learning).

## M2: model manager and multi-model serving
Goal: serve several models from one process, switch without restart, evict under memory pressure.

Build:
- Load, unload, and track models with resident memory.
- LRU eviction, manual pin and unpin, per-model TTL.
- Memory accounting against the profile ceiling, using MLX introspection.
- Profile-aware budgeting: ceiling and single-vs-multi-resident come from the active profile.
- `/v1/models`; route requests by `model`.

Acceptance: register a MoE, a small dense, and an embedding model; serve all three; switch mid-session; observe LRU eviction when a load would exceed the ceiling; confirm single-resident behavior under a small profile.

## M3: continuous batching and KV-cache management (systems core)
Goal: concurrent requests share the GPU; throughput scales with load.

Build:
- A request scheduler with an admission queue.
- In-flight batching using the `mlx-lm` batch primitive (`BatchGenerator`).
- A prefix or paged KV cache so shared prefixes skip recomputation.
- Optional 4-bit KV cache for long contexts.
- Optional depth track: a custom Metal kernel for a hot op, benchmarked against stock.

Acceptance: aggregate throughput rises with N concurrent requests up to a measured saturation point; a repeated prefix shows a measurable prefill saving; memory stays bounded under the cap.

Tradeoff: start with prefix caching (simple, large wins for shared system prompts and RAG); add paging only under heavy multi-tenant load. Batched mode shares one RNG lane, so expose a non-batched path when reproducible sampling is required.

## M4: observability and benchmarking
Goal: a live metrics dashboard and a repeatable benchmark harness.

Build:
- Prometheus `/metrics`: prefill and decode throughput (separate), TTFT, queue depth, batch size, cache hit rate, resident memory, evictions, per-model counts and latencies.
- An in-app dashboard served at `/observability` (reads `/metrics/summary`, keeps a short in-memory history for sparklines). No external daemons, since the end goal is a self-contained installable app. External Prometheus and Grafana are an optional add-on the standard `/metrics` endpoint enables, not part of the default install.
- A benchmark harness sweeping model, quantization, context length, and batch size, emitting a Markdown report with charts.

Acceptance: the in-app dashboard shows live traffic with no external services; the harness produces a reproducible report comparing a MoE against a dense model, with the bandwidth-ceiling effect visible.

## M5: RAG layer
Goal: ingest local documents and answer with grounded citations. No external calls.

Build:
- Embedding service via `mlx-embeddings` (`/v1/embeddings`).
- A vector store and an ingestion pipeline (load, chunk, embed, upsert).
- Two-stage retrieval: dense search then cross-encoder rerank (`/v1/rerank`).
- A grounded-generation path returning citations. `/rag/*` endpoints.

Acceptance: ingest a folder of PDFs and notes; ask a question; get an answer grounded in the right chunks with sources; confirm no external network calls.

Tradeoff: the store is an in-process NumPy brute-force index (zero dependencies, zero daemons), which fits the self-contained-app goal and is fast for local document counts. It sits behind a small interface, so LanceDB (embedded ANN) or Qdrant (server-side filtering, multi-client) can drop in later if scale needs it. Keep rerank toggleable to quantify its lift.

## M6: vision and multimodal
Goal: accept image-plus-text requests and answer about the image.

Build:
- `mlx-vlm` behind the model manager.
- Accept the OpenAI vision message shape (`image_url`, including base64).
- Vision feature caching across turns.
- Route image-bearing requests to a VLM, text-only to the text model.

Acceptance: post an image and a question, get an accurate answer; run a document-QA or screenshot-to-structure demo; multi-turn over one image reuses cached vision features (visible in metrics).

Tradeoff: VLM batching is more limited than text (prefix caching for VLMs is constrained), so set conservative VLM concurrency.

## M7: LoRA fine-tuning and adapter serving
Goal: fine-tune on local data and serve the adapter.

Build:
- A training pipeline on `mlx-lm` LoRA (LoRA, DoRA, QLoRA, full).
- Dataset prep to JSONL, an eval split, an eval step.
- Two serving modes: fuse (`mlx_lm.fuse`) or dynamic per-request adapter routing.

Acceptance: fine-tune a small model on a focused dataset; serve the adapter; A/B it against the base on held-out examples with numbers.

Tradeoff: LoRA for most cases; QLoRA when the base is large and memory is tight; DoRA for a quality bump; full only for small models. Build fuse first; add dynamic routing for the multi-tenant story.

## M8: packaging and deploy
Goal: one command up, survives reboot, documented.

Build:
- A `launchd` plist so the engine runs native on login (not Docker).
- A CLI (`mlxd serve`, `mlxd models`, `mlxd bench`, `mlxd profile`).
- Config validation on boot with clear errors.
- Docker Compose for CPU side-services only (Prometheus, Grafana, optional Qdrant), with a README note that the engine is host-native by design.

Acceptance: a fresh login brings the server up; the CLI works; the README takes a new reader from clone to first cited answer in under 15 minutes.

## M9: web UI (control plane and demo surface)
Goal: a local web app exposing every capability behind one interface.

Build:
- A Vite and React SPA served by the gateway at `/`, talking only to the public API.
- Views: chat (with model picker and per-message readouts), vision, RAG (ingest plus cited answers), models (load/unload/pin plus live memory bar plus active profile), observability, and an optional fine-tune view.
- Capability-aware rendering from `/v1/models` and the active profile.

Acceptance: a full demo runs in the browser (pick a model, chat, drop an image, ingest a doc and get a cited answer, watch memory and throughput live); on a 16GB profile the UI shows the smaller model and hides the vision view.

Note: a thin chat-only UI may land right after M1 for fast feedback; the full capability UI belongs here.

## Definition of done

- One config-driven server serves text, vision, embedding, and reranker models, switching and evicting under the active profile budget.
- Concurrent requests batch and share a KV cache, with throughput and cache metrics on a live dashboard.
- A RAG pipeline answers over local documents with citations and no external calls.
- A fine-tuned LoRA adapter is served and benchmarked against its base.
- A reproducible benchmark report shows MoE decode speed, the dense bandwidth ceiling, and Neural-Accelerator prefill gains.
- A single web UI exposes chat, vision, RAG, model management, and live metrics, and degrades gracefully on smaller Macs.
- The same build runs on a 16GB Mac by auto-selecting a smaller-model profile.
- A new reader reaches a first cited answer in under 15 minutes from the README.
