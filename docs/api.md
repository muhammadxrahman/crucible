# API contract

The gateway exposes an OpenAI-compatible surface, an Anthropic messages surface, and ops endpoints. Compatibility is the goal: existing OpenAI and Anthropic client libraries must work by changing only the base URL. Default base: `http://127.0.0.1:8000`.

## OpenAI-compatible endpoints

### POST /v1/chat/completions
Chat with a text or vision model. Honors `model`, `messages`, `stream`, `temperature`, `top_p`, `repetition_penalty`, `enable_thinking`, `max_tokens`, `stop`, and tool-calling fields. Any omitted sampling field falls back to the server `sampling` defaults (chat-sane, with a repetition penalty so generation does not collapse into loops). `enable_thinking` is off by default, so reasoning models (Qwen3) answer directly rather than emitting a `<think>` block; set it true per request to opt into reasoning. `max_tokens` defaults to unlimited (the server `sampling.max_tokens` is `0`, meaning run until the model stops); send a positive `max_tokens` to cap the output. Streaming uses SSE with `data:` chunks terminated by `data: [DONE]`. Vision input uses the standard content-parts shape with `image_url` items (HTTP URLs and base64 data URLs both accepted).

### POST /v1/completions
Legacy text completion for a single prompt. Same sampling fields, same streaming behavior.

### POST /v1/embeddings
Returns dense vectors for `input` (a string or array of strings), served by an `embedding` model. Response matches the OpenAI embeddings shape.

### POST /v1/rerank
Reorders `documents` by relevance to `query`, served by a `rerank` model (cross-encoder). Returns documents with scores, sorted descending.

### GET /v1/models
Lists models. Each entry reports its `id` (served_name), `type`, the underlying model `path` (the Hugging Face repo it serves, so the UI shows the real model rather than only the served_name), residency `state` (`resident`, `loading`, or `available`), `pin` state, approximate resident memory, and an `error` string if the last load failed. Clients and the UI use this to discover capabilities.

## Anthropic-compatible endpoint

### POST /v1/messages
Maps the Anthropic Messages format onto the same text backends, so the Anthropic SDK works by changing only the base URL. Honors `model`, `messages`, `system`, `max_tokens` (required, as Anthropic mandates), `stream`, `temperature`, `top_p`, and `stop_sequences`. Text content is supported; image blocks and tool use are not mapped yet. Non-streaming returns a `message` object (`content: [{type:"text",...}]`, `stop_reason` mapped to `end_turn`/`max_tokens`, `usage.input_tokens`/`output_tokens`). Streaming emits Anthropic's named SSE events: `message_start`, `content_block_start`, `content_block_delta` (`text_delta`), `content_block_stop`, `message_delta`, `message_stop`.

## Ops endpoints

### GET /metrics
Prometheus exposition format. Series include: prefill throughput, decode throughput, time-to-first-token, queue depth, active batch size, KV-cache hit rate, resident memory, eviction count, and per-model request counts and latencies. Prefill and decode are always separate series. This standard endpoint lets an external Prometheus scrape the server, but it is optional: the server needs no external monitoring stack.

### GET /metrics/summary
JSON snapshot of current metric values plus a short rolling history, for the in-app observability dashboard. Includes the active profile and memory ceiling.

### GET /observability
Serves the built-in dashboard (a self-contained page that polls `/metrics/summary`). The default observability surface, requiring no Prometheus or Grafana.

### GET /healthz
Liveness and readiness. Reports server status, the active hardware profile, and resident models.

## Control-plane endpoints

These back the web UI's model and RAG views. They are part of the public API and hold no special privilege beyond ordinary authorization.

### POST /admin/models/load, POST /admin/models/unload, POST /admin/models/pin
Load, unload, or pin a model by `served_name`. Subject to the active profile's memory ceiling and residency rules. Loads run with the manager lock released, so concurrent calls (and `/v1/models` polling) stay responsive while a large model loads.

### GET /admin/models/available
Lists MLX models already present in the local Hugging Face cache (no download): each item reports `repo_id`, `size_bytes`, `size_str`, a best-effort `guessed_type`, and whether it is already `registered`. Backs the web UI's "add model" picker.

### POST /admin/models/add
Registers a downloaded model at runtime (`path`, `type`, `served_name`, optional `pin`), appends it to the active config file so it survives a restart (comments preserved), and starts loading it in the background. Returns immediately with `state: "loading"`; poll `/v1/models` until it becomes `resident`. A duplicate `served_name` returns a `409` `served_name_conflict`; an invalid entry returns `400`.

### POST /admin/shutdown
Gracefully stops the server (the same clean shutdown as Ctrl-C), an alternative for clients without terminal access such as the web UI's shutdown button. Localhost-only. Has no effect against the optional login service, which restarts on exit.

### POST /rag/ingest
Ingests files or a directory by server-side path: load, chunk, embed, upsert into the vector store. Returns indexed document identifiers.

### POST /rag/upload
Multipart file upload for browser clients (which cannot supply a server path). Persists the uploaded files under the RAG store and ingests them. Same response shape as `/rag/ingest`.

### POST /rag/query
Runs two-stage retrieval (dense search then rerank) and grounded generation. Returns the answer plus the source chunks used, so the UI can render citations.

### GET /rag/documents
Lists indexed documents.

### Sessions (chat history)
Back the web app's saved-conversation list. Enabled when the server is started with a history store (the default for `mlxd serve`); when disabled they return `503 history_unavailable`. The store is local SQLite (`.crucible/history.db`, overridable with `CRUCIBLE_HISTORY_DB`).

- `GET /sessions` — list sessions (most-recently-updated first) with `id`, `title`, `model`, timestamps, and `messages_count`.
- `POST /sessions` — create a session (`{title?, model?}`); returns the new session.
- `GET /sessions/{id}` — the session plus its ordered `messages` (`{role, content, created_at}`); `404 session_not_found` if unknown.
- `POST /sessions/{id}/messages` — append a message (`{role, content}`).
- `PATCH /sessions/{id}` — rename (`{title}`).
- `DELETE /sessions/{id}` — delete the session and its messages.

### Streaming and cancellation
Streaming responses (`/v1/chat/completions`, `/v1/completions`, `/v1/messages`) stop generating when the client disconnects: the request is dropped from the running batch and its KV slot freed, instead of generating to EOS for output no one will read. The web app's send button becomes a stop button mid-stream.

## Conventions

- Errors follow the OpenAI error envelope (`error.type`, `error.message`) for compatibility.
- Streaming defaults on for chat where the client requests it.
- All endpoints bind to `127.0.0.1` by default. Exposing on a network requires adding authentication first.
- Request routing is by the `model` field against `served_name`. An unknown model returns a clear 404-class error rather than silently falling back.
- Image-bearing chat requests (content parts with `image_url`) route to a VLM automatically; text-only requests use the text model.
- The built web UI is served at `/` when a build is present (see `ui.md`). It is a client of these endpoints with no privileged path.

This file is the contract. When an endpoint's behavior changes, update this file in the same change.
