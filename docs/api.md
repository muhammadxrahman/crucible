# API contract

The gateway exposes an OpenAI-compatible surface, an Anthropic messages surface, and ops endpoints. Compatibility is the goal: existing OpenAI and Anthropic client libraries must work by changing only the base URL. Default base: `http://127.0.0.1:8000`.

## OpenAI-compatible endpoints

### POST /v1/chat/completions
Chat with a text or vision model. Honors `model`, `messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `stop`, and tool-calling fields. Streaming uses SSE with `data:` chunks terminated by `data: [DONE]`. Vision input uses the standard content-parts shape with `image_url` items (HTTP URLs and base64 data URLs both accepted).

### POST /v1/completions
Legacy text completion for a single prompt. Same sampling fields, same streaming behavior.

### POST /v1/embeddings
Returns dense vectors for `input` (a string or array of strings), served by an `embedding` model. Response matches the OpenAI embeddings shape.

### POST /v1/rerank
Reorders `documents` by relevance to `query`, served by a `rerank` model (cross-encoder). Returns documents with scores, sorted descending.

### GET /v1/models
Lists models. Each entry reports its `served_name`, `type`, residency state (resident or available), `pin` state, and approximate resident memory. Clients and the UI use this to discover capabilities.

## Anthropic-compatible endpoint

### POST /v1/messages
Maps the Anthropic messages format onto the same backends. Honors `model`, `messages`, `system`, `max_tokens`, `stream`, and tool use. Enables Anthropic SDK clients without code changes beyond the base URL.

## Ops endpoints

### GET /metrics
Prometheus exposition format. Series include: prefill throughput, decode throughput, time-to-first-token, queue depth, active batch size, KV-cache hit rate, resident memory, eviction count, and per-model request counts and latencies. Prefill and decode are always separate series.

### GET /healthz
Liveness and readiness. Reports server status, the active hardware profile, and resident models.

## Control-plane endpoints

These back the web UI's model and RAG views. They are part of the public API and hold no special privilege beyond ordinary authorization.

### POST /admin/models/load, POST /admin/models/unload, POST /admin/models/pin
Load, unload, or pin a model by `served_name`. Subject to the active profile's memory ceiling and residency rules.

### POST /rag/ingest
Ingests files or a directory: load, chunk, embed, upsert into the vector store. Returns indexed document identifiers.

### POST /rag/query
Runs two-stage retrieval (dense search then rerank) and grounded generation. Returns the answer plus the source chunks used, so the UI can render citations.

### GET /rag/documents
Lists indexed documents.

## Conventions

- Errors follow the OpenAI error envelope (`error.type`, `error.message`) for compatibility.
- Streaming defaults on for chat where the client requests it.
- All endpoints bind to `127.0.0.1` by default. Exposing on a network requires adding authentication first.
- Request routing is by the `model` field against `served_name`. An unknown model returns a clear 404-class error rather than silently falling back.

This file is the contract. When an endpoint's behavior changes, update this file in the same change.
