# Web UI

The web UI is the control plane: it exposes every capability behind one interface so the platform is usable and demoable without curl. It is a client of the public API with no privileged path, and it renders only the views the current hardware supports.

## Stack and serving

- Vite + React, with a single small hand-written stylesheet. Deliberately lean (no Tailwind or component library): the durable contract is the HTTP API, and the UI is intentionally thin and replaceable. The planned end state moves the UI to Swift/SwiftUI bundled as a resource-efficient `.app`; because the React UI only ever calls the public API, a Swift client reuses the same endpoints unchanged.
- Built to `web/dist` and served by the gateway at `/` (mounted last so API routes win), so the platform is one process and one origin (no CORS). The mount is skipped when no build is present, so the API runs without a build.
- During development, run the Vite dev server (`npm --prefix web run dev`), which proxies the API paths to the gateway.

## Layout (M8)

A chat-centric single view, not a multi-tab dashboard. The chat box is the centerpiece (model switcher, streaming, image and document attachments, a Grounded toggle for cited RAG answers). A collapsible side panel handles model load/unload/pin, shows the active profile and a resident-memory bar, and polls live prefill/decode/TTFT from `/metrics/summary`. Images attach inline (routed to the VLM); documents upload to the RAG store via `/rag/upload`.

## Capability-aware rendering

On load, the UI calls `GET /v1/models` and reads the active hardware profile from `GET /healthz`. It then shows only the supported views. On a 16GB profile with `vision: false`, the vision view is hidden or disabled and the model picker shows the smaller default model. This is how the UI degrades on small Macs without separate builds.

## Views

### Chat
Model picker, streaming responses, and per-message readouts for tokens, time-to-first-token, and decode rate. The primary daily-use surface.

### Vision
Drag-and-drop an image, ask a question, see the answer. Hidden when the active profile disables vision.

### RAG
Ingest a folder or files, list indexed documents, ask questions, and read answers with clickable source chunks drawn from the retrieval response. Backed by the `/rag/*` endpoints.

### Models
List available and resident models. Load, unload, and pin. Show a live resident-memory bar against the profile budget, and display the active hardware profile. Backed by `GET /v1/models` and the `/admin/models/*` endpoints.

### Observability
Show time-to-first-token, decode and prefill throughput (as separate readings), queue depth, batch size, and KV-cache hit rate from `/metrics`, or embed a Grafana panel.

### Fine-tune (optional)
Start a LoRA job, watch loss, and fuse or register the resulting adapter. Lands with Milestone 7's capabilities.

## Build order

A thin chat-only version can land right after Milestone 1 for fast feedback. The full capability UI is Milestone 9, after the underlying capabilities exist. See `roadmap.md`.

## Constraints

- The UI calls the same public API as any other client. It must not require or assume a privileged backdoor. Anything the UI does, an API caller can do.
- Round every displayed number to a sensible precision; do not surface raw floats.
- Keep the UI usable on a small window; this is a localhost tool, not a marketing site.
