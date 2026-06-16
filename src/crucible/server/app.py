"""FastAPI gateway: OpenAI-compatible surface over the model manager (M2).

Requests route by the `model` field to a served model; the manager loads, evicts, and
tracks residency against the profile ceiling. Route handlers are sync `def` so FastAPI
runs blocking MLX work in a threadpool. Batching and the Anthropic surface come later.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from collections.abc import Iterator
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from crucible.backends import Delta, Final
from crucible.config import Sampling
from crucible.manager import (
    ModelManager,
    ModelStatus,
    ModelTypeUnsupported,
    RuntimeProfile,
    UnknownModel,
)
from crucible.observability import CONTENT_TYPE, Metrics

from . import payloads
from .schemas import (
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingsRequest,
    RagIngestRequest,
    RagQueryRequest,
    RerankRequest,
)

_MB = 1024 * 1024


@lru_cache(maxsize=1)
def _dashboard_html() -> str:
    return (files("crucible.observability") / "dashboard.html").read_text()


def _observe(events: Iterator, metrics: Metrics, model: str) -> Iterator:
    """Tap an engine stream to record TTFT and the final per-request metrics."""
    start = time.perf_counter()
    first_seen = False
    for ev in events:
        if not first_seen and isinstance(ev, Delta):
            first_seen = True
            metrics.observe_ttft(model, time.perf_counter() - start)
        if isinstance(ev, Final):
            metrics.observe_final(model, ev, time.perf_counter() - start)
        yield ev


class AdminModelRequest(BaseModel):
    served_name: str
    pinned: bool = True


class AddModelRequest(BaseModel):
    path: str  # Hugging Face repo id (already in the local cache)
    type: str = "lm"
    served_name: str
    pin: bool = False


class SessionCreate(BaseModel):
    title: str | None = None
    model: str | None = None


class SessionRename(BaseModel):
    title: str


class MessageCreate(BaseModel):
    role: str
    content: str


def _default_shutdown() -> None:
    """Gracefully stop the server: SIGTERM to ourselves (what Ctrl-C does), after a short
    delay so the HTTP response is flushed first. uvicorn handles the signal cleanly."""

    def _stop() -> None:
        time.sleep(0.4)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_stop, daemon=True).start()


def create_app(
    manager: ModelManager,
    runtime: RuntimeProfile,
    rag=None,
    *,
    web_dist: str = "web/dist",
    sampling: Sampling | None = None,
    on_shutdown=None,
    config_path: str | Path | None = None,
    history=None,
) -> FastAPI:
    app = FastAPI(title="Crucible", version="0.0.0")
    metrics = Metrics()
    sampling_defaults = sampling or Sampling()
    app.state.manager = manager
    app.state.runtime = runtime
    app.state.metrics = metrics
    app.state.rag = rag
    app.state.sampling = sampling_defaults
    if rag is not None:
        rag.metrics = metrics  # so RAG generation updates throughput metrics too

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "profile": runtime.name,
            "resident_models": manager.resident_models(),
            "memory_ceiling_gb": runtime.ceiling_gb,
            "resident_gb": round(manager.resident_bytes() / (1024**3), 2),
        }

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        metrics.collect(manager)
        return Response(content=metrics.expose(), media_type=CONTENT_TYPE)

    @app.get("/metrics/summary")
    def metrics_summary() -> dict:
        metrics.collect(manager)
        s = metrics.summary()
        s["profile"] = runtime.name
        s["ceiling_gb"] = runtime.ceiling_gb
        return s

    @app.get("/observability", response_class=HTMLResponse)
    def observability() -> str:
        return _dashboard_html()

    @app.get("/v1/models")
    def list_models() -> dict:
        return {"object": "list", "data": [_model_view(s) for s in manager.list_status()]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        from crucible.backends.images import extract_images

        raw = [{"role": m.role, "content": m.content} for m in req.messages]
        images = extract_images(raw)
        params = req.sampling(sampling_defaults)

        if images:  # image-bearing requests route to a VLM (M6)
            name, engine, miss = _resolve_vision(manager, req.model)
            if miss:
                return miss
            stream = engine.stream_vision(raw, params, images)
        else:
            engine, miss = _resolve(manager, req.model)
            if miss:
                return miss
            stream = engine.stream(req.rendered_messages(), params)

        events = _observe(stream, metrics, req.model)
        if req.stream:
            return StreamingResponse(
                _sse(payloads.chat_stream(events, req.model)),
                media_type="text/event-stream",
            )
        return JSONResponse(payloads.chat_full(events, req.model))

    @app.post("/v1/messages")
    def anthropic_messages(req: AnthropicMessagesRequest):
        """Anthropic Messages API mapped onto the text backends (text content; image blocks
        ignored). Lets the Anthropic SDK work by changing only the base URL."""
        engine, miss = _resolve(manager, req.model)
        if miss:
            return miss
        params = req.sampling(sampling_defaults)
        events = _observe(engine.stream(req.rendered_messages(), params), metrics, req.model)
        if req.stream:
            return StreamingResponse(
                _anthropic_sse(payloads.messages_stream(events, req.model)),
                media_type="text/event-stream",
            )
        return JSONResponse(payloads.messages_full(events, req.model))

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        engine, miss = _resolve(manager, req.model)
        if miss:
            return miss
        params = req.sampling(sampling_defaults)
        messages = [{"role": "user", "content": req.first_prompt()}]
        events = _observe(engine.stream(messages, params), metrics, req.model)
        if req.stream:
            return StreamingResponse(
                _sse(payloads.completion_stream(events, req.model)),
                media_type="text/event-stream",
            )
        return JSONResponse(payloads.completion_full(events, req.model))

    @app.post("/admin/models/load")
    def admin_load(req: AdminModelRequest):
        return _admin(lambda: manager.load(req.served_name))

    @app.post("/admin/models/unload")
    def admin_unload(req: AdminModelRequest):
        return _admin(lambda: manager.unload(req.served_name))

    @app.post("/admin/models/pin")
    def admin_pin(req: AdminModelRequest):
        return _admin(lambda: manager.pin(req.served_name, req.pinned))

    @app.get("/admin/models/available")
    def admin_available() -> dict:
        """Downloaded MLX models in the local HF cache, for the UI's 'add model' picker."""
        from crucible.manager.catalog import available_models

        registered = {s.path for s in manager.list_status()}
        return {"data": available_models(registered)}

    @app.post("/admin/models/add")
    def admin_add(req: AddModelRequest):
        """Register a downloaded model at runtime, persist it to config, and start loading it.
        Returns immediately with state 'loading'; the UI polls /v1/models until 'resident'."""
        return _admin_add(manager, req, config_path)

    @app.post("/admin/shutdown")
    def admin_shutdown() -> dict:
        """Gracefully stop the server (an alternative to Ctrl-C). Localhost-only."""
        (on_shutdown or _default_shutdown)()
        return {"status": "shutting down"}

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingsRequest):
        engine, miss = _resolve(manager, req.model)
        if miss:
            return miss
        if not hasattr(engine, "embed"):
            return _type_error(req.model, "embedding")
        vectors = engine.embed(req.texts())
        return JSONResponse(
            {
                "object": "list",
                "model": req.model,
                "data": [
                    {"object": "embedding", "index": i, "embedding": v}
                    for i, v in enumerate(vectors)
                ],
            }
        )

    @app.post("/v1/rerank")
    def rerank(req: RerankRequest):
        engine, miss = _resolve(manager, req.model)
        if miss:
            return miss
        if not hasattr(engine, "rerank"):
            return _type_error(req.model, "rerank")
        scores = engine.rerank(req.query, req.documents)
        order = sorted(range(len(req.documents)), key=lambda i: scores[i], reverse=True)
        if req.top_n:
            order = order[: req.top_n]
        return JSONResponse(
            {
                "model": req.model,
                "results": [
                    {"index": i, "relevance_score": scores[i], "document": req.documents[i]}
                    for i in order
                ],
            }
        )

    @app.post("/rag/ingest")
    def rag_ingest(req: RagIngestRequest):
        if rag is None:
            return _rag_disabled()
        return JSONResponse(rag.ingest(req.paths))

    @app.post("/rag/query")
    def rag_query(req: RagQueryRequest):
        if rag is None:
            return _rag_disabled()
        return JSONResponse(
            rag.query(req.query, rerank=req.rerank, top_k=req.top_k, top_n=req.top_n)
        )

    @app.get("/rag/documents")
    def rag_documents():
        if rag is None:
            return _rag_disabled()
        return JSONResponse({"documents": rag.documents()})

    @app.post("/rag/upload")
    def rag_upload(files: list[UploadFile] = File(...)):  # noqa: B008
        # Browsers send file bytes, not server paths: persist them, then ingest.
        if rag is None:
            return _rag_disabled()
        upload_dir = Path(rag.cfg.store_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in files:
            dest = upload_dir / Path(f.filename or "upload").name
            dest.write_bytes(f.file.read())
            paths.append(str(dest))
        return JSONResponse(rag.ingest(paths))

    @app.get("/sessions")
    def list_sessions():
        if history is None:
            return _history_disabled()
        return {"sessions": history.list_sessions()}

    @app.post("/sessions")
    def create_session(req: SessionCreate):
        if history is None:
            return _history_disabled()
        return JSONResponse(history.create_session(title=req.title or "New chat", model=req.model))

    @app.get("/sessions/{sid}")
    def get_session(sid: str):
        if history is None:
            return _history_disabled()
        s = history.get_session(sid)
        if s is None:
            return _session_not_found(sid)
        return JSONResponse(s)

    @app.post("/sessions/{sid}/messages")
    def add_message(sid: str, req: MessageCreate):
        if history is None:
            return _history_disabled()
        if not history.append_message(sid, req.role, req.content):
            return _session_not_found(sid)
        return {"ok": True}

    @app.patch("/sessions/{sid}")
    def rename_session(sid: str, req: SessionRename):
        if history is None:
            return _history_disabled()
        if not history.rename(sid, req.title):
            return _session_not_found(sid)
        return {"ok": True}

    @app.delete("/sessions/{sid}")
    def delete_session(sid: str):
        if history is None:
            return _history_disabled()
        if not history.delete(sid):
            return _session_not_found(sid)
        return {"ok": True}

    app.state.history = history

    # Serve the built web UI at / when present. Mounted last so API routes win.
    if Path(web_dist).is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app


def _resolve_vision(manager: ModelManager, model: str):
    """Resolve an engine for an image request: the named model if it is a VLM, else the
    first registered VLM. Returns (served_name, engine, error_response)."""
    try:
        named_type = manager.entry(model).type
    except UnknownModel:
        named_type = None
    name = model if named_type == "vlm" else manager.first_of_type("vlm")
    if name is None:
        return (
            None,
            None,
            JSONResponse(
                status_code=400,
                content=payloads.error("no vision model is configured", "no_vision_model"),
            ),
        )
    try:
        engine = manager.acquire(name)
    except (UnknownModel, ModelTypeUnsupported) as e:
        return (
            None,
            None,
            JSONResponse(status_code=400, content=payloads.error(str(e), "model_type_error")),
        )
    if not hasattr(engine, "stream_vision"):
        return None, None, _type_error(name, "vision")
    return name, engine, None


def _type_error(model: str, expected: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=payloads.error(f"model '{model}' is not a {expected} model", "model_type_error"),
    )


def _rag_disabled() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=payloads.error(
            "RAG is not configured (no embedding model in the registry)", "rag_unavailable"
        ),
    )


def _history_disabled() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=payloads.error("chat history is not enabled on this server", "history_unavailable"),
    )


def _session_not_found(sid: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=payloads.error(f"session '{sid}' not found", "session_not_found"),
    )


def _model_view(s: ModelStatus) -> dict:
    return {
        "id": s.served_name,
        "object": "model",
        "owned_by": "crucible",
        "type": s.type,
        "path": s.path,
        "state": s.state,
        "pinned": s.pinned,
        "resident_mb": round(s.resident_bytes / _MB, 1),
        "error": s.error,
    }


def _resolve(manager: ModelManager, model: str):
    """Acquire the engine for `model`, or return an OpenAI-shaped error response."""
    try:
        return manager.acquire(model), None
    except UnknownModel:
        return None, JSONResponse(
            status_code=404,
            content=payloads.error(f"model '{model}' not found", "model_not_found"),
        )
    except ModelTypeUnsupported as e:
        return None, JSONResponse(
            status_code=400,
            content=payloads.error(str(e), "model_type_unsupported"),
        )


def _admin(action):
    try:
        return JSONResponse(_status_view(action()))
    except UnknownModel as e:
        return JSONResponse(
            status_code=404,
            content=payloads.error(f"model '{e.args[0]}' not found", "model_not_found"),
        )
    except ModelTypeUnsupported as e:
        return JSONResponse(
            status_code=400, content=payloads.error(str(e), "model_type_unsupported")
        )


def _status_view(s: ModelStatus) -> dict:
    return {
        "served_name": s.served_name,
        "type": s.type,
        "path": s.path,
        "state": s.state,
        "pinned": s.pinned,
        "resident_mb": round(s.resident_bytes / _MB, 1),
        "error": s.error,
    }


def _admin_add(manager: ModelManager, req: AddModelRequest, config_path) -> JSONResponse:
    from pydantic import ValidationError

    from crucible.config import ModelEntry
    from crucible.config.store import append_model

    try:
        entry = ModelEntry(path=req.path, type=req.type, served_name=req.served_name, pin=req.pin)
    except ValidationError as e:
        return JSONResponse(status_code=400, content=payloads.error(str(e), "invalid_model_entry"))
    try:
        manager.register(entry)
    except ValueError as e:
        return JSONResponse(status_code=409, content=payloads.error(str(e), "served_name_conflict"))
    if config_path is not None:
        try:
            append_model(config_path, entry)
        except Exception:  # noqa: BLE001 - persistence is best-effort; the model still serves now
            pass
    return JSONResponse(_status_view(manager.load_async(req.served_name)))


def _sse(events: Iterator[dict]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


def _anthropic_sse(events: Iterator[dict]) -> Iterator[str]:
    """Anthropic SSE: a named `event:` line per chunk, ended by `message_stop` (no [DONE])."""
    for event in events:
        yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


__all__ = ["create_app"]
