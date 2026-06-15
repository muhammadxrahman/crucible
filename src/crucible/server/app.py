"""FastAPI gateway: OpenAI-compatible surface over the model manager (M2).

Requests route by the `model` field to a served model; the manager loads, evicts, and
tracks residency against the profile ceiling. Route handlers are sync `def` so FastAPI
runs blocking MLX work in a threadpool. Batching and the Anthropic surface come later.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from functools import lru_cache
from importlib.resources import files

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from crucible.backends import Delta, Final
from crucible.manager import (
    ModelManager,
    ModelStatus,
    ModelTypeUnsupported,
    RuntimeProfile,
    UnknownModel,
)
from crucible.observability import CONTENT_TYPE, Metrics

from . import payloads
from .schemas import ChatCompletionRequest, CompletionRequest

DEFAULT_MAX_TOKENS = 512
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


def create_app(manager: ModelManager, runtime: RuntimeProfile) -> FastAPI:
    app = FastAPI(title="Crucible", version="0.0.0")
    metrics = Metrics()
    app.state.manager = manager
    app.state.runtime = runtime
    app.state.metrics = metrics

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
        engine, miss = _resolve(manager, req.model)
        if miss:
            return miss
        params = req.sampling(DEFAULT_MAX_TOKENS)
        messages = req.rendered_messages()
        events = _observe(engine.stream(messages, params), metrics, req.model)
        if req.stream:
            return StreamingResponse(
                _sse(payloads.chat_stream(events, req.model)),
                media_type="text/event-stream",
            )
        return JSONResponse(payloads.chat_full(events, req.model))

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        engine, miss = _resolve(manager, req.model)
        if miss:
            return miss
        params = req.sampling(DEFAULT_MAX_TOKENS)
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

    return app


def _model_view(s: ModelStatus) -> dict:
    return {
        "id": s.served_name,
        "object": "model",
        "owned_by": "crucible",
        "type": s.type,
        "state": s.state,
        "pinned": s.pinned,
        "resident_mb": round(s.resident_bytes / _MB, 1),
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
        "state": s.state,
        "pinned": s.pinned,
        "resident_mb": round(s.resident_bytes / _MB, 1),
    }


def _sse(events: Iterator[dict]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


__all__ = ["create_app"]
