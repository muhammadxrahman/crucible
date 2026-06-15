"""FastAPI gateway: OpenAI-compatible surface over the model manager (M2).

Requests route by the `model` field to a served model; the manager loads, evicts, and
tracks residency against the profile ceiling. Route handlers are sync `def` so FastAPI
runs blocking MLX work in a threadpool. Batching and the Anthropic surface come later.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from crucible.manager import (
    ModelManager,
    ModelStatus,
    ModelTypeUnsupported,
    RuntimeProfile,
    UnknownModel,
)

from . import payloads
from .schemas import ChatCompletionRequest, CompletionRequest

DEFAULT_MAX_TOKENS = 512
_MB = 1024 * 1024


class AdminModelRequest(BaseModel):
    served_name: str
    pinned: bool = True


def create_app(manager: ModelManager, runtime: RuntimeProfile) -> FastAPI:
    app = FastAPI(title="Crucible", version="0.0.0")
    app.state.manager = manager
    app.state.runtime = runtime

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "profile": runtime.name,
            "resident_models": manager.resident_models(),
            "memory_ceiling_gb": runtime.ceiling_gb,
            "resident_gb": round(manager.resident_bytes() / (1024**3), 2),
        }

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
        if req.stream:
            return StreamingResponse(
                _sse(payloads.chat_stream(engine.stream(messages, params), req.model)),
                media_type="text/event-stream",
            )
        return JSONResponse(payloads.chat_full(engine.stream(messages, params), req.model))

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        engine, miss = _resolve(manager, req.model)
        if miss:
            return miss
        params = req.sampling(DEFAULT_MAX_TOKENS)
        messages = [{"role": "user", "content": req.first_prompt()}]
        if req.stream:
            return StreamingResponse(
                _sse(payloads.completion_stream(engine.stream(messages, params), req.model)),
                media_type="text/event-stream",
            )
        return JSONResponse(payloads.completion_full(engine.stream(messages, params), req.model))

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
