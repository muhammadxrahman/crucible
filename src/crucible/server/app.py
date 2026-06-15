"""FastAPI gateway: the OpenAI-compatible spine over a single text engine (M1).

Route handlers are sync `def` so FastAPI runs the blocking MLX generation in a
threadpool and the event loop stays free. Multi-model routing, batching, and the
Anthropic and ops surfaces arrive in later milestones.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from crucible.backends import Delta, Final, TextEngine

from . import payloads
from .schemas import ChatCompletionRequest, CompletionRequest

DEFAULT_MAX_TOKENS = 512


def create_app(engine: TextEngine, profile: str) -> FastAPI:
    app = FastAPI(title="Crucible", version="0.0.0")
    app.state.engine = engine
    app.state.profile = profile

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "profile": profile,
            "resident_models": [engine.served_name],
        }

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.served_name,
                    "object": "model",
                    "owned_by": "crucible",
                    "type": "lm",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        miss = _wrong_model(req.model, engine.served_name)
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
        miss = _wrong_model(req.model, engine.served_name)
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

    @app.exception_handler(404)
    async def _not_found(request: Request, exc) -> JSONResponse:  # noqa: ANN001
        return JSONResponse(status_code=404, content=payloads.error("not found", "not_found_error"))

    return app


def _wrong_model(requested: str, served: str) -> JSONResponse | None:
    if requested == served:
        return None
    return JSONResponse(
        status_code=404,
        content=payloads.error(
            f"model '{requested}' not found; this server serves '{served}'",
            "model_not_found",
        ),
    )


def _sse(events: Iterator[dict]) -> Iterator[str]:
    for event in events:
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


# Re-exported for tests that assert event typing.
__all__ = ["create_app", "Delta", "Final"]
