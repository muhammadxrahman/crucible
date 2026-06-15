"""Build OpenAI-shaped response payloads from engine events."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

from crucible.backends import Delta, Final, GenEvent


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _now() -> int:
    return int(time.time())


def error(message: str, type_: str) -> dict:
    return {"error": {"message": message, "type": type_}}


def _usage(final: Final) -> dict:
    return {
        "prompt_tokens": final.prompt_tokens,
        "completion_tokens": final.completion_tokens,
        "total_tokens": final.prompt_tokens + final.completion_tokens,
        # Crucible extension: prefill and decode are reported separately, never blended.
        "prefill_tps": round(final.prefill_tps, 2),
        "decode_tps": round(final.decode_tps, 2),
    }


def _drain(events: Iterator[GenEvent]) -> tuple[str, Final]:
    text = ""
    final = Final(prompt_tokens=0, completion_tokens=0, finish_reason="stop")
    for ev in events:
        if isinstance(ev, Delta):
            text += ev.text
        else:
            final = ev
    return text, final


# --- chat ---


def chat_full(events: Iterator[GenEvent], model: str) -> dict:
    text, final = _drain(events)
    return {
        "id": _id("chatcmpl"),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": final.finish_reason,
            }
        ],
        "usage": _usage(final),
    }


def chat_stream(events: Iterator[GenEvent], model: str) -> Iterator[dict]:
    cid, created = _id("chatcmpl"), _now()

    def chunk(delta: dict, finish: str | None) -> dict:
        return {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    yield chunk({"role": "assistant"}, None)
    for ev in events:
        if isinstance(ev, Delta):
            yield chunk({"content": ev.text}, None)
        else:
            yield chunk({}, ev.finish_reason)


# --- legacy completions ---


def completion_full(events: Iterator[GenEvent], model: str) -> dict:
    text, final = _drain(events)
    return {
        "id": _id("cmpl"),
        "object": "text_completion",
        "created": _now(),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": final.finish_reason}],
        "usage": _usage(final),
    }


def completion_stream(events: Iterator[GenEvent], model: str) -> Iterator[dict]:
    cid, created = _id("cmpl"), _now()
    for ev in events:
        if isinstance(ev, Delta):
            yield {
                "id": cid,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": ev.text, "finish_reason": None}],
            }
        else:
            yield {
                "id": cid,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": ev.finish_reason}],
            }
