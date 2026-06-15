"""M1 acceptance: the OpenAI-compatible HTTP contract.

These run against a fake engine via FastAPI's TestClient, so they are fast and need no
model download or GPU. Real-model curl/openai-client checks live in
scripts/smoke_server.py.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from crucible.backends import Delta, Final, GenEvent, SamplingParams
from crucible.server import create_app


class FakeEngine:
    """Deterministic engine that echoes a fixed token stream."""

    served_name = "primary"
    model_path = "fake/tiny"

    def __init__(self, pieces: list[str] | None = None):
        self.pieces = pieces if pieces is not None else ["Hello", ", ", "world", "!"]
        self.last_params: SamplingParams | None = None
        self.last_messages: list[dict] | None = None

    def stream(self, messages: list[dict], params: SamplingParams) -> Iterator[GenEvent]:
        self.last_messages = messages
        self.last_params = params
        for p in self.pieces:
            yield Delta(p)
        yield Final(
            prompt_tokens=7,
            completion_tokens=len(self.pieces),
            finish_reason="stop",
            prefill_tps=120.0,
            decode_tps=85.5,
        )


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(FakeEngine(), profile="pro64"))


def test_healthz_reports_profile_and_model(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["profile"] == "pro64"
    assert body["resident_models"] == ["primary"]


def test_list_models(client: TestClient) -> None:
    data = client.get("/v1/models").json()["data"]
    assert data[0]["id"] == "primary"
    assert data[0]["type"] == "lm"


def test_chat_non_streaming(client: TestClient) -> None:
    r = client.post(
        "/v1/chat/completions",
        json={"model": "primary", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello, world!"
    assert body["choices"][0]["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    # prefill and decode reported separately
    assert usage["prefill_tps"] == 120.0
    assert usage["decode_tps"] == 85.5


def test_chat_streaming_sse(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "primary", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        chunks = [line for line in r.iter_lines() if line.startswith("data: ")]

    payloads = [c[len("data: ") :] for c in chunks]
    assert payloads[-1] == "[DONE]"
    parsed = [json.loads(p) for p in payloads[:-1]]
    assert parsed[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in parsed)
    assert text == "Hello, world!"
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


def test_unknown_model_returns_openai_error(client: TestClient) -> None:
    r = client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["type"] == "model_not_found"
    assert "nope" in err["message"]


def test_completions_legacy(client: TestClient) -> None:
    r = client.post("/v1/completions", json={"model": "primary", "prompt": "once"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == "Hello, world!"


def test_sampling_and_stop_are_forwarded() -> None:
    engine = FakeEngine()
    c = TestClient(create_app(engine, profile="pro64"))
    c.post(
        "/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 64,
            "stop": ["\n\n", "END"],
        },
    )
    assert engine.last_params.temperature == 0.2
    assert engine.last_params.top_p == 0.9
    assert engine.last_params.max_tokens == 64
    assert engine.last_params.stop == ["\n\n", "END"]


def test_vision_parts_flatten_to_text() -> None:
    engine = FakeEngine()
    c = TestClient(create_app(engine, profile="pro64"))
    c.post(
        "/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                    ],
                }
            ],
        },
    )
    # image parts dropped until M6; text preserved
    assert engine.last_messages[0]["content"] == "describe"
