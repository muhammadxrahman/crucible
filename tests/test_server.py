"""M1 acceptance: the OpenAI-compatible HTTP contract.

These run against a fake engine via FastAPI's TestClient, so they are fast and need no
model download or GPU. Real-model curl/openai-client checks live in
scripts/smoke_server.py.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from crucible.backends import Delta, Final, GenEvent, SamplingParams
from crucible.config import Registry
from crucible.manager import ModelManager, RuntimeProfile
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


def make_client(pieces: list[str] | None = None) -> tuple[TestClient, FakeEngine]:
    reg = Registry.model_validate(
        {"models": [{"path": "fake/tiny", "type": "lm", "served_name": "primary", "pin": True}]}
    )
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=10**12,
        single_resident=False,
        default_context=32768,
        kv_bits=8,
        vision=True,
    )
    engine = FakeEngine(pieces)
    manager = ModelManager(reg, runtime, lambda entry: (engine, 1000))
    manager.warmup()
    return TestClient(create_app(manager, runtime)), engine


@pytest.fixture
def client() -> TestClient:
    c, _ = make_client()
    return c


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
    c, engine = make_client()
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


def test_sampling_defaults_applied_when_omitted() -> None:
    # A request that omits sampling fields gets the chat-sane server defaults, including the
    # repetition penalty that prevents degenerate loops on any model.
    c, engine = make_client()
    c.post(
        "/v1/chat/completions",
        json={"model": "primary", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert engine.last_params.temperature == 0.7
    assert engine.last_params.top_p == 0.95
    assert engine.last_params.repetition_penalty == 1.1
    assert engine.last_params.max_tokens == 0  # unlimited unless the client caps it


def test_request_overrides_sampling_defaults() -> None:
    c, engine = make_client()
    c.post(
        "/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
            "repetition_penalty": 1.3,
        },
    )
    assert engine.last_params.temperature == 0.0
    assert engine.last_params.repetition_penalty == 1.3


def test_shutdown_endpoint_triggers_shutdown_and_responds() -> None:
    # The injected callback avoids actually signalling the test process.
    called = []
    reg = Registry.model_validate(
        {"models": [{"path": "f/p", "type": "lm", "served_name": "primary"}]}
    )
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=10**12,
        single_resident=False,
        default_context=8192,
        kv_bits=8,
        vision=True,
    )
    manager = ModelManager(reg, runtime, lambda e: (FakeEngine(), 1))
    c = TestClient(create_app(manager, runtime, on_shutdown=lambda: called.append(True)))
    r = c.post("/admin/shutdown")
    assert r.status_code == 200
    assert r.json()["status"] == "shutting down"
    assert called == [True]


def test_thinking_disabled_by_default_and_overridable() -> None:
    # Reasoning models (Qwen3) otherwise spend the whole budget in <think> and never answer.
    c, engine = make_client()
    c.post(
        "/v1/chat/completions",
        json={"model": "primary", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert engine.last_params.enable_thinking is False
    c.post(
        "/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": True,
        },
    )
    assert engine.last_params.enable_thinking is True


def test_loop_guard_default_on_and_overridable() -> None:
    c, engine = make_client()
    c.post(
        "/v1/chat/completions",
        json={"model": "primary", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert engine.last_params.loop_guard is True
    c.post(
        "/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [{"role": "user", "content": "hi"}],
            "loop_guard": False,
        },
    )
    assert engine.last_params.loop_guard is False


def make_multi_client() -> tuple[TestClient, dict[str, FakeEngine]]:
    reg = Registry.model_validate(
        {
            "models": [
                {"path": "f/a", "type": "lm", "served_name": "a"},
                {"path": "f/b", "type": "lm", "served_name": "b"},
                {"path": "f/e", "type": "embedding", "served_name": "embed"},
            ]
        }
    )
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=10**12,
        single_resident=False,
        default_context=32768,
        kv_bits=8,
        vision=True,
    )
    from crucible.manager import ModelTypeUnsupported

    engines: dict[str, FakeEngine] = {}

    def loader(entry):  # noqa: ANN001
        if entry.type != "lm":
            raise ModelTypeUnsupported(f"serving '{entry.type}' arrives in M5")
        e = FakeEngine([entry.served_name])  # echoes its own name
        engines[entry.served_name] = e
        return e, 1000

    manager = ModelManager(reg, runtime, loader)
    return TestClient(create_app(manager, runtime)), engines


def test_routing_selects_model_by_field() -> None:
    c, _ = make_multi_client()
    ra = c.post("/v1/chat/completions", json={"model": "a", "messages": []}).json()
    rb = c.post("/v1/chat/completions", json={"model": "b", "messages": []}).json()
    assert ra["choices"][0]["message"]["content"] == "a"
    assert rb["choices"][0]["message"]["content"] == "b"


def test_admin_load_unload_pin_over_http() -> None:
    c, _ = make_multi_client()
    assert c.post("/admin/models/load", json={"served_name": "a"}).json()["state"] == "resident"

    states = {m["id"]: m["state"] for m in c.get("/v1/models").json()["data"]}
    assert states["a"] == "resident"

    assert c.post("/admin/models/unload", json={"served_name": "a"}).json()["state"] == "available"

    pinned = c.post("/admin/models/pin", json={"served_name": "b", "pinned": True}).json()
    assert pinned["pinned"] is True and pinned["state"] == "resident"

    miss = c.post("/admin/models/load", json={"served_name": "ghost"})
    assert miss.status_code == 404 and miss.json()["error"]["type"] == "model_not_found"


def test_unsupported_model_type_returns_400() -> None:
    c, _ = make_multi_client()
    r = c.post("/v1/chat/completions", json={"model": "embed", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "model_type_unsupported"


def test_healthz_reports_ceiling_and_resident_gb() -> None:
    c, _ = make_multi_client()
    body = c.get("/healthz").json()
    assert "memory_ceiling_gb" in body and "resident_gb" in body


def test_metrics_endpoint_exposes_series_after_traffic() -> None:
    c, _ = make_multi_client()
    c.post("/v1/chat/completions", json={"model": "a", "messages": []})
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "crucible_requests_total" in body
    assert "crucible_decode_tps" in body
    assert "crucible_resident_bytes" in body


def test_metrics_summary_shape() -> None:
    c, _ = make_multi_client()
    c.post("/v1/chat/completions", json={"model": "a", "messages": []})
    s = c.get("/metrics/summary").json()
    assert {"current", "history", "per_model"} <= set(s)
    assert s["profile"] == "pro64"
    assert "ceiling_gb" in s
    assert s["current"]["requests_total"] >= 1


def test_observability_serves_dashboard_html() -> None:
    c, _ = make_multi_client()
    r = c.get("/observability")
    assert r.status_code == 200
    assert "<title>Crucible" in r.text


def make_history_client() -> TestClient:
    from crucible.history import HistoryStore

    reg = Registry.model_validate(
        {"models": [{"path": "fake/tiny", "type": "lm", "served_name": "primary", "pin": True}]}
    )
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=10**12,
        single_resident=False,
        default_context=32768,
        kv_bits=8,
        vision=True,
    )
    manager = ModelManager(reg, runtime, lambda entry: (FakeEngine(), 1000))
    manager.warmup()
    return TestClient(create_app(manager, runtime, history=HistoryStore(":memory:")))


def test_session_lifecycle_over_http() -> None:
    c = make_history_client()
    sid = c.post("/sessions", json={"title": "My chat", "model": "primary"}).json()["id"]

    c.post("/sessions/" + sid + "/messages", json={"role": "user", "content": "hello"})
    c.post("/sessions/" + sid + "/messages", json={"role": "assistant", "content": "hi"})

    sessions = c.get("/sessions").json()["sessions"]
    assert sessions[0]["id"] == sid and sessions[0]["messages_count"] == 2

    full = c.get("/sessions/" + sid).json()
    assert [m["content"] for m in full["messages"]] == ["hello", "hi"]

    assert c.patch("/sessions/" + sid, json={"title": "Renamed"}).status_code == 200
    assert c.get("/sessions").json()["sessions"][0]["title"] == "Renamed"

    assert c.delete("/sessions/" + sid).status_code == 200
    assert c.get("/sessions").json()["sessions"] == []


def test_session_not_found_is_404() -> None:
    c = make_history_client()
    r = c.get("/sessions/ghost")
    assert r.status_code == 404 and r.json()["error"]["type"] == "session_not_found"


def test_history_disabled_returns_503_by_default() -> None:
    c, _ = make_client()  # no history store wired
    r = c.get("/sessions")
    assert r.status_code == 503 and r.json()["error"]["type"] == "history_unavailable"


def test_anthropic_messages_non_streaming() -> None:
    c, engine = make_client()
    r = c.post(
        "/v1/messages",
        json={
            "model": "primary",
            "max_tokens": 64,
            "system": "You are terse.",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["content"][0] == {"type": "text", "text": "Hello, world!"}
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["output_tokens"] == 4
    # the `system` field is threaded in as a leading system message
    assert engine.last_messages[0] == {"role": "system", "content": "You are terse."}


def test_anthropic_messages_streaming_event_sequence() -> None:
    c, _ = make_client()
    with c.stream(
        "POST",
        "/v1/messages",
        json={
            "model": "primary",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    ) as r:
        assert r.status_code == 200
        lines = [ln for ln in r.iter_lines()]

    events = [ln[len("event: ") :] for ln in lines if ln.startswith("event: ")]
    assert events[0] == "message_start"
    assert "content_block_delta" in events
    assert events[-1] == "message_stop"
    text = ""
    for ln in lines:
        if ln.startswith("data: "):
            payload = json.loads(ln[len("data: ") :])
            if payload["type"] == "content_block_delta":
                text += payload["delta"]["text"]
    assert text == "Hello, world!"


def test_anthropic_max_tokens_required() -> None:
    c, _ = make_client()
    r = c.post("/v1/messages", json={"model": "primary", "messages": []})
    assert r.status_code == 422  # Anthropic requires max_tokens


def test_list_models_reports_real_path() -> None:
    c, _ = make_client()
    data = c.get("/v1/models").json()["data"]
    assert data[0]["id"] == "primary"
    assert data[0]["path"] == "fake/tiny"  # the underlying model, not just the served_name


def test_available_models_endpoint(monkeypatch) -> None:
    from crucible.manager import catalog

    monkeypatch.setattr(
        catalog,
        "available_models",
        lambda registered: [
            {
                "repo_id": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "size_bytes": 1_800_000_000,
                "size_str": "1.8G",
                "guessed_type": "lm",
                "registered": False,
            }
        ],
    )
    c, _ = make_client()
    data = c.get("/admin/models/available").json()["data"]
    assert data[0]["repo_id"] == "mlx-community/Llama-3.2-3B-Instruct-4bit"
    assert data[0]["guessed_type"] == "lm"


def _add_client(tmp_path):
    """A client whose loader serves any lm entry and whose adds persist to a temp config."""
    cfg = tmp_path / "models.yaml"
    cfg.write_text("models:\n  - {path: fake/tiny, type: lm, served_name: primary}\n")
    from crucible.config import load_registry

    reg = load_registry(cfg)
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=10**12,
        single_resident=False,
        default_context=32768,
        kv_bits=8,
        vision=True,
    )
    manager = ModelManager(reg, runtime, lambda entry: (FakeEngine([entry.served_name]), 1000))
    return TestClient(create_app(manager, runtime, config_path=cfg)), cfg, manager


def test_add_model_registers_persists_and_loads(tmp_path) -> None:
    c, cfg, manager = _add_client(tmp_path)

    r = c.post(
        "/admin/models/add",
        json={"path": "mlx-community/Llama-3.2-3B-4bit", "type": "lm", "served_name": "llama3"},
    )
    assert r.status_code == 200
    assert r.json()["state"] in ("loading", "resident")  # load kicked off in the background

    # The model is registered and persisted to the config file.
    ids = {m["id"] for m in c.get("/v1/models").json()["data"]}
    assert "llama3" in ids
    assert "served_name: llama3" in cfg.read_text()

    # And it becomes usable once the background load settles.
    for _ in range(100):
        if manager.is_resident("llama3"):
            break
        time.sleep(0.02)
    assert manager.is_resident("llama3")


def test_add_model_duplicate_served_name_conflicts(tmp_path) -> None:
    c, _, _ = _add_client(tmp_path)
    r = c.post(
        "/admin/models/add",
        json={"path": "fake/other", "type": "lm", "served_name": "primary"},
    )
    assert r.status_code == 409
    assert r.json()["error"]["type"] == "served_name_conflict"


def test_add_model_invalid_type_rejected(tmp_path) -> None:
    c, _, _ = _add_client(tmp_path)
    r = c.post(
        "/admin/models/add",
        json={"path": "fake/x", "type": "diffusion", "served_name": "x"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_model_entry"


def test_image_request_without_vision_model_is_rejected() -> None:
    # M6: image-bearing requests route to a VLM; a text-only deployment rejects them.
    c, _ = make_client()
    r = c.post(
        "/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "no_vision_model"
