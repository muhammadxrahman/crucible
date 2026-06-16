"""Chaos under load: hot-add/load/unload models and abuse the API while the engine is busy.
The control plane must stay responsive and nothing may wedge or corrupt."""

from __future__ import annotations

import threading
import time

import httpx
import pytest


def test_dynamic_add_and_admin_while_under_load(server):
    """The headline stress test. While several threads stream chats non-stop:
    - the control plane (/v1/models, /healthz) stays fast (lock released during loads),
    - a downloaded model is hot-added, loads in the background, serves, and is persisted,
    - unload/reload churns without errors,
    - and not a single in-flight chat fails."""
    lm = server.caps()["lm"]
    stop = threading.Event()
    errors: list[str] = []

    def hammer():
        while not stop.is_set():
            try:
                server.chat(
                    lm, [{"role": "user", "content": "Count from one to three."}], max_tokens=24
                )
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

    pool = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
    for t in pool:
        t.start()
    time.sleep(1.0)  # let the load ramp up

    try:
        # Control plane must answer quickly even while the GPU is saturated.
        latencies = []
        for _ in range(12):
            t0 = time.perf_counter()
            server.models()
            server.health()
            latencies.append(time.perf_counter() - t0)
            time.sleep(0.1)
        assert max(latencies) < 2.0, f"control plane stalled under load: {max(latencies):.2f}s"

        # Find a spare small lm in the cache that isn't already served.
        avail = server.client.get("/admin/models/available").json()["data"]
        cand = next(
            (
                m
                for m in avail
                if not m["registered"]
                and m["guessed_type"] == "lm"
                and m["size_bytes"] < 3_000_000_000
            ),
            None,
        )
        if cand is None:
            pytest.skip("no spare small lm in the cache to hot-add")

        added = "stress_added"
        r = server.client.post(
            "/admin/models/add",
            json={"path": cand["repo_id"], "type": "lm", "served_name": added},
        )
        assert r.status_code == 200, r.text
        assert r.json()["state"] in ("loading", "resident")

        # It loads in the background; poll to resident while load continues elsewhere.
        deadline = time.time() + 180
        state = None
        while time.time() < deadline:
            state = {m["id"]: m["state"] for m in server.models()}.get(added)
            if state == "resident":
                break
            assert state in ("loading", "resident"), f"unexpected add state: {state}"
            time.sleep(0.5)
        assert state == "resident", f"hot-added model never became resident (stuck {state})"

        # The new model actually serves, and the add was persisted to the (temp) config.
        text, _ = server.chat(added, [{"role": "user", "content": "Say hi."}], max_tokens=16)
        assert text.strip(), "hot-added model returned nothing"
        assert added in server.config_path.read_text(), "add was not persisted"

        # Churn residency under load — must not 5xx.
        assert (
            server.client.post("/admin/models/unload", json={"served_name": added}).status_code
            == 200
        )
        assert (
            server.client.post("/admin/models/load", json={"served_name": added}).status_code == 200
        )

        ms = max(latencies) * 1000
        print(f"\n[chaos] hot-added {cand['repo_id']}; control-plane max {ms:.0f}ms")
    finally:
        stop.set()
        for t in pool:
            t.join(timeout=15)

    assert not errors, f"{len(errors)} chat(s) failed during chaos: {errors[:3]}"


def test_memory_ceiling_respected(server):
    """After exercising several large models, resident memory must stay within the profile
    ceiling (the manager evicts to fit; the unified pool is never oversubscribed)."""
    h = server.health()
    ceiling = h["memory_ceiling_gb"]
    assert h["resident_gb"] <= ceiling * 1.1, (
        f"resident {h['resident_gb']}GB over ceiling {ceiling}GB"
    )


def test_unknown_model_returns_clean_404(server):
    r = server.client.post(
        "/v1/chat/completions",
        json={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"


def test_malformed_request_is_rejected_not_crashing(server):
    r = server.client.post(
        "/v1/chat/completions", json={"model": server.caps()["lm"]}
    )  # no messages
    assert r.status_code == 422  # validation error, server stays up
    assert server.health()["status"] == "ok"


def test_client_disconnect_midstream_keeps_server_healthy(server):
    """Aborting a stream mid-generation must not destabilize the server: a fresh request right
    after must still succeed."""
    lm = server.caps()["lm"]
    with httpx.Client(base_url=server.base, timeout=60.0) as c:
        with c.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": lm,
                "messages": [{"role": "user", "content": "Write a very long essay about the sea."}],
                "stream": True,
            },
        ) as r:
            # Read a couple of lines, then bail out (context-manager close = client disconnect).
            for i, _ in enumerate(r.iter_lines()):
                if i >= 2:
                    break

    text, _ = server.chat(lm, [{"role": "user", "content": "Reply with: ok"}], max_tokens=8)
    assert text.strip(), "server did not recover after a client disconnect"
    assert server.health()["status"] == "ok"
