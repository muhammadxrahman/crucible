"""Manual M4 acceptance: live metrics over HTTP, no external services.

Boots the gateway (batched, tiny model), fires concurrent traffic, then reads /metrics
and /metrics/summary and confirms the in-app /observability dashboard is served.

    uv run python scripts/smoke_metrics.py
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import uvicorn

from crucible.config import Registry
from crucible.manager import ModelManager, RuntimeProfile, make_loader
from crucible.server import create_app

HOST, PORT = "127.0.0.1", 8124
BASE = f"http://{HOST}:{PORT}"
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def _serve():  # noqa: ANN202
    reg = Registry.model_validate(
        {"models": [{"path": MODEL, "type": "lm", "served_name": "primary", "pin": True}]}
    )
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=50 * 1024**3,
        single_resident=False,
        default_context=8192,
        kv_bits=8,
        vision=True,
    )
    manager = ModelManager(reg, runtime, make_loader(batching=True))
    manager.warmup()
    app = create_app(manager, runtime)
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            if httpx.get(f"{BASE}/healthz", timeout=1).status_code == 200:
                return server
        except httpx.HTTPError:
            time.sleep(0.2)
    raise RuntimeError("server did not start")


def _chat(i: int) -> None:
    httpx.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": "primary",
            "messages": [{"role": "user", "content": f"Count to {i}."}],
            "max_tokens": 48,
        },
        timeout=60,
    )


def main() -> None:
    print(f"loading {MODEL} ...")
    server = _serve()

    print("firing 8 concurrent requests ...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_chat, range(8)))

    text = httpx.get(f"{BASE}/metrics").text
    for series in [
        "crucible_requests_total",
        "crucible_prefill_tps",
        "crucible_decode_tps",
        "crucible_ttft_seconds",
        "crucible_batch_size",
        "crucible_resident_bytes",
    ]:
        assert series in text, f"missing {series}"
    print("  /metrics: all expected series present")

    summary = httpx.get(f"{BASE}/metrics/summary").json()
    cur = summary["current"]
    print(
        f"  decode_tps={cur['decode_tps']:.0f}  prefill_tps={cur['prefill_tps']:.0f}  "
        f"ttft_ms={cur['ttft_ms']:.0f}  peak_batch via requests={cur['requests_total']}"
    )
    assert cur["requests_total"] >= 8
    assert cur["decode_tps"] > 0

    html = httpx.get(f"{BASE}/observability").text
    assert "<title>Crucible" in html
    print("  /observability: dashboard HTML served")

    server.should_exit = True
    print("\nOK: live metrics + in-app dashboard, zero external services.")


if __name__ == "__main__":
    main()
