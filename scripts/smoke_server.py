"""Manual M1 acceptance: boot the gateway on a tiny real model and exercise it with
the stock OpenAI client and a raw SSE read. Confirms compatibility end to end.

    uv run python scripts/smoke_server.py
"""

from __future__ import annotations

import threading
import time

import httpx
import uvicorn
from openai import OpenAI

from crucible.backends.text import MLXTextEngine
from crucible.server import create_app

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
HOST, PORT = "127.0.0.1", 8123
BASE = f"http://{HOST}:{PORT}/v1"


def _serve() -> uvicorn.Server:
    print(f"loading {MODEL} ...")
    engine = MLXTextEngine(MODEL, "primary")
    app = create_app(engine, profile="pro64")
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            if httpx.get(f"http://{HOST}:{PORT}/healthz", timeout=1).status_code == 200:
                return server
        except httpx.HTTPError:
            time.sleep(0.2)
    raise RuntimeError("server did not become ready")


def main() -> None:
    server = _serve()
    client = OpenAI(base_url=BASE, api_key="not-needed")

    print("\n[1] non-streaming chat via openai client")
    r = client.chat.completions.create(
        model="primary",
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        max_tokens=16,
    )
    text = r.choices[0].message.content
    print("   ->", repr(text))
    assert text and text.strip(), "expected non-empty content"

    print("[2] streaming chat via openai client")
    stream = client.chat.completions.create(
        model="primary",
        messages=[{"role": "user", "content": "Count: 1 2 3"}],
        max_tokens=24,
        stream=True,
    )
    acc = "".join(c.choices[0].delta.content or "" for c in stream)
    print("   ->", repr(acc))
    assert acc.strip(), "expected streamed content"

    print("[3] raw SSE shape over httpx")
    with httpx.stream(
        "POST",
        f"{BASE}/chat/completions",
        json={"model": "primary", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        timeout=60,
    ) as resp:
        lines = [ln for ln in resp.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]", "stream must end with [DONE]"
    print(f"   -> {len(lines)} SSE data lines, terminated by [DONE]")

    print("[4] unknown model returns OpenAI error envelope")
    err = httpx.post(
        f"{BASE}/chat/completions",
        json={"model": "ghost", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert err.status_code == 404 and err.json()["error"]["type"] == "model_not_found"
    print("   -> 404 model_not_found")

    server.should_exit = True
    print("\nOK: M1 gateway is OpenAI-compatible (non-stream, stream, SSE, errors).")


if __name__ == "__main__":
    main()
