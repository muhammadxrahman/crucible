"""Manual M6 acceptance: post an image + question through the OpenAI vision API, get an
accurate answer, and confirm a multi-turn re-send of the same image is a vision-cache hit.

    uv run python scripts/smoke_vision.py
"""

from __future__ import annotations

import base64
import io
import threading
import time

import httpx
import uvicorn
from PIL import Image, ImageDraw

from crucible.config import Registry
from crucible.manager import ModelManager, RuntimeProfile, make_loader
from crucible.server import create_app

HOST, PORT = "127.0.0.1", 8125
BASE = f"http://{HOST}:{PORT}"
MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"


def _data_url() -> str:
    img = Image.new("RGB", (260, 90), "white")
    ImageDraw.Draw(img).text((10, 35), "HELLO 42", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _serve() -> uvicorn.Server:
    reg = Registry.model_validate(
        {"models": [{"path": MODEL, "type": "vlm", "served_name": "vision", "pin": True}]}
    )
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=50 * 1024**3,
        single_resident=False,
        default_context=8192,
        kv_bits=8,
        vision=True,
    )
    manager = ModelManager(reg, runtime, make_loader())
    manager.warmup()
    app = create_app(manager, runtime)
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(150):
        try:
            if httpx.get(f"{BASE}/healthz", timeout=1).status_code == 200:
                return server
        except httpx.HTTPError:
            time.sleep(0.2)
    raise RuntimeError("server did not start")


def _ask(url: str, question: str) -> str:
    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": "vision",
            "max_tokens": 32,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
        },
        timeout=120,
    )
    return r.json()["choices"][0]["message"]["content"]


def main() -> None:
    print(f"loading VLM {MODEL} ...")
    server = _serve()
    url = _data_url()

    answer = _ask(url, "What text appears in the image? Answer briefly.")
    print("Q: What text appears in the image?")
    print("A:", answer)
    assert "42" in answer or "HELLO" in answer.upper(), "expected the model to read the image text"

    # Second turn with the same image -> vision cache hit.
    _ask(url, "Is the text a word or a number?")
    summary = httpx.get(f"{BASE}/metrics/summary").json()
    hits = summary["current"]["vision_cache_hits"]
    print(f"\nvision_cache_hits after re-sending the same image: {hits}")
    assert hits >= 1, "expected the repeated image to be a cache hit"

    server.should_exit = True
    print(
        "\nOK: VLM answered about the image; repeated image reused the vision cache (in metrics)."
    )


if __name__ == "__main__":
    main()
