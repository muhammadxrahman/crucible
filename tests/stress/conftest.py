"""Real-model stress / integration harness.

These tests spin up an actual Crucible server with the real MLX models and hammer it from
many angles at once. They are the opposite of the GPU-free unit gate: they prove the running
system is iron-clad under concurrency, model hot-swaps, and the full capability surface.

They are opt-in and excluded from the default `pytest` run. To run the full gauntlet:

    uv run pytest -m real tests/stress -v

Knobs (env):
    CRUCIBLE_STRESS_CONFIG       base registry to copy (default: config/models.yaml)
    CRUCIBLE_STRESS_CONCURRENCY  parallel requests in the concurrency tests (default: 16)
    CRUCIBLE_STRESS_STARTUP      server readiness timeout, seconds (default: 600)

The server runs from a *temp copy* of the config, so the dynamic add-model tests never touch
the real config/models.yaml. Every test under tests/stress is auto-marked `real`.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTUP_TIMEOUT = float(os.environ.get("CRUCIBLE_STRESS_STARTUP", "600"))


def pytest_collection_modifyitems(config, items):
    """Auto-mark everything under tests/stress as `real` so it's excluded from the default gate."""
    here = str(Path(__file__).parent)
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.real)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_ready(base: str, proc: subprocess.Popen, log: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log.read_text()[-3000:] if log.is_file() else ""
            raise RuntimeError(f"server exited early (code {proc.returncode}):\n{tail}")
        try:
            r = httpx.get(base + "/healthz", timeout=3.0)
            if r.status_code == 200 and r.json().get("resident_models"):
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("server did not become ready before the startup timeout")


@pytest.fixture(scope="session")
def server() -> Api:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        pytest.skip("real-model stress tests require Apple Silicon")
    base_cfg = Path(os.environ.get("CRUCIBLE_STRESS_CONFIG", REPO_ROOT / "config" / "models.yaml"))
    if not base_cfg.is_file():
        pytest.skip(f"stress config not found: {base_cfg}")

    port = _free_port()
    tmpdir = Path(tempfile.mkdtemp(prefix="crucible-stress-"))
    # A temp copy with an isolated RAG store so the suite never touches the real config or the
    # real .crucible/rag store. add-model persistence lands in this throwaway copy.
    import yaml

    data = yaml.safe_load(base_cfg.read_text()) or {}
    data.setdefault("rag", {})["store_dir"] = str(tmpdir / "rag-store")
    cfg = tmpdir / "models.yaml"
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))
    log = tmpdir / "server.log"

    with log.open("w") as logf:
        proc = subprocess.Popen(
            ["uv", "run", "mlxd", "serve", "-c", str(cfg), "--port", str(port), "--no-open"],
            cwd=REPO_ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_ready(base, proc, log, STARTUP_TIMEOUT)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise

    api = Api(base, cfg)
    yield api

    api.close()
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(tmpdir, ignore_errors=True)


class Api:
    """Thin client over the running server with the stress helpers tests share."""

    def __init__(self, base: str, config_path: Path):
        self.base = base
        self.config_path = config_path
        self.client = httpx.Client(base_url=base, timeout=600.0)

    def close(self) -> None:
        self.client.close()

    # --- discovery (tests adapt to whatever the config serves) ---
    def models(self) -> list[dict]:
        return self.client.get("/v1/models").json()["data"]

    def caps(self) -> dict[str, str]:
        caps: dict[str, str] = {}
        for m in self.models():
            caps.setdefault(m["type"], m["id"])
        return caps

    def health(self) -> dict:
        return self.client.get("/healthz").json()

    def metrics(self) -> dict:
        return self.client.get("/metrics/summary").json()

    # --- generation ---
    def chat(self, model: str, messages, *, stream: bool = True, **params):
        """Returns (text, final_event_or_usage). Streams by default."""
        body = {"model": model, "messages": messages, **params}
        if not stream:
            r = self.client.post("/v1/chat/completions", json=body)
            r.raise_for_status()
            d = r.json()
            return d["choices"][0]["message"]["content"], d
        body["stream"] = True
        text, final = "", {}
        with self.client.stream("POST", "/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                ch = ev["choices"][0]
                text += ch["delta"].get("content", "") or ""
                if ch.get("finish_reason"):
                    final = ev
        return text, final
