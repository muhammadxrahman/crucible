"""M8: backend support for the web UI — file upload and SPA static serving (no GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from crucible.backends import Delta, Final
from crucible.config import Registry
from crucible.manager import ModelManager, RuntimeProfile
from crucible.rag import RagPipeline, VectorStore, resolve_rag_roles
from crucible.server import create_app

_RUNTIME = RuntimeProfile(
    name="pro64",
    ceiling_bytes=10**12,
    single_resident=False,
    default_context=8192,
    kv_bits=8,
    vision=True,
)
_VOCAB = ["apple", "silicon", "memory", "banana"]


def _vec(text: str) -> list[float]:
    v = np.array([float(text.lower().count(w)) for w in _VOCAB], dtype=np.float32)
    if v.sum() == 0:
        v = np.ones(len(_VOCAB), dtype=np.float32)
    return (v / (np.linalg.norm(v) + 1e-9)).tolist()


class FakeEmbed:
    type = "embedding"

    def embed(self, texts):
        return [_vec(t) for t in texts]


class FakeRerank:
    type = "rerank"

    def rerank(self, query, docs):
        qs = set(query.lower().split())
        return [float(sum(w in qs for w in d.lower().split())) for d in docs]


class FakeGen:
    type = "lm"

    def stream(self, messages, params):
        yield Delta("Unified memory is shared [1].")
        yield Final(prompt_tokens=8, completion_tokens=5, finish_reason="stop")


def _manager(tmp_path: Path):
    reg = Registry.model_validate(
        {
            "models": [
                {"path": "f/g", "type": "lm", "served_name": "gen"},
                {"path": "f/e", "type": "embedding", "served_name": "embed"},
                {"path": "f/r", "type": "rerank", "served_name": "rerank"},
            ],
            "rag": {"store_dir": str(tmp_path / "store"), "top_k": 5, "top_n": 2, "chunk_size": 40},
        }
    )
    engines = {"gen": FakeGen(), "embed": FakeEmbed(), "rerank": FakeRerank()}
    manager = ModelManager(reg, _RUNTIME, lambda e: (engines[e.served_name], 10))
    roles = resolve_rag_roles(reg)
    rag = RagPipeline(
        manager,
        reg.rag,
        embed_name=roles["embed_name"],
        generator_name=roles["generator_name"],
        rerank_name=roles["rerank_name"],
        store=VectorStore(),
    )
    return manager, rag


def test_rag_upload_ingests_uploaded_file(tmp_path: Path) -> None:
    manager, rag = _manager(tmp_path)
    client = TestClient(create_app(manager, _RUNTIME, rag))
    content = b"Apple Silicon uses unified memory shared between CPU and GPU. " * 4
    r = client.post("/rag/upload", files={"files": ("notes.txt", content, "text/plain")})
    assert r.status_code == 200
    assert r.json()["indexed_chunks"] >= 1
    answer = client.post("/rag/query", json={"query": "apple silicon memory"}).json()
    assert answer["sources"]  # the uploaded file is now queryable


def test_rag_upload_without_rag_returns_503(tmp_path: Path) -> None:
    reg = Registry.model_validate({"models": [{"path": "f/g", "type": "lm", "served_name": "gen"}]})
    manager = ModelManager(reg, _RUNTIME, lambda e: (FakeGen(), 10))
    client = TestClient(create_app(manager, _RUNTIME, None))
    r = client.post("/rag/upload", files={"files": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 503


def test_spa_served_at_root_when_built(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>Crucible</title>")
    manager, rag = _manager(tmp_path)
    client = TestClient(create_app(manager, _RUNTIME, rag, web_dist=str(dist)))
    r = client.get("/")
    assert r.status_code == 200
    assert "Crucible" in r.text


def test_api_works_without_a_build(tmp_path: Path) -> None:
    manager, rag = _manager(tmp_path)
    client = TestClient(create_app(manager, _RUNTIME, rag, web_dist=str(tmp_path / "absent")))
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/").status_code == 404
