"""M5: RAG chunking, vector store, document loading, pipeline, and endpoints.

All driven by deterministic fake engines, so no model or GPU is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from crucible.backends import Delta, Final
from crucible.config import RagConfig, Registry
from crucible.manager import ModelManager, RuntimeProfile
from crucible.rag import (
    Chunk,
    RagPipeline,
    VectorStore,
    chunk_text,
    iter_files,
    load_text,
    resolve_rag_roles,
)
from crucible.server import create_app

VOCAB = ["apple", "silicon", "memory", "banana", "potassium", "bandwidth"]


def _vec(text: str) -> list[float]:
    t = text.lower()
    v = np.array([float(t.count(w)) for w in VOCAB], dtype=np.float32)
    if v.sum() == 0:
        v = np.ones(len(VOCAB), dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


class FakeEmbed:
    type = "embedding"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_vec(t) for t in texts]


class FakeRerank:
    type = "rerank"

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        qs = set(query.lower().split())
        return [float(sum(w in qs for w in d.lower().split())) for d in docs]


class FakeGen:
    type = "lm"

    def stream(self, messages, params):
        yield Delta("Unified memory is shared between CPU and GPU [1].")
        yield Final(prompt_tokens=10, completion_tokens=9, finish_reason="stop")


# --- chunking ---


def test_chunk_overlap_and_coverage() -> None:
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, size=30, overlap=10)
    assert len(chunks) > 1
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]  # overlap


def test_chunk_short_text_single() -> None:
    assert chunk_text("just a few words", size=50) == ["just a few words"]
    assert chunk_text("   ") == []


# --- store ---


def test_store_search_ranks_by_cosine() -> None:
    s = VectorStore()
    s.add(
        [Chunk("1", "d", "a.txt", "apple silicon"), Chunk("2", "d", "a.txt", "banana potassium")],
        [_vec("apple silicon"), _vec("banana potassium")],
    )
    hits = s.search(_vec("apple memory"), k=2)
    assert hits[0][0].text == "apple silicon"
    assert hits[0][1] >= hits[1][1]


def test_store_persists_round_trip(tmp_path: Path) -> None:
    s = VectorStore()
    s.add([Chunk("1", "d", "a.txt", "hello world")], [_vec("apple")])
    s.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 1
    assert loaded.search(_vec("apple"), 1)[0][0].text == "hello world"


# --- documents ---


def test_iter_files_filters_supported(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hi")
    (tmp_path / "b.md").write_text("hi")
    (tmp_path / "c.bin").write_text("nope")
    files = {f.name for f in iter_files(tmp_path)}
    assert files == {"a.txt", "b.md"}


def test_load_text_reads_markdown(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# Title\nbody")
    assert "body" in load_text(p)


# --- pipeline ---


class FakeManager:
    def __init__(self) -> None:
        self._e = {"embed": FakeEmbed(), "rerank": FakeRerank(), "gen": FakeGen()}

    def acquire(self, name: str):
        return self._e[name]


def _pipeline(tmp_path: Path, **cfg_kw) -> RagPipeline:
    cfg = RagConfig(store_dir=str(tmp_path / "store"), chunk_size=40, chunk_overlap=8, **cfg_kw)
    return RagPipeline(
        FakeManager(),
        cfg,
        embed_name="embed",
        generator_name="gen",
        rerank_name="rerank",
        store=VectorStore(),
    )


def _corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "apple.txt").write_text("Apple Silicon shares unified memory between CPU and GPU. " * 6)
    (docs / "banana.txt").write_text("Bananas are a good source of potassium and fiber. " * 6)
    return docs


def test_ingest_then_grounded_query(tmp_path: Path) -> None:
    pipe = _pipeline(tmp_path, top_k=5, top_n=2)
    res = pipe.ingest(str(_corpus(tmp_path)))
    assert res["indexed_chunks"] >= 2
    assert len(res["documents"]) == 2

    out = pipe.query("apple silicon unified memory")
    assert out["reranked"] is True
    assert out["sources"]
    assert "apple" in out["sources"][0]["text"].lower()  # retrieved the right chunk
    assert "[1]" in out["answer"]  # grounded answer carries a citation


def test_rerank_toggle(tmp_path: Path) -> None:
    pipe = _pipeline(tmp_path)
    pipe.ingest(str(_corpus(tmp_path)))
    assert pipe.query("apple", rerank=False)["reranked"] is False


def test_query_with_empty_store(tmp_path: Path) -> None:
    out = _pipeline(tmp_path).query("anything")
    assert out["sources"] == []


# --- endpoints ---


def _rag_client(tmp_path: Path) -> TestClient:
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
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=10**12,
        single_resident=False,
        default_context=8192,
        kv_bits=8,
        vision=True,
    )
    engines = {"gen": FakeGen(), "embed": FakeEmbed(), "rerank": FakeRerank()}

    def loader(entry):
        return engines[entry.served_name], 10

    manager = ModelManager(reg, runtime, loader)
    roles = resolve_rag_roles(reg)
    rag = RagPipeline(
        manager,
        reg.rag,
        embed_name=roles["embed_name"],
        generator_name=roles["generator_name"],
        rerank_name=roles["rerank_name"],
        store=VectorStore(),
    )
    return TestClient(create_app(manager, runtime, rag))


def test_embeddings_endpoint(tmp_path: Path) -> None:
    c = _rag_client(tmp_path)
    body = c.post("/v1/embeddings", json={"model": "embed", "input": ["apple", "banana"]}).json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2
    assert len(body["data"][0]["embedding"]) == len(VOCAB)


def test_embeddings_wrong_type_400(tmp_path: Path) -> None:
    c = _rag_client(tmp_path)
    r = c.post("/v1/embeddings", json={"model": "gen", "input": "x"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "model_type_error"


def test_rerank_endpoint_sorts(tmp_path: Path) -> None:
    c = _rag_client(tmp_path)
    r = c.post(
        "/v1/rerank",
        json={
            "model": "rerank",
            "query": "apple silicon",
            "documents": ["nothing here", "apple silicon memory", "apple only"],
        },
    ).json()
    assert r["results"][0]["index"] == 1  # best match first


def test_rag_ingest_query_documents_endpoints(tmp_path: Path) -> None:
    c = _rag_client(tmp_path)
    ingest = c.post("/rag/ingest", json={"paths": str(_corpus(tmp_path))}).json()
    assert ingest["indexed_chunks"] >= 2

    docs = c.get("/rag/documents").json()["documents"]
    assert len(docs) == 2

    answer = c.post("/rag/query", json={"query": "apple silicon memory"}).json()
    assert answer["sources"]
    assert "[1]" in answer["answer"]


def test_rag_disabled_returns_503(tmp_path: Path) -> None:
    reg = Registry.model_validate({"models": [{"path": "f/g", "type": "lm", "served_name": "gen"}]})
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=10**12,
        single_resident=False,
        default_context=8192,
        kv_bits=8,
        vision=True,
    )
    manager = ModelManager(reg, runtime, lambda e: (FakeGen(), 10))
    c = TestClient(create_app(manager, runtime, None))
    assert c.post("/rag/query", json={"query": "x"}).status_code == 503
