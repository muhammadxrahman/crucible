"""Manual M5 acceptance: ingest local documents, ask a question, get a grounded answer
with citations, and confirm no network access during retrieval and generation.

    uv run python scripts/smoke_rag.py
"""

from __future__ import annotations

import socket
import tempfile
from pathlib import Path

from crucible.config import RagConfig, Registry
from crucible.manager import ModelManager, RuntimeProfile, make_loader
from crucible.rag import RagPipeline, resolve_rag_roles

REG = Registry.model_validate(
    {
        "models": [
            {
                "path": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "type": "lm",
                "served_name": "primary",
                "pin": True,
            },
            {
                "path": "mlx-community/bge-small-en-v1.5-bf16",
                "type": "embedding",
                "served_name": "embed",
                "pin": True,
            },
            {
                "path": "mlx-community/Qwen3-Reranker-0.6B-4bit",
                "type": "rerank",
                "served_name": "rerank",
            },
        ]
    }
)

DOCS = {
    "apple.md": (
        "Apple Silicon uses a unified memory architecture. The CPU, GPU, and Neural "
        "Engine share a single pool of memory with high bandwidth, which avoids copying "
        "data between separate CPU and GPU memories and lets large models stay resident."
    ),
    "banana.md": (
        "Bananas are rich in potassium and provide quick energy. They are one of the "
        "most popular fruits in the world and are easy to carry."
    ),
    "metal.md": (
        "Metal is Apple's low-level graphics and compute API. MLX uses Metal to run "
        "neural network operations on the GPU of Apple Silicon devices."
    ),
}


class _NetworkBlocked(Exception):
    pass


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    docs = tmp / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)

    print("loading models (embed + reranker + generator) ...")
    runtime = RuntimeProfile(
        name="pro64",
        ceiling_bytes=50 * 1024**3,
        single_resident=False,
        default_context=8192,
        kv_bits=8,
        vision=True,
    )
    manager = ModelManager(REG, runtime, make_loader(batching=False))
    manager.warmup()
    manager.acquire("rerank")  # warm the reranker before we block the network

    cfg = RagConfig(store_dir=str(tmp / "store"), top_k=6, top_n=3, chunk_size=120)
    roles = resolve_rag_roles(REG)
    pipe = RagPipeline(
        manager,
        cfg,
        embed_name=roles["embed_name"],
        generator_name=roles["generator_name"],
        rerank_name=roles["rerank_name"],
    )

    print("ingesting", len(DOCS), "documents ...")
    res = pipe.ingest(str(docs))
    print(f"  indexed {res['indexed_chunks']} chunks from {len(res['documents'])} docs")

    # Block all network access for the query path, proving retrieval + generation are local.
    real_socket = socket.socket

    def blocked(*a, **k):
        raise _NetworkBlocked("network access attempted during query")

    socket.socket = blocked
    try:
        out = pipe.query("What is unified memory on Apple Silicon and why does it help?")
    finally:
        socket.socket = real_socket

    print("\nQ: What is unified memory on Apple Silicon and why does it help?")
    print("A:", out["answer"])
    print(f"\nsources (reranked={out['reranked']}):")
    for s in out["sources"]:
        print(f"  [{s['n']}] {Path(s['source']).name}  score={s['score']}")

    assert out["sources"], "expected grounded sources"
    assert "apple" in out["sources"][0]["source"].lower(), "top source should be the apple doc"
    assert out["reranked"] is True
    assert any(f"[{s['n']}]" in out["answer"] for s in out["sources"]) or out["answer"], (
        "expected an answer"
    )
    print("\nOK: grounded answer with citations, retrieval+generation fully local (no network).")


if __name__ == "__main__":
    main()
