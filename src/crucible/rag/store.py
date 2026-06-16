"""In-process vector store: NumPy brute-force cosine search, persisted to local files.

Embedded and zero-ops, with no external daemon, matching the self-contained-app goal.
Embeddings are L2-normalized, so a dot product is cosine similarity. Behind this simple
interface a LanceDB or Qdrant backend can drop in later if ANN or filtering is needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str


class VectorStore:
    def __init__(self) -> None:
        self._vecs: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        arr = np.asarray(embeddings, dtype=np.float32)
        self._vecs = arr if self._vecs is None else np.vstack([self._vecs, arr])
        self._chunks.extend(chunks)

    def remove_doc(self, doc_id: str) -> int:
        """Drop all chunks (and their vectors) for a document. Returns how many were removed.

        Re-ingesting a document calls this first so re-uploads replace rather than duplicate.
        """
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._vecs = self._vecs[keep] if (self._vecs is not None and keep) else None
        return removed

    def search(self, query_vec: list[float], k: int) -> list[tuple[Chunk, float]]:
        if self._vecs is None or not self._chunks:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        sims = self._vecs @ q  # cosine, since rows and query are normalized
        k = min(k, len(self._chunks))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._chunks[i], float(sims[i])) for i in idx]

    def documents(self) -> list[dict]:
        by_doc: dict[str, dict] = {}
        for c in self._chunks:
            d = by_doc.setdefault(c.doc_id, {"doc_id": c.doc_id, "source": c.source, "chunks": 0})
            d["chunks"] += 1
        return list(by_doc.values())

    # --- persistence ---

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self._vecs is not None:
            np.save(d / "vectors.npy", self._vecs)
        with (d / "chunks.jsonl").open("w") as f:
            for c in self._chunks:
                f.write(json.dumps(asdict(c)) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        store = cls()
        d = Path(directory)
        chunks_file = d / "chunks.jsonl"
        vecs_file = d / "vectors.npy"
        if not chunks_file.is_file():
            return store
        chunks = [Chunk(**json.loads(line)) for line in chunks_file.read_text().splitlines()]
        vecs = np.load(vecs_file) if vecs_file.is_file() else None
        # Drop duplicate chunk ids, keeping the first. This heals stores written before
        # re-ingest replaced instead of appended (the duplicate-document bug).
        seen: set[str] = set()
        keep = [i for i, c in enumerate(chunks) if not (c.id in seen or seen.add(c.id))]
        store._chunks = [chunks[i] for i in keep]
        if vecs is not None and len(vecs) == len(chunks):
            store._vecs = vecs[keep] if keep else None
        elif vecs is not None:
            store._vecs = vecs  # length mismatch (shouldn't happen): leave untouched
        return store
