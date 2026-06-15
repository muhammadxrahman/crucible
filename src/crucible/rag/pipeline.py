"""RAG pipeline: ingest local documents, then answer with grounded citations.

Two-stage retrieval (dense search, then optional cross-encoder rerank) feeds a grounded
generation prompt. Engines are acquired from the model manager by served_name, so RAG
reuses the same residency, routing, and eviction as every other model. No network access.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from crucible.backends import Delta, SamplingParams

from .chunk import chunk_text
from .documents import iter_files, load_text
from .store import Chunk, VectorStore

_SYSTEM = (
    "Answer the question using only the numbered sources below. Cite the sources you use "
    "inline as [1], [2], etc. If the sources do not contain the answer, say you don't know."
)


@dataclass
class Source:
    n: int
    source: str
    doc_id: str
    score: float
    text: str


def resolve_rag_roles(registry) -> dict:
    cfg = registry.rag

    def first(t: str) -> str | None:
        return next((m.served_name for m in registry.models if m.type == t), None)

    return {
        "embed_name": cfg.embed_model or first("embedding"),
        "generator_name": cfg.generator_model or first("lm"),
        "rerank_name": cfg.rerank_model or first("rerank"),
    }


class RagPipeline:
    def __init__(
        self,
        manager,
        cfg,
        *,
        embed_name: str,
        generator_name: str,
        rerank_name: str | None = None,
        store: VectorStore | None = None,
    ):
        self.manager = manager
        self.cfg = cfg
        self.embed_name = embed_name
        self.generator_name = generator_name
        self.rerank_name = rerank_name
        self.store = store if store is not None else VectorStore.load(cfg.store_dir)

    # --- ingestion ---

    def ingest(self, paths: str | list[str]) -> dict:
        targets = paths if isinstance(paths, list) else [paths]
        files: list[Path] = []
        for p in targets:
            files.extend(iter_files(p))
        embedder = self.manager.acquire(self.embed_name)

        documents = []
        for f in files:
            text = load_text(f)
            texts = chunk_text(text, size=self.cfg.chunk_size, overlap=self.cfg.chunk_overlap)
            if not texts:
                continue
            doc_id = _doc_id(f)
            embeds = embedder.embed(texts)
            chunks = [
                Chunk(id=f"{doc_id}:{i}", doc_id=doc_id, source=str(f), text=t)
                for i, t in enumerate(texts)
            ]
            self.store.add(chunks, embeds)
            documents.append({"doc_id": doc_id, "source": str(f), "chunks": len(chunks)})

        self.store.save(self.cfg.store_dir)
        return {"documents": documents, "indexed_chunks": len(self.store)}

    # --- query ---

    def query(
        self,
        question: str,
        *,
        rerank: bool | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> dict:
        use_rerank = self.cfg.rerank if rerank is None else rerank
        top_k = top_k or self.cfg.top_k
        top_n = top_n or self.cfg.top_n

        embedder = self.manager.acquire(self.embed_name)
        qvec = embedder.embed([question])[0]
        cands = self.store.search(qvec, top_k)
        if not cands:
            return {"answer": "No documents are indexed yet.", "sources": [], "reranked": False}

        reranked = False
        if use_rerank and self.rerank_name:
            engine = self.manager.acquire(self.rerank_name)
            scores = engine.rerank(question, [c.text for c, _ in cands])
            order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
            cands = [(cands[i][0], scores[i]) for i in order]
            reranked = True

        top = cands[:top_n]
        sources = [
            Source(n=i + 1, source=c.source, doc_id=c.doc_id, score=round(s, 4), text=c.text)
            for i, (c, s) in enumerate(top)
        ]
        answer = self._generate(question, sources)
        return {"answer": answer, "sources": [asdict(s) for s in sources], "reranked": reranked}

    def documents(self) -> list[dict]:
        return self.store.documents()

    # --- internals ---

    def _generate(self, question: str, sources: list[Source]) -> str:
        engine = self.manager.acquire(self.generator_name)
        context = _format_context(sources, self.cfg.max_context_chars)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
        ]
        text = ""
        for ev in engine.stream(messages, SamplingParams(max_tokens=400, temperature=0.0)):
            if isinstance(ev, Delta):
                text += ev.text
        return text.strip()


def _format_context(sources: list[Source], max_chars: int) -> str:
    out, used = [], 0
    for s in sources:
        block = f"[{s.n}] (source: {Path(s.source).name})\n{s.text}"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return "\n\n".join(out)


def _doc_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:12]
