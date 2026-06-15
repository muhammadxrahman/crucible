"""MLX cross-encoder reranker over an mlx-lm reranker model (Qwen3-Reranker class).

Scores a (query, document) pair jointly by reading the model's yes/no logits, the
standard Qwen3-Reranker mechanism. This is the second-stage precision pass after dense
retrieval. All MLX work runs on one owned thread (arrays are thread-affine).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from crucible.manager.memory import MlxMemory

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_INSTRUCT = "Given a web search query, judge whether the document is relevant."


class MLXRerankEngine:
    type = "rerank"

    def __init__(self, model_path: str, served_name: str, mem: MlxMemory | None = None):
        self.model_path = model_path
        self.served_name = served_name
        self._mem = mem or MlxMemory()
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rerank-{served_name}")
        self.nbytes = self._ex.submit(self._load).result()

    def _load(self) -> int:
        import mlx.core as mx
        from mlx_lm import load

        self._mem.clear_cache()
        before = self._mem.active_bytes()
        self._model, self._tok = load(self.model_path)
        mx.eval(self._model.parameters())
        self._yes = self._tok.encode("yes", add_special_tokens=False)[0]
        self._no = self._tok.encode("no", add_special_tokens=False)[0]
        return max(self._mem.active_bytes() - before, 0)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return a relevance score in [0, 1] for each document, aligned to input order."""
        if not documents:
            return []
        return self._ex.submit(self._rerank, query, documents).result()

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        import mlx.core as mx

        scores = []
        for doc in documents:
            text = (
                _PREFIX + f"<Instruct>: {_INSTRUCT}\n<Query>: {query}\n<Document>: {doc}" + _SUFFIX
            )
            ids = self._tok.encode(text)
            logits = self._model(mx.array(ids)[None])[:, -1, :]
            pair = mx.softmax(mx.array([logits[0, self._yes], logits[0, self._no]]))
            scores.append(float(pair[0]))
        return scores

    def stats(self) -> dict:
        return {}

    def close(self) -> None:
        def _free() -> None:
            import mlx.core as mx

            self._model = None
            self._tok = None
            mx.clear_cache()

        self._ex.submit(_free).result()
        self._ex.shutdown(wait=False)
