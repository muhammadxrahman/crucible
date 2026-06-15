"""MLX embedding backend over mlx-embeddings.

Produces L2-normalized sentence embeddings (`text_embeds`) for dense retrieval. All MLX
work, including the model load, runs on a single owned thread: MLX arrays are
thread-affine, so the load and every forward pass must share one thread (the lesson from
the batching milestone).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from crucible.manager.memory import MlxMemory


class MLXEmbeddingEngine:
    type = "embedding"

    def __init__(self, model_path: str, served_name: str, mem: MlxMemory | None = None):
        self.model_path = model_path
        self.served_name = served_name
        self._mem = mem or MlxMemory()
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"embed-{served_name}")
        self.nbytes = self._ex.submit(self._load).result()

    def _load(self) -> int:
        import mlx.core as mx
        import mlx_embeddings as me

        self._mem.clear_cache()
        before = self._mem.active_bytes()
        self._model, self._tok = me.load(self.model_path)
        mx.eval(self._model.parameters())
        return max(self._mem.active_bytes() - before, 0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._ex.submit(self._embed, texts).result()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import mlx.core as mx
        import mlx_embeddings as me

        out = me.generate(self._model, self._tok, texts)
        emb = out.text_embeds
        mx.eval(emb)
        return emb.tolist()

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
