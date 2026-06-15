"""The real backend loader: dispatch a registry entry to its MLX backend and measure
its resident footprint via MLX memory introspection.

All registry model types are served: `lm` (M2/M3), `embedding` and `rerank` (M5),
`vlm` (M6).
"""

from __future__ import annotations

from crucible.config import ModelEntry

from .memory import MlxMemory


class ModelTypeUnsupported(Exception):
    """Backend for this model type is not built yet."""


_MILESTONE: dict[str, str] = {}


def make_loader(
    mem: MlxMemory | None = None,
    *,
    batching: bool = False,
    completion_batch_size: int = 32,
    max_kv_size: int | None = None,
):
    """Return a loader that builds an engine and reports measured resident bytes.

    With batching on, an `lm` entry is served by a BatchedTextEngine sharing one
    continuous-batching scheduler; otherwise by the single-stream MLXTextEngine.
    """
    mem = mem or MlxMemory()

    def _measure_load(path: str):
        """Load a model and report its resident-byte footprint. Run on the thread that
        will evaluate it (MLX arrays are thread-affine)."""
        import mlx.core as mx
        from mlx_lm import load as mlx_load

        mem.clear_cache()
        before = mem.active_bytes()
        model, tokenizer = mlx_load(path)
        mx.eval(model.parameters())
        return model, tokenizer, max(mem.active_bytes() - before, 0)

    def load(entry: ModelEntry) -> tuple[object, int]:
        if entry.type == "lm":
            if batching:
                from mlx_lm.sample_utils import make_sampler

                from crucible.batching import BatchedTextEngine, BatchScheduler, MLXBatchBackend

                # build() runs on the scheduler's worker thread, so the model and the
                # BatchGenerator share that thread's MLX stream.
                def build():
                    model, tokenizer, nbytes = _measure_load(entry.path)
                    backend = MLXBatchBackend(
                        model,
                        tokenizer,
                        completion_batch_size=completion_batch_size,
                        max_kv_size=max_kv_size,
                    )
                    return backend, tokenizer, nbytes

                scheduler = BatchScheduler(build, make_sampler=make_sampler)
                nbytes = scheduler.wait_ready()
                engine = BatchedTextEngine(
                    scheduler, scheduler.tokenizer, entry.served_name, entry.path
                )
                return engine, nbytes

            from crucible.backends.text import MLXTextEngine
            from crucible.batching import PrefixCache

            model, tokenizer, nbytes = _measure_load(entry.path)
            engine = MLXTextEngine(
                entry.path,
                entry.served_name,
                model=model,
                tokenizer=tokenizer,
                prefix_cache=PrefixCache(),
            )
            return engine, nbytes

        if entry.type == "embedding":
            from crucible.backends.embedding import MLXEmbeddingEngine

            engine = MLXEmbeddingEngine(entry.path, entry.served_name, mem)
            return engine, engine.nbytes

        if entry.type == "rerank":
            from crucible.backends.rerank import MLXRerankEngine

            engine = MLXRerankEngine(entry.path, entry.served_name, mem)
            return engine, engine.nbytes

        if entry.type == "vlm":
            from crucible.backends.vision import MLXVLMEngine

            engine = MLXVLMEngine(entry.path, entry.served_name, mem)
            return engine, engine.nbytes

        milestone = _MILESTONE.get(entry.type, "a later milestone")
        raise ModelTypeUnsupported(
            f"serving '{entry.type}' models ({entry.served_name}) arrives in {milestone}"
        )

    return load
