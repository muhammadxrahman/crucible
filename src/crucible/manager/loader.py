"""The real backend loader: dispatch a registry entry to its MLX backend and measure
its resident footprint via MLX memory introspection.

Only `lm` is implemented in M2. Other types report a clear not-yet error so the manager
architecture is type-aware before the vision and embedding backends land.
"""

from __future__ import annotations

from crucible.config import ModelEntry

from .memory import MlxMemory


class ModelTypeUnsupported(Exception):
    """Backend for this model type is not built yet."""


_MILESTONE = {"vlm": "M6", "embedding": "M5", "rerank": "M5"}


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

    def load(entry: ModelEntry) -> tuple[object, int]:
        if entry.type == "lm":
            import mlx.core as mx
            from mlx_lm import load as mlx_load

            mem.clear_cache()
            before = mem.active_bytes()
            model, tokenizer = mlx_load(entry.path)
            mx.eval(model.parameters())
            nbytes = max(mem.active_bytes() - before, 0)

            if batching:
                from mlx_lm.sample_utils import make_sampler

                from crucible.batching import BatchedTextEngine, BatchScheduler, MLXBatchBackend

                backend = MLXBatchBackend(
                    model,
                    tokenizer,
                    completion_batch_size=completion_batch_size,
                    max_kv_size=max_kv_size,
                )
                scheduler = BatchScheduler(backend, tokenizer, make_sampler=make_sampler)
                engine = BatchedTextEngine(scheduler, tokenizer, entry.served_name, entry.path)
            else:
                from crucible.backends.text import MLXTextEngine
                from crucible.batching import PrefixCache

                engine = MLXTextEngine(
                    entry.path,
                    entry.served_name,
                    model=model,
                    tokenizer=tokenizer,
                    prefix_cache=PrefixCache(),
                )
            return engine, nbytes

        milestone = _MILESTONE.get(entry.type, "a later milestone")
        raise ModelTypeUnsupported(
            f"serving '{entry.type}' models ({entry.served_name}) arrives in {milestone}"
        )

    return load
