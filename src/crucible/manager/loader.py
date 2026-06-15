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


def make_loader(mem: MlxMemory | None = None):
    """Return a loader that builds an engine and reports measured resident bytes."""
    mem = mem or MlxMemory()

    def load(entry: ModelEntry) -> tuple[object, int]:
        if entry.type == "lm":
            from crucible.backends.text import MLXTextEngine

            mem.clear_cache()
            before = mem.active_bytes()
            engine = MLXTextEngine(entry.path, entry.served_name)
            engine.materialize()
            after = mem.active_bytes()
            return engine, max(after - before, 0)

        milestone = _MILESTONE.get(entry.type, "a later milestone")
        raise ModelTypeUnsupported(
            f"serving '{entry.type}' models ({entry.served_name}) arrives in {milestone}"
        )

    return load
