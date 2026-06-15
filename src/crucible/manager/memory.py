"""Thin, injectable wrapper over MLX memory introspection.

Isolated behind a class so the manager can be tested with a fake accountant that
reports deterministic sizes instead of touching the GPU.
"""

from __future__ import annotations

import mlx.core as mx


class MlxMemory:
    def active_bytes(self) -> int:
        return mx.get_active_memory()

    def clear_cache(self) -> None:
        mx.clear_cache()

    def set_limit(self, num_bytes: int) -> None:
        mx.set_memory_limit(num_bytes)
