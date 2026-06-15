"""Batch backend abstraction over mlx-lm's BatchGenerator.

The scheduler depends on this Protocol, not on BatchGenerator directly, so the
scheduling logic (admission, in-flight insertion, streaming, completion, prefix reuse)
is tested with a deterministic fake and no GPU.
"""

from __future__ import annotations

from typing import Any, Protocol


class BatchResponse(Protocol):
    uid: int
    token: int
    finish_reason: str | None
    prompt_cache: Any


class BatchBackend(Protocol):
    def insert(
        self,
        prompts: list[list[int]],
        max_tokens: list[int],
        caches: list[Any] | None = None,
        samplers: list[Any] | None = None,
    ) -> list[int]:
        """Add sequences to the running batch; return their uids."""
        ...

    def next_generated(self) -> list[BatchResponse]:
        """Step the batch once; return one response per active sequence (or [] if idle)."""
        ...

    def remove(self, uids: list[int]) -> None: ...

    def kv_nbytes(self) -> int: ...

    def close(self) -> None: ...


class MLXBatchBackend:
    """Concrete backend driving mlx_lm.generate.BatchGenerator."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        max_kv_size: int | None = None,
    ):
        import importlib

        gen_mod = importlib.import_module("mlx_lm.generate")
        self._gen = gen_mod.BatchGenerator(
            model,
            stop_tokens=[[t] for t in tokenizer.eos_token_ids],
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
            max_kv_size=max_kv_size,
        )

    def insert(
        self,
        prompts: list[list[int]],
        max_tokens: list[int],
        caches: list[Any] | None = None,
        samplers: list[Any] | None = None,
    ) -> list[int]:
        return self._gen.insert(prompts, max_tokens, caches=caches, samplers=samplers)

    def next_generated(self) -> list[BatchResponse]:
        return self._gen.next_generated()

    def remove(self, uids: list[int]) -> None:
        self._gen.remove(uids)

    def kv_nbytes(self) -> int:
        try:
            return int(self._gen.prompt_cache_nbytes())
        except Exception:
            return 0

    def close(self) -> None:
        self._gen.close()
