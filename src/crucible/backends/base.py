"""Backend-agnostic generation interface.

The orchestration and gateway layers depend on this, never on mlx-lm directly, so the
MLX engine can be swapped for a fake in tests or a different runtime later.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class SamplingParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    # A repetition penalty > 1 is what stops weak/quantized models collapsing into verbatim
    # loops (and never emitting EOS). 1.0 disables it.
    repetition_penalty: float = 1.1
    repetition_context_size: int = 20
    # Hard stop if generation collapses into a tight repetition loop (a safety net under the
    # penalty, for weak models). Disable for intentionally repetitive output.
    loop_guard: bool = True
    stop: list[str] = field(default_factory=list)


@dataclass
class Delta:
    """An incremental piece of generated text."""

    text: str


@dataclass
class Final:
    """End-of-generation marker with token accounting and separated throughput.

    Prefill and decode throughput are kept distinct, never blended (docs/hardware.md).
    """

    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length"
    prefill_tps: float = 0.0
    decode_tps: float = 0.0


GenEvent = Delta | Final


@runtime_checkable
class TextEngine(Protocol):
    served_name: str
    model_path: str

    def stream(self, messages: list[dict], params: SamplingParams) -> Iterator[GenEvent]: ...
