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
    temperature: float = 1.0
    top_p: float = 1.0
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
