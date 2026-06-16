"""Detect runaway repetition and stop generation cleanly.

A safety net beneath the repetition penalty: weak/quantized models can still occasionally
collapse into a short verbatim cycle and emit it until `max_tokens`. This guard watches the
recent token ids and reports a loop once the last `window` tokens form an exact short cycle,
so the engine can stop after ~`window` repeated tokens instead of a wall of them. It is
deliberately conservative (a sustained, exact, short-period cycle) so normal text and brief
legitimate repeats never trigger it.
"""

from __future__ import annotations

from collections import deque


class LoopGuard:
    def __init__(self, window: int = 24, max_period: int = 6):
        self._window = window
        self._max_period = max_period
        self._buf: deque[int] = deque(maxlen=window)

    def feed(self, token: int) -> bool:
        """Append a generated token id; return True once the recent output is a tight loop."""
        self._buf.append(token)
        if len(self._buf) < self._window:
            return False
        seq = list(self._buf)
        for period in range(1, self._max_period + 1):
            if all(seq[i] == seq[i - period] for i in range(period, self._window)):
                return True
        return False

    def reset(self) -> None:
        self._buf.clear()
