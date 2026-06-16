"""Shared helpers for the stress suite (imported as a top-level module; tests/stress is on
sys.path under pytest's default import mode)."""

from __future__ import annotations

import os

CONCURRENCY = int(os.environ.get("CRUCIBLE_STRESS_CONCURRENCY", "16"))


def max_run_of_repeated_tokens(text: str) -> int:
    """Longest run of an identical whitespace-delimited token — a cheap degenerate-loop probe."""
    toks = text.split()
    if not toks:
        return 0
    best = run = 1
    for i in range(1, len(toks)):
        run = run + 1 if toks[i] == toks[i - 1] else 1
        best = max(best, run)
    return best
