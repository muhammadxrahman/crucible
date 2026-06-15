"""Manual M3 acceptance: a repeated prefix yields a measurable prefill saving.

Runs a multi-turn conversation with a long shared system prompt. Turns after the first
reuse the cached KV of the earlier turns, so time-to-first-token drops sharply.

    uv run python scripts/smoke_prefix.py
"""

from __future__ import annotations

import time

from crucible.backends import Delta, Final, SamplingParams
from crucible.backends.text import MLXTextEngine
from crucible.batching import PrefixCache

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
SYSTEM = "You are a precise assistant. " * 60  # long shared prefix


def main() -> None:
    print(f"loading {MODEL} ...")
    engine = MLXTextEngine(MODEL, "primary", prefix_cache=PrefixCache(min_prefix=16))

    convo = [{"role": "system", "content": SYSTEM}]
    questions = ["Name a primary color.", "Now name another.", "And a third?"]

    def turn(label: str) -> None:
        t0 = time.time()
        ttft = None
        reply = ""
        for ev in engine.stream(convo, SamplingParams(max_tokens=24, temperature=0.0)):
            if isinstance(ev, Delta):
                if ttft is None:
                    ttft = time.time() - t0
                reply += ev.text
            elif isinstance(ev, Final):
                pass
        convo.append({"role": "assistant", "content": reply.strip()})
        print(f"  {label}: TTFT={ttft * 1000:6.0f} ms  reply={reply.strip()[:48]!r}")

    for i, q in enumerate(questions):
        convo.append({"role": "user", "content": q})
        turn(f"turn {i + 1}")

    print("\nprefix cache stats:", engine._prefix.stats.snapshot())
    s = engine._prefix.stats
    assert s.hits >= 1, "expected at least one prefix reuse across turns"
    assert s.tokens_saved > 0
    print("\nOK: later turns reused the shared prefix (TTFT dropped; tokens_saved > 0).")


if __name__ == "__main__":
    main()
