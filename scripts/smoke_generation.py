"""Guard against degenerate repetition / non-terminating generation.

Runs the tiny model through the real engine with the new config defaults (which include a
repetition penalty) across several samples and a multi-turn case where the prior reply is
fed back. Asserts no verbatim loop collapse. Collapse is stochastic, so this samples.

    uv run python scripts/smoke_generation.py
"""

from __future__ import annotations

from collections import Counter

from crucible.backends import Delta, Final, SamplingParams
from crucible.backends.text import MLXTextEngine
from crucible.batching import PrefixCache

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
LOOP_LIMIT = 6  # a 4-gram repeating more than this = degenerate collapse


def loop_score(text: str) -> int:
    words = text.split()
    if len(words) <= 4:
        return 0
    counts = Counter(tuple(words[i : i + 4]) for i in range(len(words) - 4))
    return counts.most_common(1)[0][1]


def main() -> None:
    print(f"loading {MODEL} ...")
    engine = MLXTextEngine(MODEL, "primary", prefix_cache=PrefixCache(min_prefix=4))

    def chat(messages: list[dict]) -> tuple[str, str]:
        text, finish = "", "length"
        for ev in engine.stream(
            messages, SamplingParams(max_tokens=200)
        ):  # defaults: temp .7, rep 1.1
            if isinstance(ev, Delta):
                text += ev.text
            elif isinstance(ev, Final):
                finish = ev.finish_reason
        return text, finish

    prompts = [
        "What's the current quarter and year?",  # the screenshot's question
        "Explain Apple Silicon unified memory in two sentences.",
        "List three primary colors.",
    ]
    failures = 0
    for q in prompts:
        for trial in range(3):  # stochastic collapse -> sample a few
            text, finish = chat([{"role": "user", "content": q}])
            ls = loop_score(text)
            ok = ls <= LOOP_LIMIT
            print(
                f"  [{q[:34]!r:38} #{trial}] finish={finish:6} loop={ls} {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failures += 1
                print("     ", repr(text[:160]))

    # Multi-turn: feed the prior reply back (the cascade that poisoned the screenshots).
    convo = [{"role": "user", "content": "Name three primary colors."}]
    t1, _ = chat(convo)
    convo += [{"role": "assistant", "content": t1}, {"role": "user", "content": "Why those three?"}]
    t2, f2 = chat(convo)
    ls2 = loop_score(t2)
    ok2 = "ok" if ls2 <= LOOP_LIMIT else "FAIL"
    print(f"  [multi-turn fed-back reply] finish={f2} loop={ls2} {ok2}")
    if ls2 > LOOP_LIMIT:
        failures += 1

    assert failures == 0, f"{failures} generation(s) collapsed into repetition"
    print("\nOK: coherent, terminating output across samples; no repetition collapse.")


if __name__ == "__main__":
    main()
