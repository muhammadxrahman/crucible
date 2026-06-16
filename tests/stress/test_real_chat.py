"""Hammer the text engine: high concurrency, long unlimited generation, and loop safety."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from _util import CONCURRENCY, max_run_of_repeated_tokens


def test_high_concurrency_chat_all_complete(server):
    """Fire many streamed chats at once; every one must finish cleanly with real content and
    no degenerate repetition. This exercises continuous batching and thread-affinity safety."""
    lm = server.caps().get("lm")
    assert lm, "config must serve an lm"

    prompts = [
        "In one sentence, what is a transformer in machine learning?",
        "Name three primary colors.",
        "Write a haiku about the ocean.",
        "What is 17 times 23? Answer with the number.",
        "Give one tip for writing clean code.",
        "Explain recursion in one sentence.",
        "What is the capital of Japan?",
        "List two benefits of unit testing.",
    ]

    def one(i: int):
        msg = [{"role": "user", "content": prompts[i % len(prompts)]}]
        t0 = time.perf_counter()
        text, final = server.chat(lm, msg, max_tokens=120, temperature=0.7)
        return {
            "text": text,
            "finish": final.get("choices", [{}])[0].get("finish_reason"),
            "dt": time.perf_counter() - t0,
        }

    n = CONCURRENCY
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = [f.result() for f in as_completed([ex.submit(one, i) for i in range(n)])]

    assert len(results) == n
    for r in results:
        assert r["text"].strip(), "empty completion under load"
        assert r["finish"] in ("stop", "length"), f"bad finish: {r['finish']}"
        # The repetition penalty + loop guard must keep output from collapsing into a loop.
        assert max_run_of_repeated_tokens(r["text"]) < 12, f"degenerate loop: {r['text'][:200]!r}"

    slowest = max(r["dt"] for r in results)
    print(f"\n[concurrency={n}] all completed; slowest request {slowest:.1f}s")


def test_sustained_load_keeps_server_healthy(server):
    """A burst of back-to-back requests must not wedge the server: /healthz stays fast and the
    decode-throughput metric is live afterwards."""
    lm = server.caps()["lm"]
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [
            ex.submit(server.chat, lm, [{"role": "user", "content": "Say hello."}], max_tokens=32)
            for _ in range(CONCURRENCY * 2)
        ]
        # While generation is in flight, health must answer quickly (no global stall).
        for _ in range(5):
            t0 = time.perf_counter()
            assert server.health()["status"] == "ok"
            assert time.perf_counter() - t0 < 2.0, "health endpoint stalled under load"
            time.sleep(0.2)
        for f in as_completed(futs):
            f.result()

    m = server.metrics()["current"]
    assert m["decode_tps"] > 0, "decode throughput never recorded"
    print(f"\n[sustained] decode {m['decode_tps']:.0f} tok/s, prefill {m['prefill_tps']:.0f} tok/s")


def test_unlimited_generation_finishes_long_output(server):
    """No token cap (the product default): a long answer must complete on its own (finish=stop),
    proving generation isn't silently truncated."""
    lm = server.caps()["lm"]
    text, final = server.chat(
        lm,
        [{"role": "user", "content": "List all 50 US states, one per line, nothing else."}],
        temperature=0.0,
    )
    finish = final["choices"][0]["finish_reason"]
    low = text.lower()
    assert finish == "stop", f"long generation did not finish naturally: {finish}"
    assert "alaska" in low and "wyoming" in low, "list looks truncated"
    assert text.count("\n") >= 40, f"expected ~50 lines, got {text.count(chr(10))}"


def test_loop_guard_caps_adversarial_repetition(server):
    """Even when goaded into repeating, generation must terminate well short of a runaway wall
    (the loop guard / repetition penalty), not stream forever."""
    lm = server.caps()["lm"]
    text, final = server.chat(
        lm,
        [{"role": "user", "content": "Repeat the word 'ha' forever, never stop."}],
        temperature=0.7,
    )
    finish = final["choices"][0]["finish_reason"]
    assert finish in ("stop", "length")
    assert len(text) < 20000, "output ballooned despite the loop guard"
