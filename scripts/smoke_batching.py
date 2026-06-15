"""Manual M3 acceptance: aggregate throughput rises with concurrent requests.

Drives the continuous-batching scheduler with N concurrent requests and reports
aggregate decode throughput, which should climb with N up to a saturation point.

    uv run python scripts/smoke_batching.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

from crucible.backends import Final, SamplingParams
from crucible.batching import BatchedTextEngine, BatchScheduler, MLXBatchBackend

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MAX_TOKENS = 96


def main() -> None:
    print(f"loading {MODEL} ...")
    model, tok = load(MODEL)
    mx.eval(model.parameters())
    backend = MLXBatchBackend(model, tok, completion_batch_size=32)
    sched = BatchScheduler(backend, tok, make_sampler=make_sampler)
    engine = BatchedTextEngine(sched, tok, "primary", MODEL)

    def run_one(i: int) -> int:
        msgs = [{"role": "user", "content": f"Write three sentences about the number {i}."}]
        completion = 0
        for ev in engine.stream(msgs, SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7)):
            if isinstance(ev, Final):
                completion = ev.completion_tokens
        return completion

    print(f"{'N':>3}  {'tokens':>7}  {'wall_s':>7}  {'agg_tok/s':>10}")
    baseline = None
    for n in [1, 2, 4, 8]:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(run_one, range(n)))
        dt = time.time() - t0
        agg = sum(results) / dt
        if baseline is None:
            baseline = agg
        print(f"{n:>3}  {sum(results):>7}  {dt:>7.2f}  {agg:>10.1f}")

    final = sched.snapshot()
    print("\nscheduler counters:", final)
    sched.stop()
    print("\nOK: continuous batching ran; compare agg_tok/s growth from N=1 upward.")


if __name__ == "__main__":
    main()
