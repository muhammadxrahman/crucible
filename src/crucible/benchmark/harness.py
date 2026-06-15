"""Benchmark harness: measure prefill and decode throughput, TTFT, and throughput vs
concurrency by driving an engine through its .stream() interface.

Engine-driven so it works against any backend and is testable with a fake (no GPU).
Prefill and decode are reported separately, never blended (docs/hardware.md).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from statistics import mean

from crucible.backends import Delta, Final, SamplingParams


@dataclass
class CaseResult:
    concurrency: int
    prompt_tokens: int
    completion_tokens: int
    prefill_tps: float
    decode_tps: float
    ttft_ms: float
    agg_decode_tps: float
    wall_s: float
    model: str = ""


def run_case(
    engine,
    messages: list[dict],
    max_tokens: int,
    concurrency: int,
    *,
    clock=time.perf_counter,
) -> CaseResult:
    def one(_: int) -> dict:
        start = clock()
        ttft = None
        comp = prompt = 0
        prefill = decode = 0.0
        for ev in engine.stream(messages, SamplingParams(max_tokens=max_tokens, temperature=0.0)):
            if isinstance(ev, Delta) and ttft is None:
                ttft = clock() - start
            elif isinstance(ev, Final):
                comp, prompt = ev.completion_tokens, ev.prompt_tokens
                prefill, decode = ev.prefill_tps, ev.decode_tps
        return {"ttft": ttft or 0.0, "comp": comp, "prompt": prompt, "pf": prefill, "dc": decode}

    t0 = clock()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        rs = list(ex.map(one, range(concurrency)))
    wall = max(clock() - t0, 1e-9)
    n = len(rs)
    return CaseResult(
        concurrency=concurrency,
        prompt_tokens=sum(r["prompt"] for r in rs) // n,
        completion_tokens=sum(r["comp"] for r in rs),
        prefill_tps=mean(r["pf"] for r in rs),
        decode_tps=mean(r["dc"] for r in rs),
        ttft_ms=1000 * mean(r["ttft"] for r in rs),
        agg_decode_tps=sum(r["comp"] for r in rs) / wall,
        wall_s=wall,
    )


@dataclass
class BenchRun:
    prompt: str
    max_tokens: int
    concurrency: list[int]
    results: list[CaseResult] = field(default_factory=list)
