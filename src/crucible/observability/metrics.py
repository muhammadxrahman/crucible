"""Metrics registry: native Prometheus exposition plus a JSON summary for the in-app view.

Prefill and decode throughput are always separate series, never blended (docs/hardware.md).
A per-app CollectorRegistry keeps state out of the global default registry so tests and
multiple app instances stay isolated. A bounded ring buffer holds recent samples for the
in-app dashboard's sparklines, so no external time-series database is needed.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

CONTENT_TYPE = CONTENT_TYPE_LATEST

_TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
_LAT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30)


def _aggregate(resident_stats: dict[str, dict]) -> dict:
    q = b = kv = hits = misses = 0
    for s in resident_stats.values():
        q += s.get("queue_depth", 0)
        b += s.get("batch_size", 0)
        kv += s.get("kv_cache_bytes", 0)
        hits += s.get("prefix_hits", 0)
        misses += s.get("prefix_misses", 0)
    ratio = hits / (hits + misses) if (hits + misses) else 0.0
    return {"queue_depth": q, "batch_size": b, "kv_cache_bytes": kv, "prefix_hit_ratio": ratio}


class Metrics:
    def __init__(self, *, history: int = 120):
        r = CollectorRegistry()
        self.reg = r
        self._requests = Counter(
            "crucible_requests_total", "Completed requests", ["model"], registry=r
        )
        self._tokens = Counter(
            "crucible_completion_tokens_total", "Completion tokens generated", ["model"], registry=r
        )
        self._ttft = Histogram(
            "crucible_ttft_seconds",
            "Time to first token",
            ["model"],
            buckets=_TTFT_BUCKETS,
            registry=r,
        )
        self._latency = Histogram(
            "crucible_request_latency_seconds",
            "End-to-end request latency",
            ["model"],
            buckets=_LAT_BUCKETS,
            registry=r,
        )
        self._g_prefill = Gauge(
            "crucible_prefill_tps", "Last prefill throughput", ["model"], registry=r
        )
        self._g_decode = Gauge(
            "crucible_decode_tps", "Last decode throughput", ["model"], registry=r
        )
        self._g_queue = Gauge("crucible_queue_depth", "Admitted but not yet running", registry=r)
        self._g_batch = Gauge("crucible_batch_size", "In-flight batched sequences", registry=r)
        self._g_resident = Gauge("crucible_resident_bytes", "Resident model memory", registry=r)
        self._g_evict = Gauge("crucible_evictions_total", "Model evictions", registry=r)
        self._g_hit = Gauge("crucible_prefix_hit_ratio", "Prefix cache hit ratio", registry=r)
        self._g_kv = Gauge("crucible_kv_cache_bytes", "KV cache bytes", registry=r)

        self._cur = {
            "prefill_tps": 0.0,
            "decode_tps": 0.0,
            "ttft_ms": 0.0,
            "queue_depth": 0,
            "batch_size": 0,
            "resident_bytes": 0,
            "evictions": 0,
            "prefix_hit_ratio": 0.0,
            "kv_cache_bytes": 0,
            "requests_total": 0,
        }
        self._per_model: dict[str, dict] = {}
        self._ring: deque[dict] = deque(maxlen=history)
        self._lock = threading.Lock()

    # --- record per-request ---

    def observe_ttft(self, model: str, dt: float) -> None:
        self._ttft.labels(model).observe(dt)
        with self._lock:
            self._cur["ttft_ms"] = dt * 1000

    def observe_final(self, model: str, final, latency: float) -> None:
        self._requests.labels(model).inc()
        self._tokens.labels(model).inc(final.completion_tokens)
        self._latency.labels(model).observe(latency)
        self._g_prefill.labels(model).set(final.prefill_tps)
        self._g_decode.labels(model).set(final.decode_tps)
        with self._lock:
            self._cur["prefill_tps"] = final.prefill_tps
            self._cur["decode_tps"] = final.decode_tps
            self._cur["requests_total"] += 1
            self._per_model[model] = {
                "prefill_tps": round(final.prefill_tps, 1),
                "decode_tps": round(final.decode_tps, 1),
            }

    # --- refresh pull-based gauges from live state ---

    def collect(self, manager) -> None:
        agg = _aggregate(manager.resident_stats())
        resident = manager.resident_bytes()
        evictions = manager.evictions()
        self._g_queue.set(agg["queue_depth"])
        self._g_batch.set(agg["batch_size"])
        self._g_kv.set(agg["kv_cache_bytes"])
        self._g_hit.set(agg["prefix_hit_ratio"])
        self._g_resident.set(resident)
        self._g_evict.set(evictions)
        with self._lock:
            self._cur.update(
                queue_depth=agg["queue_depth"],
                batch_size=agg["batch_size"],
                resident_bytes=resident,
                evictions=evictions,
                prefix_hit_ratio=round(agg["prefix_hit_ratio"], 3),
                kv_cache_bytes=agg["kv_cache_bytes"],
            )
            self._ring.append(
                {
                    "t": round(time.time(), 1),
                    "prefill_tps": round(self._cur["prefill_tps"], 1),
                    "decode_tps": round(self._cur["decode_tps"], 1),
                    "ttft_ms": round(self._cur["ttft_ms"], 1),
                    "batch_size": agg["batch_size"],
                    "queue_depth": agg["queue_depth"],
                }
            )

    # --- expose ---

    def expose(self) -> bytes:
        return generate_latest(self.reg)

    def summary(self) -> dict:
        with self._lock:
            return {
                "current": dict(self._cur),
                "per_model": dict(self._per_model),
                "history": list(self._ring),
            }
