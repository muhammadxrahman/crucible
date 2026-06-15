"""M4: metrics registry records, aggregates, and exposes (no GPU)."""

from crucible.backends import Final
from crucible.observability import Metrics


class FakeManager:
    def __init__(self, stats: dict, resident: int = 1000, evict: int = 2):
        self._stats, self._r, self._e = stats, resident, evict

    def resident_stats(self) -> dict:
        return self._stats

    def resident_bytes(self) -> int:
        return self._r

    def evictions(self) -> int:
        return self._e


def _final(prefill: float, decode: float, comp: int = 20) -> Final:
    return Final(
        prompt_tokens=10,
        completion_tokens=comp,
        finish_reason="stop",
        prefill_tps=prefill,
        decode_tps=decode,
    )


def test_observe_and_expose_series() -> None:
    m = Metrics()
    m.observe_ttft("primary", 0.05)
    m.observe_final("primary", _final(300, 80), latency=0.5)
    text = m.expose().decode()
    assert "crucible_requests_total" in text
    assert "crucible_decode_tps" in text
    assert "crucible_ttft_seconds" in text
    assert 'model="primary"' in text

    s = m.summary()
    assert s["current"]["decode_tps"] == 80
    assert s["per_model"]["primary"]["prefill_tps"] == 300.0


def test_collect_aggregates_and_hit_ratio() -> None:
    m = Metrics()
    mgr = FakeManager(
        {
            "a": {"queue_depth": 2, "batch_size": 3, "kv_cache_bytes": 1000},
            "b": {"prefix_hits": 3, "prefix_misses": 1},
        }
    )
    m.collect(mgr)
    cur = m.summary()["current"]
    assert cur["queue_depth"] == 2
    assert cur["batch_size"] == 3
    assert cur["kv_cache_bytes"] == 1000
    assert cur["prefix_hit_ratio"] == 0.75
    assert cur["evictions"] == 2
    assert "crucible_prefix_hit_ratio" in m.expose().decode()


def test_ring_buffer_is_bounded() -> None:
    m = Metrics(history=5)
    mgr = FakeManager({})
    for _ in range(10):
        m.collect(mgr)
    assert len(m.summary()["history"]) == 5
