"""M2 acceptance: the model manager's residency, eviction, pin, and TTL policy.

Driven by a fake loader and a manual clock, so the logic is verified without a GPU or
any model download.
"""

from __future__ import annotations

import pytest

from crucible.config import Registry
from crucible.manager import ModelManager, RuntimeProfile, UnknownModel


class FakeEngine:
    def __init__(self, name: str):
        self.served_name = name
        self.closed = False

    def close(self) -> None:
        self.closed = True


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def make_loader(sizes: dict[str, int]):
    engines: dict[str, FakeEngine] = {}

    def load(entry):  # noqa: ANN001
        e = FakeEngine(entry.served_name)
        engines[entry.served_name] = e
        return e, sizes.get(entry.served_name, 100)

    return load, engines


def registry(models: list[dict]) -> Registry:
    return Registry.model_validate({"models": models})


def runtime(ceiling_bytes: int, single: bool = False) -> RuntimeProfile:
    return RuntimeProfile(
        name="test",
        ceiling_bytes=ceiling_bytes,
        single_resident=single,
        default_context=8192,
        kv_bits=8,
        vision=True,
    )


def lm(name: str, **kw) -> dict:
    return {"path": f"fake/{name}", "type": "lm", "served_name": name, **kw}


def test_acquire_loads_and_routes() -> None:
    loader, engines = make_loader({"a": 100})
    m = ModelManager(registry([lm("a")]), runtime(10_000), loader)
    eng = m.acquire("a")
    assert eng is engines["a"]
    assert m.is_resident("a")
    assert m.resident_bytes() == 100


def test_unknown_model_raises() -> None:
    loader, _ = make_loader({})
    m = ModelManager(registry([lm("a")]), runtime(10_000), loader)
    with pytest.raises(UnknownModel):
        m.acquire("ghost")


def test_lru_eviction_when_load_exceeds_ceiling() -> None:
    loader, engines = make_loader({"a": 100, "b": 100, "c": 100})
    clock = Clock()
    m = ModelManager(registry([lm("a"), lm("b"), lm("c")]), runtime(250), loader, clock=clock)

    clock.t = 1
    m.acquire("a")
    clock.t = 2
    m.acquire("b")
    clock.t = 3
    m.acquire("c")  # 300 > 250 -> evict LRU (a)

    assert set(m.resident_models()) == {"b", "c"}
    assert engines["a"].closed is True
    assert m.resident_bytes() == 200
    assert m.evictions() == 1


def test_recently_used_survives_eviction() -> None:
    loader, _ = make_loader({"a": 100, "b": 100, "c": 100, "d": 100})
    clock = Clock()
    models = [lm("a"), lm("b"), lm("c"), lm("d")]
    m = ModelManager(registry(models), runtime(250), loader, clock=clock)

    clock.t = 1
    m.acquire("a")
    clock.t = 2
    m.acquire("b")  # a,b resident
    clock.t = 3
    m.acquire("a")  # touch a -> b now LRU
    clock.t = 4
    m.acquire("c")  # 300>250 -> evict b (LRU), not a

    assert set(m.resident_models()) == {"a", "c"}


def test_pinned_model_never_evicted() -> None:
    loader, engines = make_loader({"keep": 200, "b": 100, "c": 100})
    clock = Clock()
    models = [lm("keep", pin=True), lm("b"), lm("c")]
    m = ModelManager(registry(models), runtime(250), loader, clock=clock)

    m.warmup()  # loads pinned "keep" (200)
    clock.t = 1
    m.acquire("b")  # 300>250 -> must evict unpinned, not "keep"
    clock.t = 2
    m.acquire("c")

    assert "keep" in m.resident_models()
    assert engines["keep"].closed is False


def test_ttl_idle_eviction() -> None:
    loader, engines = make_loader({"a": 100})
    clock = Clock()
    m = ModelManager(registry([lm("a", ttl_seconds=600)]), runtime(10_000), loader, clock=clock)

    clock.t = 100
    m.acquire("a")
    clock.t = 200  # 100s idle, under ttl
    assert m.sweep_ttl() == []
    clock.t = 800  # 700s idle, over ttl
    assert m.sweep_ttl() == ["a"]
    assert engines["a"].closed is True
    assert not m.is_resident("a")


def test_pinned_ignores_ttl() -> None:
    loader, _ = make_loader({"a": 100})
    clock = Clock()
    m = ModelManager(
        registry([lm("a", pin=True, ttl_seconds=1)]), runtime(10_000), loader, clock=clock
    )
    m.warmup()
    clock.t = 10_000
    assert m.sweep_ttl() == []
    assert m.is_resident("a")


def test_single_resident_evicts_previous() -> None:
    loader, engines = make_loader({"a": 100, "b": 100})
    m = ModelManager(registry([lm("a"), lm("b")]), runtime(10_000, single=True), loader)

    m.acquire("a")
    m.acquire("b")  # single_resident: a must be evicted first

    assert m.resident_models() == ["b"]
    assert engines["a"].closed is True


def test_warmup_resilient_to_a_failing_model() -> None:
    # A pinned model that fails to load must not sink warmup; the rest still load.
    engines: dict[str, FakeEngine] = {}

    def loader(entry):  # noqa: ANN001
        if entry.served_name == "bad":
            raise RuntimeError("download failed")
        e = FakeEngine(entry.served_name)
        engines[entry.served_name] = e
        return e, 100

    m = ModelManager(registry([lm("good", pin=True), lm("bad", pin=True)]), runtime(10_000), loader)
    failures = m.warmup()
    assert m.is_resident("good")
    assert not m.is_resident("bad")
    assert [name for name, _ in failures] == ["bad"]


def test_admin_unload_and_pin() -> None:
    loader, _ = make_loader({"a": 100})
    m = ModelManager(registry([lm("a")]), runtime(10_000), loader)

    m.acquire("a")
    s = m.unload("a")
    assert s.state == "available"
    assert not m.is_resident("a")

    s = m.pin("a")  # pin loads if absent
    assert s.pinned is True
    assert m.is_resident("a")


def test_list_status_reports_residency_and_pin() -> None:
    loader, _ = make_loader({"a": 100, "b": 100})
    m = ModelManager(registry([lm("a", pin=True), lm("b")]), runtime(10_000), loader)
    m.acquire("b")

    by_name = {s.served_name: s for s in m.list_status()}
    assert by_name["a"].pinned is True
    assert by_name["a"].state == "available"  # pinned but not yet warmed
    assert by_name["b"].state == "resident"
    assert by_name["b"].resident_bytes == 100
