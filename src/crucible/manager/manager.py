"""The model manager: lifecycle, residency, eviction, and routing.

Owns every loaded model. Enforces the profile memory ceiling so the unified memory pool
is never oversubscribed, evicts by LRU, and honors pins and per-model TTL. Decoupled
from concrete backends through an injected loader so the policy is tested without a GPU.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from crucible.config import ModelEntry, Registry

from .runtime import RuntimeProfile

# Builds an engine for an entry and reports its resident-byte footprint.
Loader = Callable[[ModelEntry], tuple[object, int]]


class UnknownModel(KeyError):
    """Requested served_name is not in the registry."""


@dataclass
class _Resident:
    entry: ModelEntry
    engine: object
    nbytes: int
    pinned: bool
    last_used: float


@dataclass(frozen=True)
class ModelStatus:
    served_name: str
    type: str
    state: str  # "resident" | "available"
    pinned: bool
    resident_bytes: int


class ModelManager:
    def __init__(
        self,
        registry: Registry,
        runtime: RuntimeProfile,
        loader: Loader,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._entries: dict[str, ModelEntry] = {m.served_name: m for m in registry.models}
        self._config_pins = {m.served_name for m in registry.models if m.pin}
        self._runtime = runtime
        self._loader = loader
        self._clock = clock
        self._resident: dict[str, _Resident] = {}
        self._evictions = 0
        self._lock = threading.RLock()

    # --- public queries ---

    def list_status(self) -> list[ModelStatus]:
        with self._lock:
            out = []
            for name, entry in self._entries.items():
                r = self._resident.get(name)
                out.append(
                    ModelStatus(
                        served_name=name,
                        type=entry.type,
                        state="resident" if r else "available",
                        pinned=r.pinned if r else (name in self._config_pins),
                        resident_bytes=r.nbytes if r else 0,
                    )
                )
            return out

    def resident_models(self) -> list[str]:
        with self._lock:
            return list(self._resident)

    def first_of_type(self, model_type: str) -> str | None:
        """The served_name of the first registered model of a given type, if any."""
        return next((n for n, e in self._entries.items() if e.type == model_type), None)

    def resident_bytes(self) -> int:
        with self._lock:
            return self._resident_bytes()

    def evictions(self) -> int:
        with self._lock:
            return self._evictions

    def resident_stats(self) -> dict[str, dict]:
        """Per-resident-model engine stats (queue depth, batch size, prefix hits, ...)."""
        with self._lock:
            out: dict[str, dict] = {}
            for name, r in self._resident.items():
                fn = getattr(r.engine, "stats", None)
                out[name] = fn() if callable(fn) else {}
            return out

    def is_resident(self, name: str) -> bool:
        with self._lock:
            return name in self._resident

    # --- routing ---

    def acquire(self, name: str) -> object:
        """Ensure the model is resident, mark it most-recently-used, return its engine."""
        with self._lock:
            self._sweep_ttl()
            if name not in self._entries:
                raise UnknownModel(name)
            r = self._resident.get(name)
            if r is not None:
                r.last_used = self._clock()
                return r.engine
            return self._load(name).engine

    def entry(self, name: str) -> ModelEntry:
        with self._lock:
            if name not in self._entries:
                raise UnknownModel(name)
            return self._entries[name]

    # --- admin ---

    def load(self, name: str) -> ModelStatus:
        with self._lock:
            if name not in self._entries:
                raise UnknownModel(name)
            self.acquire(name)
            return self._status(name)

    def unload(self, name: str) -> ModelStatus:
        with self._lock:
            if name not in self._entries:
                raise UnknownModel(name)
            if name in self._resident:
                self._evict(name)
            return self._status(name)

    def pin(self, name: str, pinned: bool = True) -> ModelStatus:
        with self._lock:
            if name not in self._entries:
                raise UnknownModel(name)
            if pinned and name not in self._resident:
                self._load(name)
            if name in self._resident:
                self._resident[name].pinned = pinned
            return self._status(name)

    def warmup(self) -> list[tuple[str, str]]:
        """Eagerly load config-pinned models. A model that fails to load (download error,
        out of memory) is skipped with its error returned, not raised, so the server still
        starts and serves the rest; that model loads lazily on first request."""
        failures: list[tuple[str, str]] = []
        with self._lock:
            for name in self._config_pins:
                if name in self._resident:
                    continue
                try:
                    self._load(name)
                except Exception as e:  # noqa: BLE001 - one bad model must not sink the server
                    failures.append((name, str(e)))
        return failures

    def sweep_ttl(self) -> list[str]:
        with self._lock:
            return self._sweep_ttl()

    # --- internals (call while holding the lock) ---

    def _resident_bytes(self) -> int:
        return sum(r.nbytes for r in self._resident.values())

    def _status(self, name: str) -> ModelStatus:
        entry = self._entries[name]
        r = self._resident.get(name)
        return ModelStatus(
            served_name=name,
            type=entry.type,
            state="resident" if r else "available",
            pinned=r.pinned if r else (name in self._config_pins),
            resident_bytes=r.nbytes if r else 0,
        )

    def _load(self, name: str) -> _Resident:
        entry = self._entries[name]
        if self._runtime.single_resident:
            for other in list(self._resident):
                if not self._resident[other].pinned:
                    self._evict(other)
        engine, nbytes = self._loader(entry)
        r = _Resident(
            entry=entry,
            engine=engine,
            nbytes=nbytes,
            pinned=name in self._config_pins,
            last_used=self._clock(),
        )
        self._resident[name] = r
        self._evict_to_fit(protect=name)
        return r

    def _evict_to_fit(self, protect: str) -> None:
        while self._resident_bytes() > self._runtime.ceiling_bytes:
            victim = self._lru_evictable(protect)
            if victim is None:
                break  # only the protected and pinned models remain
            self._evict(victim)

    def _lru_evictable(self, protect: str) -> str | None:
        cands = [
            (r.last_used, name)
            for name, r in self._resident.items()
            if name != protect and not r.pinned
        ]
        return min(cands)[1] if cands else None

    def _evict(self, name: str) -> None:
        r = self._resident.pop(name)
        self._evictions += 1
        close = getattr(r.engine, "close", None)
        if callable(close):
            close()

    def _sweep_ttl(self) -> list[str]:
        now = self._clock()
        evicted = []
        for name in list(self._resident):
            r = self._resident[name]
            ttl = r.entry.ttl_seconds
            if ttl is not None and not r.pinned and (now - r.last_used) > ttl:
                self._evict(name)
                evicted.append(name)
        return evicted
