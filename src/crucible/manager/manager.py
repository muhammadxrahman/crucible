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
    state: str  # "resident" | "loading" | "available"
    pinned: bool
    resident_bytes: int
    path: str = ""
    error: str | None = None  # message from the last failed load, if any


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
        # In-flight loads: served_name -> Event signalled when the load settles. The actual
        # load runs with the lock released, so other manager calls stay responsive (e.g. the UI
        # keeps polling /v1/models while a large model downloads).
        self._loading: dict[str, threading.Event] = {}
        self._errors: dict[str, str] = {}  # last load error per served_name

    # --- public queries ---

    def list_status(self) -> list[ModelStatus]:
        with self._lock:
            return [self._status(name) for name in self._entries]

    def resident_models(self) -> list[str]:
        with self._lock:
            return list(self._resident)

    def first_of_type(self, model_type: str) -> str | None:
        """The served_name of the first registered model of a given type, if any."""
        with self._lock:
            return next((n for n, e in self._entries.items() if e.type == model_type), None)

    def status(self, name: str) -> ModelStatus:
        with self._lock:
            if name not in self._entries:
                raise UnknownModel(name)
            return self._status(name)

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
        return self._ensure_resident(name).engine

    def entry(self, name: str) -> ModelEntry:
        with self._lock:
            if name not in self._entries:
                raise UnknownModel(name)
            return self._entries[name]

    # --- admin ---

    def register(self, entry: ModelEntry) -> None:
        """Add a model to the registry at runtime (UI 'add model'). Idempotent guard on
        served_name; loading is a separate step (see load_async)."""
        with self._lock:
            if entry.served_name in self._entries:
                raise ValueError(f"served_name already registered: {entry.served_name}")
            self._entries[entry.served_name] = entry
            if entry.pin:
                self._config_pins.add(entry.served_name)

    def load(self, name: str) -> ModelStatus:
        self._ensure_resident(name)
        return self.status(name)

    def load_async(self, name: str) -> ModelStatus:
        """Kick off a load on a background thread and return immediately. The returned status
        reports 'loading'; callers (the UI) poll list_status until it flips to 'resident'."""
        kind, ev = self._claim(name)
        if kind == "resident":
            return self.status(name)
        if kind == "owner":
            threading.Thread(
                target=self._safe_run_load, args=(name, ev), daemon=True, name=f"load-{name}"
            ).start()
        return self.status(name)

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
        if pinned and not self.is_resident(name):
            self._ensure_resident(name)
        with self._lock:
            if name in self._resident:
                self._resident[name].pinned = pinned
            return self._status(name)

    def warmup(self) -> list[tuple[str, str]]:
        """Eagerly load config-pinned models. A model that fails to load (download error,
        out of memory) is skipped with its error returned, not raised, so the server still
        starts and serves the rest; that model loads lazily on first request."""
        failures: list[tuple[str, str]] = []
        for name in sorted(self._config_pins):
            if self.is_resident(name):
                continue
            try:
                self._ensure_resident(name)
            except Exception as e:  # noqa: BLE001 - one bad model must not sink the server
                failures.append((name, str(e)))
        return failures

    def sweep_ttl(self) -> list[str]:
        with self._lock:
            return self._sweep_ttl()

    # --- internals (call while holding the lock) ---

    def _resident_bytes(self) -> int:
        return sum(r.nbytes for r in self._resident.values())

    def _state(self, name: str) -> str:
        if name in self._resident:
            return "resident"
        if name in self._loading:
            return "loading"
        return "available"

    def _status(self, name: str) -> ModelStatus:
        entry = self._entries[name]
        r = self._resident.get(name)
        return ModelStatus(
            served_name=name,
            type=entry.type,
            state=self._state(name),
            pinned=r.pinned if r else (name in self._config_pins),
            resident_bytes=r.nbytes if r else 0,
            path=entry.path,
            error=self._errors.get(name),
        )

    # --- loading: claim ownership under the lock, do the heavy load with it released ---

    def _ensure_resident(self, name: str) -> _Resident:
        """Block until `name` is resident, returning it. At most one thread loads a given
        model; concurrent callers wait on its Event rather than re-loading or blocking the lock."""
        kind, val = self._claim(name)
        if kind == "resident":
            return val
        if kind == "owner":
            return self._run_load(name, val)
        val.wait()  # waiter
        with self._lock:
            r = self._resident.get(name)
            if r is not None:
                r.last_used = self._clock()
                return r
            err = self._errors.get(name, "load failed")
        raise RuntimeError(f"{name}: {err}")

    def _claim(self, name: str) -> tuple[str, object]:
        """Returns ("resident", _Resident) | ("owner", Event) | ("wait", Event).
        The owner must call _run_load to settle the Event."""
        with self._lock:
            self._sweep_ttl()
            if name not in self._entries:
                raise UnknownModel(name)
            r = self._resident.get(name)
            if r is not None:
                r.last_used = self._clock()
                return "resident", r
            ev = self._loading.get(name)
            if ev is not None:
                return "wait", ev
            ev = threading.Event()
            self._loading[name] = ev
            # Free room before the load on single-resident profiles (small Macs), preserving the
            # original pre-load eviction semantics.
            if self._runtime.single_resident:
                for other in list(self._resident):
                    if not self._resident[other].pinned:
                        self._evict(other)
            return "owner", ev

    def _run_load(self, name: str, ev: threading.Event) -> _Resident:
        """Owner path: run the loader OUTSIDE the lock, then register and signal waiters."""
        try:
            engine, nbytes = self._loader(self._entries[name])
        except Exception as e:
            with self._lock:
                self._errors[name] = str(e)
                self._loading.pop(name, None)
                ev.set()
            raise
        with self._lock:
            r = _Resident(
                entry=self._entries[name],
                engine=engine,
                nbytes=nbytes,
                pinned=name in self._config_pins,
                last_used=self._clock(),
            )
            self._resident[name] = r
            self._errors.pop(name, None)
            self._evict_to_fit(protect=name)
            self._loading.pop(name, None)
            ev.set()
            return r

    def _safe_run_load(self, name: str, ev: threading.Event) -> None:
        """Background-thread entrypoint for load_async; the error is recorded in _errors and
        surfaced via status, so it must not propagate out of the thread."""
        try:
            self._run_load(name, ev)
        except Exception:  # noqa: BLE001 - already recorded in _errors
            pass

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
