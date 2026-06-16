"""Continuous-batching scheduler.

A single worker thread owns one BatchGenerator for one model. Concurrent requests are
admitted through a queue and folded into the running decode batch (in-flight batching)
rather than serialized. Each request streams its own tokens back through a channel.

All MLX work happens on the worker thread; the scheduler is the only writer to the
batch, which keeps the single-threaded MLX evaluation model intact.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from crucible.backends.base import Delta, Final, SamplingParams
from crucible.backends.loopguard import LoopGuard
from crucible.backends.text import _apply_stop, _logits_processors

from .backend import BatchBackend


@dataclass
class Counters:
    total_requests: int = 0
    batch_size: int = 0  # current in-flight sequences
    queue_depth: int = 0  # admitted but not yet inserted
    peak_batch_size: int = 0

    def snapshot(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "batch_size": self.batch_size,
            "queue_depth": self.queue_depth,
            "peak_batch_size": self.peak_batch_size,
        }


@dataclass
class _Req:
    tokens: list[int]  # full prompt tokens
    params: SamplingParams
    sampler: Any
    logits: Any  # per-request logits processors (repetition penalty), or None
    guard: Any  # per-request LoopGuard, or None
    out: queue.Queue
    detok: Any
    uid: int | None = None
    emitted: str = ""
    completion_tokens: int = 0
    t_submit: float = 0.0
    t_first: float = 0.0


class BatchScheduler:
    def __init__(
        self,
        build,
        *,
        make_sampler,
        counters: Counters | None = None,
        clock=time.monotonic,
    ):
        # build() runs on the worker thread and returns (backend, tokenizer, resident_bytes).
        # MLX arrays are thread-affine: the model must be loaded on the same thread that
        # evaluates the batch, or eval fails with "no Stream(gpu, N) in current thread".
        self._build = build
        self._backend: BatchBackend | None = None
        self._tok = None
        self._nbytes = 0
        self._ready = threading.Event()
        self._init_error: BaseException | None = None
        self._make_sampler = make_sampler
        self.counters = counters or Counters()
        self._clock = clock

        self._pending: deque[_Req] = deque()
        self._active: dict[int, _Req] = {}
        self._cv = threading.Condition()
        self._stop = False
        self._error: BaseException | None = None
        self._kv_bytes = 0
        self._worker = threading.Thread(target=self._run, name="batch-scheduler", daemon=True)
        self._worker.start()

    # --- public API ---

    def wait_ready(self, timeout: float | None = None) -> int:
        """Block until the worker has loaded the model; return its resident bytes.

        Default is no timeout: a first-run model download can take a long time and must not
        be killed mid-transfer. The worker always sets the ready event (on success or
        failure), so this returns as soon as loading finishes either way.
        """
        if not self._ready.wait(timeout):
            raise TimeoutError("batch scheduler did not become ready")
        if self._init_error is not None:
            raise self._init_error
        return self._nbytes

    @property
    def tokenizer(self):
        return self._tok

    def submit(self, tokens: list[int], params: SamplingParams) -> queue.Queue:
        """Queue a request; return the channel its Delta/Final events arrive on."""
        self._ready.wait()
        out: queue.Queue = queue.Queue()
        if self._error is not None or self._init_error is not None:
            out.put(self._error_final())  # worker crashed; fail fast instead of hanging
            return out
        detok = self._tok.detokenizer
        detok.reset()
        req = _Req(
            tokens=tokens,
            params=params,
            sampler=self._make_sampler(temp=params.temperature, top_p=params.top_p),
            logits=_logits_processors(params),
            guard=LoopGuard() if params.loop_guard else None,
            out=out,
            detok=detok,
        )
        with self._cv:
            self.counters.total_requests += 1
            self._pending.append(req)
            self.counters.queue_depth = len(self._pending)
            self._cv.notify()
        return out

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._worker.join(timeout=10)  # worker closes the backend on its own thread

    def snapshot(self) -> dict:
        with self._cv:
            self.counters.batch_size = len(self._active)
            self.counters.queue_depth = len(self._pending)
        return self.counters.snapshot()

    def kv_nbytes(self) -> int:
        return self._kv_bytes  # cached; updated by the worker thread

    # --- worker ---

    def _run(self) -> None:
        try:
            self._backend, self._tok, self._nbytes = self._build()
        except BaseException as exc:  # noqa: BLE001
            self._init_error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            self._loop()
        except BaseException as exc:  # noqa: BLE001
            self._fail_all(exc)
        finally:
            if self._backend is not None:
                self._backend.close()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._stop and not self._pending and not self._active:
                    self._cv.wait()
                if self._stop:
                    return
                new = list(self._pending)
                self._pending.clear()
                self.counters.queue_depth = 0
            if new:
                self._admit(new)
            responses = self._backend.next_generated()
            self._kv_bytes = self._backend.kv_nbytes()
            if not responses:
                if not self._active:
                    continue
                time.sleep(0)  # yield; batch is draining
                continue
            self._handle(responses)

    def _fail_all(self, exc: BaseException) -> None:
        """A worker failure must terminate in-flight and queued requests, not hang them."""
        with self._cv:
            self._error = exc
            for r in list(self._active.values()) + list(self._pending):
                r.out.put(self._error_final())
            self._active.clear()
            self._pending.clear()

    @staticmethod
    def _error_final() -> Final:
        return Final(prompt_tokens=0, completion_tokens=0, finish_reason="error")

    def _admit(self, reqs: list[_Req]) -> None:
        prompts = [r.tokens for r in reqs]
        max_tokens = [r.params.max_tokens for r in reqs]
        samplers = [r.sampler for r in reqs]
        logits = [r.logits or [] for r in reqs]
        uids = self._backend.insert(
            prompts,
            max_tokens,
            samplers=samplers,
            logits_processors=logits if any(r.logits for r in reqs) else None,
        )
        now = self._clock()
        with self._cv:
            for uid, r in zip(uids, reqs, strict=True):
                r.uid = uid
                r.t_submit = now
                self._active[uid] = r
            self.counters.batch_size = len(self._active)
            self.counters.peak_batch_size = max(self.counters.peak_batch_size, len(self._active))

    def _handle(self, responses: list) -> None:
        for resp in responses:
            r = self._active.get(resp.uid)
            if r is None:
                continue
            done = resp.finish_reason is not None
            if resp.finish_reason != "stop":  # a real content token
                if r.completion_tokens == 0:
                    r.t_first = self._clock()
                r.detok.add_token(resp.token)
                r.completion_tokens += 1
                if self._emit(r, done):  # string stop sequence hit
                    self._finalize(r, "stop")
                    self._retire(r)
                    continue
                if r.guard is not None and r.guard.feed(resp.token):  # runaway repetition
                    self._finalize(r, "stop")
                    self._retire(r)
                    continue
            if done:
                self._finalize(r, resp.finish_reason)
                self._retire(r)

    def _emit(self, r: _Req, done: bool) -> bool:
        """Emit newly available text up to any stop sequence. Return True if stopped."""
        if done:
            r.detok.finalize()
        full = r.detok.text
        piece, stopped = _apply_stop(full[len(r.emitted) :], r.emitted, r.params.stop)
        if piece:
            r.out.put(Delta(piece))
            r.emitted += piece
        return stopped

    def _finalize(self, r: _Req, finish_reason: str | None) -> None:
        now = self._clock()
        prefill_dt = max(r.t_first - r.t_submit, 1e-6) if r.t_first else 1e-6
        decode_dt = max(now - r.t_first, 1e-6) if r.t_first else 1e-6
        r.out.put(
            Final(
                prompt_tokens=len(r.tokens),
                completion_tokens=r.completion_tokens,
                finish_reason=finish_reason or "length",
                prefill_tps=len(r.tokens) / prefill_dt,
                decode_tps=r.completion_tokens / decode_dt,
            )
        )

    def _retire(self, r: _Req) -> None:
        if r.uid is not None and r.uid in self._active:
            with self._cv:
                self._active.pop(r.uid, None)
                self.counters.batch_size = len(self._active)
            try:
                self._backend.remove([r.uid])
            except Exception:
                pass
