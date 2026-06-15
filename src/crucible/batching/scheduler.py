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
from crucible.backends.text import _apply_stop

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
        backend: BatchBackend,
        tokenizer,
        *,
        make_sampler,
        counters: Counters | None = None,
        clock=time.monotonic,
    ):
        self._backend = backend
        self._tok = tokenizer
        self._make_sampler = make_sampler
        self.counters = counters or Counters()
        self._clock = clock

        self._pending: deque[_Req] = deque()
        self._active: dict[int, _Req] = {}
        self._cv = threading.Condition()
        self._stop = False
        self._worker = threading.Thread(target=self._run, name="batch-scheduler", daemon=True)
        self._worker.start()

    # --- public API ---

    def submit(self, tokens: list[int], params: SamplingParams) -> queue.Queue:
        """Queue a request; return the channel its Delta/Final events arrive on."""
        req = _Req(
            tokens=tokens,
            params=params,
            sampler=self._make_sampler(temp=params.temperature, top_p=params.top_p),
            out=queue.Queue(),
            detok=self._tok.detokenizer,
        )
        req.detok.reset()
        with self._cv:
            self.counters.total_requests += 1
            self._pending.append(req)
            self.counters.queue_depth = len(self._pending)
            self._cv.notify()
        return req.out

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._worker.join(timeout=5)
        self._backend.close()

    def snapshot(self) -> dict:
        with self._cv:
            self.counters.batch_size = len(self._active)
            self.counters.queue_depth = len(self._pending)
        return self.counters.snapshot()

    # --- worker ---

    def _run(self) -> None:
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
            if not responses:
                if not self._active:
                    continue
                time.sleep(0)  # yield; batch is draining
                continue
            self._handle(responses)

    def _admit(self, reqs: list[_Req]) -> None:
        prompts = [r.tokens for r in reqs]
        max_tokens = [r.params.max_tokens for r in reqs]
        samplers = [r.sampler for r in reqs]
        uids = self._backend.insert(prompts, max_tokens, samplers=samplers)
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
