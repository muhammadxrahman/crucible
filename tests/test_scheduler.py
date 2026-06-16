"""M3: continuous-batching scheduler logic, driven by a fake backend (no GPU).

Verifies streaming, multi-request batching, stop sequences, finish reasons, and counters
without loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass

from crucible.backends import Delta, Final, SamplingParams
from crucible.batching import BatchScheduler

VOCAB = {1: "A", 2: "B", 3: "C", 9: "STOP"}


@dataclass
class FakeResp:
    uid: int
    token: int
    finish_reason: str | None


class FakeDetok:
    def reset(self) -> None:
        self._t: list[int] = []

    def add_token(self, t: int) -> None:
        self._t.append(t)

    @property
    def text(self) -> str:
        return "".join(VOCAB.get(t, "?") for t in self._t)

    def finalize(self) -> None:
        pass


class FakeTokenizer:
    @property
    def detokenizer(self) -> FakeDetok:
        return FakeDetok()  # a fresh detokenizer per request


class FakeBackend:
    """Emits scripted token lists, one content token per active uid per step."""

    def __init__(self, scripts: list[list[int]]):
        self._scripts = list(scripts)
        self._next_uid = 0
        self._seqs: dict[int, list[int]] = {}
        self.last_logits = None

    def insert(self, prompts, max_tokens, caches=None, samplers=None, logits_processors=None):  # noqa: ANN001
        self.last_logits = logits_processors
        uids = []
        for _ in prompts:
            uid = self._next_uid
            self._next_uid += 1
            self._seqs[uid] = list(self._scripts.pop(0))
            uids.append(uid)
        return uids

    def next_generated(self) -> list[FakeResp]:
        out = []
        for uid in list(self._seqs):
            toks = self._seqs[uid]
            if not toks:
                del self._seqs[uid]
                continue
            tok = toks.pop(0)
            finish = "length" if not toks else None  # last token is content + done
            out.append(FakeResp(uid, tok, finish))
            if finish:
                del self._seqs[uid]
        return out

    def remove(self, uids) -> None:  # noqa: ANN001
        for u in uids:
            self._seqs.pop(u, None)

    def kv_nbytes(self) -> int:
        return 0

    def close(self) -> None:
        pass


def make_scheduler(scripts: list[list[int]]) -> BatchScheduler:
    backend = FakeBackend(scripts)
    return BatchScheduler(lambda: (backend, FakeTokenizer(), 0), make_sampler=lambda **kw: None)


def drain(channel) -> tuple[str, Final]:  # noqa: ANN001
    text = ""
    while True:
        ev = channel.get(timeout=5)
        if isinstance(ev, Delta):
            text += ev.text
        elif isinstance(ev, Final):
            return text, ev


def test_single_request_streams_then_finalizes() -> None:
    sched = make_scheduler([[1, 2, 3]])
    ch = sched.submit([10, 11], SamplingParams(max_tokens=8))
    text, final = drain(ch)
    assert text == "ABC"
    assert final.completion_tokens == 3
    assert final.finish_reason == "length"
    assert final.prompt_tokens == 2
    sched.stop()


def test_multiple_requests_all_complete() -> None:
    sched = make_scheduler([[1, 1], [2, 2], [3, 3]])
    chans = [
        sched.submit([10], SamplingParams(max_tokens=8)),
        sched.submit([11], SamplingParams(max_tokens=8)),
        sched.submit([12], SamplingParams(max_tokens=8)),
    ]
    results = [drain(c)[0] for c in chans]
    assert results == ["AA", "BB", "CC"]
    assert sched.snapshot()["total_requests"] == 3
    assert sched.snapshot()["peak_batch_size"] >= 1
    sched.stop()


def test_stop_sequence_truncates() -> None:
    sched = make_scheduler([[1, 1, 9, 1]])  # "A","A","STOP","A"
    ch = sched.submit([10], SamplingParams(max_tokens=8, stop=["STOP"]))
    text, final = drain(ch)
    assert text == "AA"
    assert final.finish_reason == "stop"
    sched.stop()


class CrashingBackend(FakeBackend):
    def next_generated(self):
        raise RuntimeError("boom")


def test_worker_crash_fails_request_instead_of_hanging() -> None:
    backend = CrashingBackend([[1, 2]])
    sched = BatchScheduler(lambda: (backend, FakeTokenizer(), 0), make_sampler=lambda **kw: None)
    ch = sched.submit([10], SamplingParams(max_tokens=8))
    # Must not hang: a crashed worker delivers a terminal error Final.
    _, final = drain(ch)
    assert final.finish_reason == "error"
    # A request submitted after the crash also fails fast.
    _, final2 = drain(sched.submit([11], SamplingParams(max_tokens=8)))
    assert final2.finish_reason == "error"
    sched.stop()


def test_repetition_penalty_passed_to_backend() -> None:
    backend = FakeBackend([[1, 2]])
    sched = BatchScheduler(lambda: (backend, FakeTokenizer(), 0), make_sampler=lambda **kw: None)
    ch = sched.submit([10], SamplingParams(max_tokens=8, repetition_penalty=1.1))
    drain(ch)
    assert backend.last_logits  # non-empty logits processors were built and passed
    sched.stop()


def test_no_repetition_penalty_passes_none() -> None:
    backend = FakeBackend([[1, 2]])
    sched = BatchScheduler(lambda: (backend, FakeTokenizer(), 0), make_sampler=lambda **kw: None)
    ch = sched.submit([10], SamplingParams(max_tokens=8, repetition_penalty=1.0))
    drain(ch)
    assert backend.last_logits is None  # penalty disabled -> nothing passed
    sched.stop()


def test_loop_guard_stops_runaway_repetition() -> None:
    # A model stuck in a 3-token cycle for 36 tokens must be cut off, not run to the end.
    sched = make_scheduler([[1, 2, 3] * 12])
    ch = sched.submit([10], SamplingParams(max_tokens=200))
    _, final = drain(ch)
    assert final.finish_reason == "stop"
    assert final.completion_tokens < 36  # stopped mid-loop, not the full script
    sched.stop()


def test_loop_guard_disabled_runs_full() -> None:
    sched = make_scheduler([[1, 2, 3] * 12])
    ch = sched.submit([10], SamplingParams(max_tokens=200, loop_guard=False))
    _, final = drain(ch)
    assert final.completion_tokens == 36  # ran the whole script
    sched.stop()


def test_counters_track_requests() -> None:
    sched = make_scheduler([[1]])
    ch = sched.submit([10], SamplingParams(max_tokens=4))
    drain(ch)
    snap = sched.snapshot()
    assert snap["total_requests"] == 1
    assert snap["queue_depth"] == 0
    sched.stop()
