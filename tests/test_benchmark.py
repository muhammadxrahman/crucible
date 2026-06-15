"""M4: benchmark harness aggregation and Markdown rendering (no GPU)."""

from crucible.backends import Delta, Final, SamplingParams
from crucible.benchmark import run_case, to_markdown


class FakeEngine:
    def stream(self, messages: list[dict], params: SamplingParams):
        for t in ["a", "b", "c"]:
            yield Delta(t)
        yield Final(
            prompt_tokens=5,
            completion_tokens=3,
            finish_reason="length",
            prefill_tps=200.0,
            decode_tps=60.0,
        )


def test_run_case_aggregates_across_concurrency() -> None:
    res = run_case(FakeEngine(), [{"role": "user", "content": "hi"}], max_tokens=8, concurrency=4)
    assert res.concurrency == 4
    assert res.completion_tokens == 12  # 3 tokens x 4 requests
    assert res.prompt_tokens == 5
    assert res.prefill_tps == 200.0
    assert res.decode_tps == 60.0
    assert res.agg_decode_tps > 0


def test_markdown_keeps_prefill_and_decode_separate() -> None:
    res = run_case(FakeEngine(), [{"role": "user", "content": "hi"}], 8, 1)
    res.model = "tiny"
    md = to_markdown([res], {"prompt": "hi", "max_tokens": 8})
    assert "prefill tok/s" in md
    assert "decode tok/s" in md
    assert "## tiny" in md
    assert "Aggregate decode throughput" in md
