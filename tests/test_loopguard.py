"""Loop guard: detect runaway repetition, ignore normal text."""

from crucible.backends.loopguard import LoopGuard


def test_detects_single_token_loop() -> None:
    g = LoopGuard(window=8, max_period=3)
    hits = [g.feed(5) for _ in range(8)]
    assert hits[-1] is True  # 8 identical tokens -> loop
    assert hits[6] is False  # not until the window fills


def test_detects_period_three_cycle() -> None:
    g = LoopGuard(window=12, max_period=4)
    hits = [g.feed(t) for t in [1, 2, 3] * 8]
    assert any(hits)


def test_ignores_varied_tokens() -> None:
    g = LoopGuard(window=8)
    assert not any(g.feed(t) for t in range(40))  # all distinct -> never a loop


def test_short_repeat_below_window_does_not_fire() -> None:
    g = LoopGuard(window=24)
    assert not any(g.feed(1) for _ in range(10))  # 10 < window


def test_reset_clears_history() -> None:
    g = LoopGuard(window=4, max_period=2)
    for _ in range(4):
        g.feed(7)
    g.reset()
    assert g.feed(7) is False
