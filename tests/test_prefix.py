"""M3: prefix cache matching, LRU bounding, and stats (pure logic, no MLX)."""

from crucible.batching import PrefixCache


def test_strict_prefix_hit_and_tokens_saved() -> None:
    c = PrefixCache(min_prefix=4)
    base = [1, 2, 3, 4, 5]
    c.store(base, "seedA")
    seed, matched = c.lookup(base + [6, 7])  # extends the stored prefix
    assert seed == "seedA"
    assert matched == 5
    assert c.stats.hits == 1
    assert c.stats.tokens_saved == 5


def test_exact_length_is_not_a_match() -> None:
    # A match must leave a suffix to generate from, so equal length does not hit.
    c = PrefixCache(min_prefix=2)
    c.store([1, 2, 3], "s")
    seed, matched = c.lookup([1, 2, 3])
    assert seed is None and matched == 0
    assert c.stats.misses == 1


def test_longest_prefix_wins() -> None:
    c = PrefixCache(min_prefix=2)
    c.store([1, 2], "short")
    c.store([1, 2, 3, 4], "long")
    seed, matched = c.lookup([1, 2, 3, 4, 5])
    assert seed == "long"
    assert matched == 4


def test_min_prefix_skips_short_prompts() -> None:
    c = PrefixCache(min_prefix=8)
    c.store([1, 2, 3], "tooshort")
    assert len(c) == 0


def test_lru_eviction_bounds_entries() -> None:
    c = PrefixCache(max_entries=2, min_prefix=1)
    c.store([1], "a")
    c.store([2], "b")
    c.lookup([1, 9])  # touch "a" so "b" is now LRU
    c.store([3], "c")  # exceeds cap -> evict LRU ("b")
    assert len(c) == 2
    assert c.lookup([2, 9])[0] is None  # "b" evicted
    assert c.lookup([1, 9])[0] == "a"
    assert c.lookup([3, 9])[0] == "c"


def test_no_match_returns_miss() -> None:
    c = PrefixCache(min_prefix=1)
    c.store([1, 2, 3], "s")
    seed, matched = c.lookup([9, 9, 9])
    assert seed is None and matched == 0
