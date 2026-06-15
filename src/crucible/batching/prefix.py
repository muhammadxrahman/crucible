"""Prefix KV-cache: reuse the KV state of a shared prompt prefix to skip its prefill.

Shared system prompts and RAG contexts repeat across requests; caching their KV state
lets a new request prefill only the differing suffix. Start simple (longest-match over a
bounded, LRU set of stored prefixes); paging is a later concern (docs/roadmap.md M3).

Pure storage of opaque seeds: the engine decides how to build and rebuild the KV state,
so this is testable without MLX.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class PrefixStats:
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0

    def snapshot(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "tokens_saved": self.tokens_saved}


class PrefixCache:
    def __init__(self, *, max_entries: int = 16, min_prefix: int = 16):
        self._max_entries = max_entries
        self._min_prefix = min_prefix
        self._store: OrderedDict[tuple[int, ...], Any] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = PrefixStats()

    def lookup(self, tokens: list[int]) -> tuple[Any | None, int]:
        """Return (seed, matched_len) for the longest stored prefix of `tokens`.

        A match must be a strict prefix (shorter than `tokens`) so there is a suffix to
        generate from. Records hit/miss stats.
        """
        with self._lock:
            best_key: tuple[int, ...] | None = None
            for key in self._store:
                n = len(key)
                if n < len(tokens) and (best_key is None or n > len(best_key)):
                    if tuple(tokens[:n]) == key:
                        best_key = key
            if best_key is None:
                self.stats.misses += 1
                return None, 0
            self._store.move_to_end(best_key)
            self.stats.hits += 1
            self.stats.tokens_saved += len(best_key)
            return self._store[best_key], len(best_key)

    def store(self, tokens: list[int], seed: Any) -> None:
        if len(tokens) < self._min_prefix:
            return
        key = tuple(tokens)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return
            self._store[key] = seed
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)  # evict LRU

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
