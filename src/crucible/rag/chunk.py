"""Split document text into overlapping word windows for embedding."""

from __future__ import annotations


def chunk_text(text: str, *, size: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(size - overlap, 1)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks
