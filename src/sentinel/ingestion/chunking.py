"""Plain-text chunking with overlap. Deliberately dependency-free and easy to unit test."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    source: str


def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Split text into overlapping chunks on whitespace boundaries.

    Overlap must be smaller than chunk_size or the window would never advance.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    index = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(Chunk(text=" ".join(window), index=index, source=source))
        index += 1
        start += step
    return chunks
