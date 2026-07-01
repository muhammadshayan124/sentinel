"""Hybrid retriever: merges text-chunk hits and CLIP image hits for a single query."""

from __future__ import annotations

from dataclasses import dataclass

from sentinel.config import settings
from sentinel.ingestion.image_extractor import embed_text_query_for_image_search
from sentinel.retrieval.vector_store import RetrievedChunk, RetrievedImage, VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    text_chunks: list[RetrievedChunk]
    images: list[RetrievedImage]

    def as_context_string(self) -> str:
        parts = [f"[{c.source}]\n{c.text}" for c in self.text_chunks]
        if self.images:
            parts.append(
                "Relevant images found (not shown as text): "
                + ", ".join(img.path for img in self.images)
            )
        return "\n\n---\n\n".join(parts) if parts else "No relevant context found."


def retrieve(query: str, store: VectorStore, top_k: int | None = None) -> RetrievalResult:
    k = top_k or settings.retrieval_top_k
    text_chunks = store.query_text(query, top_k=k)

    try:
        image_vector = embed_text_query_for_image_search(query)
        images = store.query_images(image_vector, top_k=min(3, k))
    except Exception:
        images = []

    return RetrievalResult(text_chunks=text_chunks, images=images)
