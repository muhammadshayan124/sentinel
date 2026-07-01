"""Thin wrapper around Chroma with two collections: text chunks and image vectors.

Kept as a small seam (rather than calling chromadb directly all over the codebase) so the
backend can be swapped (e.g. for pgvector/Qdrant in production) without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentinel.config import settings
from sentinel.ingestion.chunking import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    distance: float


@dataclass(frozen=True)
class RetrievedImage:
    path: str
    distance: float


class VectorStore:
    def __init__(self, persist_dir: Path | None = None, in_memory: bool = False) -> None:
        import chromadb

        if in_memory:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=str(persist_dir or settings.chroma_dir))

        self._text_collection = self._client.get_or_create_collection("text_chunks")
        self._image_collection = self._client.get_or_create_collection(
            "images", metadata={"hnsw:space": "cosine"}
        )
        self._embedder = None

    def _text_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.text_embedding_model)
        return self._embedder

    def add_text_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._text_embedder().encode([c.text for c in chunks], convert_to_numpy=True)
        self._text_collection.add(
            ids=[f"{c.source}::{c.index}" for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "index": c.index} for c in chunks],
        )

    def add_image(self, path: str, vector: np.ndarray) -> None:
        self._image_collection.add(
            ids=[path],
            embeddings=[vector.tolist()],
            metadatas=[{"path": path}],
        )

    def query_text(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        vector = self._text_embedder().encode([query], convert_to_numpy=True)[0]
        result = self._text_collection.query(query_embeddings=[vector.tolist()], n_results=top_k)
        if not result["documents"] or not result["documents"][0]:
            return []
        out = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        ):
            out.append(RetrievedChunk(text=doc, source=meta["source"], distance=dist))
        return out

    def query_images(self, vector: np.ndarray, top_k: int = 3) -> list[RetrievedImage]:
        result = self._image_collection.query(query_embeddings=[vector.tolist()], n_results=top_k)
        if not result["ids"] or not result["ids"][0]:
            return []
        return [
            RetrievedImage(path=meta["path"], distance=dist)
            for meta, dist in zip(result["metadatas"][0], result["distances"][0], strict=True)
        ]

    def count(self) -> dict[str, int]:
        return {
            "text_chunks": self._text_collection.count(),
            "images": self._image_collection.count(),
        }
