"""Orchestrates ingestion: walk a directory, extract text/images, chunk, embed, and store."""

from __future__ import annotations

from pathlib import Path

import structlog

from sentinel.config import settings
from sentinel.ingestion.chunking import chunk_text
from sentinel.ingestion.image_extractor import embed_image
from sentinel.ingestion.text_extractor import IMAGE_SUFFIXES, extract_text
from sentinel.retrieval.vector_store import VectorStore

logger = structlog.get_logger(__name__)


def ingest_directory(directory: Path, store: VectorStore) -> dict[str, int]:
    """Ingest every supported file under `directory` into `store`.

    Returns a summary count of {"text_chunks": N, "images": N, "files_skipped": N}.
    """
    text_chunks_added = 0
    images_added = 0
    files_skipped = 0

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        is_image = path.suffix.lower() in IMAGE_SUFFIXES
        raw_text = extract_text(path)

        if raw_text.strip():
            chunks = chunk_text(
                raw_text,
                source=str(path),
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            if chunks:
                store.add_text_chunks(chunks)
                text_chunks_added += len(chunks)

        if is_image:
            try:
                vector = embed_image(path)
                store.add_image(path=str(path), vector=vector)
                images_added += 1
            except Exception:
                logger.warning("image_embedding_failed", path=str(path))
                files_skipped += 1
        elif not raw_text.strip():
            files_skipped += 1

    logger.info(
        "ingestion_complete",
        text_chunks=text_chunks_added,
        images=images_added,
        files_skipped=files_skipped,
    )
    return {
        "text_chunks": text_chunks_added,
        "images": images_added,
        "files_skipped": files_skipped,
    }
