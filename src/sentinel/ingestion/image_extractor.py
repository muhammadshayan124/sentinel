"""CLIP-based image embeddings so images are retrievable by semantic similarity to a text query.

This is the CV leg of the pipeline: images are embedded into the same style of vector space
as text (via a CLIP model) and stored alongside text chunks, enabling queries like
"show me the diagram of the network topology" to match a screenshot with no OCR text at all.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from sentinel.config import settings


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.image_embedding_model)


def embed_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB")
        embedding = _model().encode(img, convert_to_numpy=True)
    return embedding


def embed_text_query_for_image_search(query: str) -> np.ndarray:
    """CLIP's text tower shares the embedding space with its image tower, so a plain text
    query can be embedded the same way and compared directly against stored image vectors.
    """
    return _model().encode(query, convert_to_numpy=True)
