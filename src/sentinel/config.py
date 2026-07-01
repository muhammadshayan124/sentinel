"""Central configuration, loaded from environment variables with sane local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    data_dir: Path = Path(os.getenv("SENTINEL_DATA_DIR", "./data")).resolve()
    chroma_dir: Path = Path(os.getenv("SENTINEL_CHROMA_DIR", "./chroma_data")).resolve()

    text_embedding_model: str = os.getenv("SENTINEL_TEXT_EMBED_MODEL", "all-MiniLM-L6-v2")
    image_embedding_model: str = os.getenv("SENTINEL_IMAGE_EMBED_MODEL", "clip-ViT-B-32")

    chunk_size: int = int(os.getenv("SENTINEL_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("SENTINEL_CHUNK_OVERLAP", "120"))
    retrieval_top_k: int = int(os.getenv("SENTINEL_TOP_K", "5"))

    max_agent_steps: int = int(os.getenv("SENTINEL_MAX_AGENT_STEPS", "6"))

    otel_console_export: bool = os.getenv("SENTINEL_OTEL_CONSOLE", "true").lower() == "true"


settings = Settings()
