"""Extract raw text from PDFs, plain text files, and scanned images (OCR)."""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

TEXT_SUFFIXES = {".txt", ".md"}
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def extract_text(path: Path) -> str:
    """Dispatch extraction by file suffix. Returns empty string on unsupported/empty input."""
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return _extract_plain_text(path)
    if suffix in PDF_SUFFIXES:
        return _extract_pdf_text(path)
    if suffix in IMAGE_SUFFIXES:
        return _extract_image_text_ocr(path)
    logger.warning("unsupported_file_type", path=str(path), suffix=suffix)
    return ""


def _extract_plain_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf_text(path: Path) -> str:
    import fitz  # PyMuPDF

    pages: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            pages.append(page.get_text())
    return "\n".join(pages)


def _extract_image_text_ocr(path: Path) -> str:
    """Best-effort OCR pass. If tesseract isn't installed on the host, returns ''.

    The image is still separately embedded via CLIP in image_extractor.py, so a missing
    OCR text layer doesn't stop the image from being retrievable.
    """
    try:
        import pytesseract
        from PIL import Image

        with Image.open(path) as img:
            return pytesseract.image_to_string(img)
    except Exception:  # pragma: no cover - environment-dependent
        logger.warning("ocr_unavailable", path=str(path))
        return ""
