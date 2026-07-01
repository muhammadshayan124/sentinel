FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

COPY tests ./tests

ENV SENTINEL_DATA_DIR=/app/data \
    SENTINEL_CHROMA_DIR=/app/chroma_data

EXPOSE 8000

CMD ["sentinel", "serve", "--host", "0.0.0.0", "--port", "8000"]
