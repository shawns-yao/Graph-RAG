"""Batch embedding for chunks via OpenAI Embeddings API.

Extracted from RAG 2.0 enricher — dedicated module for embedding generation
using text-embedding-3-small (1536 dim by default).
"""

from __future__ import annotations

import logging

from rag_core.config import get_settings, make_openai_client
from rag_core.models import Chunk

logger = logging.getLogger(__name__)
_DEFAULT_EMBED_BATCH_SIZE = 64


def embed_chunks(chunks: list[Chunk], openai_client=None) -> list[Chunk]:
    """Batch embed chunks using OpenAI Embeddings API.

    Uses enriched_content (context + content) if available.
    Sets chunk.embedding for each chunk.
    """
    if not chunks:
        return chunks

    cfg = get_settings()
    client = openai_client or make_openai_client(cfg)

    texts = [chunk.enriched_content for chunk in chunks]
    batch_size = max(1, getattr(cfg.ingest, "embedding_batch_size", _DEFAULT_EMBED_BATCH_SIZE))

    try:
        for offset in range(0, len(chunks), batch_size):
            batch_chunks = chunks[offset : offset + batch_size]
            batch_texts = texts[offset : offset + batch_size]
            response = client.embeddings.create(
                model=cfg.openai.embedding_model,
                input=batch_texts,
            )
            for index, chunk in enumerate(batch_chunks):
                chunk.embedding = response.data[index].embedding

        logger.info("Embedded %d chunks (%s)", len(chunks), cfg.openai.embedding_model)

    except Exception as e:
        logger.error("Failed to embed chunks: %s", e)
        raise

    return chunks
