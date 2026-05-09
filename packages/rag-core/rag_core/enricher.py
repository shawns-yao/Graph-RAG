"""Contextual enrichment for chunks via LLM.

From RAG 2.0 — generates per-chunk context explaining its role
within the document using OpenAI chat completions.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from rag_core.config import get_settings, make_openai_client
from rag_core.llm_resilience import LLMFatalError
from rag_core.models import Chunk

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)
_LIGHTWEIGHT_ENRICH_MAX_CHUNKS = 6
_LIGHTWEIGHT_ENRICH_MAX_TOTAL_CHARS = 5000


def enrich_chunks(
    chunks: list[Chunk],
    document_summary: str = "",
    openai_client: OpenAI | None = None,
) -> list[Chunk]:
    """Enrich chunks with contextual information via LLM.

    If no document_summary provided, generates one from first few chunks.
    For each chunk, calls OpenAI to generate 1-2 sentence context.
    Sets chunk.context = LLM response.
    """
    if not chunks:
        return chunks

    cfg = get_settings()
    client = openai_client or make_openai_client(cfg)
    total_chars = sum(len(chunk.content) for chunk in chunks)

    if len(chunks) <= _LIGHTWEIGHT_ENRICH_MAX_CHUNKS and total_chars <= _LIGHTWEIGHT_ENRICH_MAX_TOTAL_CHARS:
        logger.info(
            "Using lightweight enrichment for %d chunks (%d chars total)",
            len(chunks),
            total_chars,
        )
        return _lightweight_enrich_chunks(chunks)

    if not document_summary:
        document_summary = _generate_summary(chunks[:3], client, cfg.openai.llm_model)
        logger.info("Generated document summary")

    enriched: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        try:
            context = _generate_context(
                chunk.content,
                document_summary,
                chunk.context,
                client,
                cfg.openai.llm_model,
            )
            if chunk.context.strip() and context.strip():
                chunk.context = f"{chunk.context.strip()}\n\n{context.strip()}"
            elif context.strip():
                chunk.context = context
            logger.debug("Enriched chunk %d/%d: %s", i + 1, len(chunks), context[:50])

            if i < len(chunks) - 1:
                time.sleep(0.1)

        except LLMFatalError:
            raise
        except Exception as e:
            logger.warning("Failed to enrich chunk %d: %s", i, e)

        enriched.append(chunk)

    logger.info("Enriched %d chunks", len(enriched))
    return enriched


def _lightweight_enrich_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Attach deterministic context for short documents without per-chunk LLM calls."""
    document_summary = _build_lightweight_summary(chunks)
    for index, chunk in enumerate(chunks, start=1):
        chunk.context = _build_lightweight_context(
            chunk=chunk,
            document_summary=document_summary,
            position=index,
            total=len(chunks),
        )
    logger.info("Enriched %d chunks with lightweight context", len(chunks))
    return chunks


def _build_lightweight_summary(chunks: list[Chunk]) -> str:
    """Create a compact document summary from the first chunks."""
    lead = " ".join(chunk.content.strip() for chunk in chunks[:2]).strip()
    lead = " ".join(lead.split())
    if not lead:
        return "Document overview unavailable."
    return lead[:320].rstrip()


def _build_lightweight_context(
    *,
    chunk: Chunk,
    document_summary: str,
    position: int,
    total: int,
) -> str:
    """Build deterministic chunk context using structural metadata."""
    lines = [f"Document summary: {document_summary}"]
    heading_path = chunk.metadata.get("heading_path") if isinstance(chunk.metadata, dict) else None
    if heading_path:
        lines.append(f"Section: {' > '.join(str(item) for item in heading_path)}")
    lines.append(f"Chunk position: {position}/{total}")
    preview = " ".join(chunk.content.strip().split())[:220].rstrip()
    if preview:
        lines.append(f"Focus: {preview}")
    return "\n".join(lines)


def _generate_summary(
    chunks: list[Chunk], client: OpenAI, model: str,
) -> str:
    """Generate document summary from first few chunks."""
    combined = "\n\n".join(c.content for c in chunks)

    prompt = (
        f"Here are the first few sections of a document:\n\n"
        f"{combined[:2000]}\n\n"
        f"Write 2-3 sentences summarizing what this document is about."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        return resp.choices[0].message.content or "Unknown document"
    except LLMFatalError:
        raise
    except Exception as e:
        logger.warning("Failed to generate summary: %s", e)
        return "Document"


def _generate_context(
    chunk_content: str,
    document_summary: str,
    structural_context: str,
    client: OpenAI,
    model: str,
) -> str:
    """Generate 1-2 sentence context for a chunk."""
    context_hint = ""
    if structural_context.strip():
        context_hint = (
            "Here's deterministic structural context for this chunk:\n\n"
            f"{structural_context[:400]}\n\n"
        )
    prompt = (
        f"Here's the document: {document_summary}\n\n"
        f"{context_hint}"
        f"Here's a chunk from the document:\n\n"
        f"{chunk_content[:500]}\n\n"
        f"Write 1-2 sentences explaining the context of this chunk "
        f"within the document. Be specific and concise."
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
        )
        return resp.choices[0].message.content or ""
    except LLMFatalError:
        raise
    except Exception as e:
        logger.warning("Failed to generate context: %s", e)
        return ""
