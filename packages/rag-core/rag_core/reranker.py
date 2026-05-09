"""Result re-ranking for improved retrieval quality.

Supports cross-encoder re-ranking with cosine fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from rag_core.config import get_settings
from rag_core.models import SearchResult

logger = logging.getLogger(__name__)
_CROSS_ENCODERS: dict[str, Any] = {}
_RERANK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "described", "did", "do",
    "does", "during", "events", "for", "from", "had", "has", "have", "he", "her",
    "him", "his", "how", "in", "into", "is", "it", "its", "of", "on", "or",
    "recorded", "that", "the", "their", "them", "there", "these", "this", "those",
    "to", "was", "were", "what", "when", "where", "which", "who", "with",
    "diary", "daily", "account",
}


def _normalized_score(score: float, max_score: float) -> float:
    if max_score <= 0.0:
        return 0.0
    return round(min(1.0, max(0.0, score / max_score)), 3)


def _finalize_reranked_results(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    trimmed = results[:top_k]
    if not trimmed:
        return []
    max_score = max((result.score for result in trimmed), default=0.0)
    return [
        result.model_copy(
            update={
                "score_normalized": _normalized_score(result.score, max_score),
                "rank": index + 1,
            }
        )
        for index, result in enumerate(trimmed)
    ]


def rerank_cosine(
    query_embedding: list[float],
    results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Re-rank results by cosine similarity between query and chunk embeddings."""
    if not results:
        return []

    valid_results = [r for r in results if r.chunk.embedding]
    if not valid_results:
        logger.warning("No results with embeddings to rerank")
        return results[:top_k]

    query_vec = np.array(query_embedding)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        logger.warning("Query embedding has zero norm, returning original results")
        return results[:top_k]

    scored_results = []
    for result in valid_results:
        chunk_vec = np.array(result.chunk.embedding)
        chunk_norm = np.linalg.norm(chunk_vec)

        if chunk_norm == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm))

        scored_results.append(result.model_copy(update={"score": similarity}))

    scored_results.sort(key=lambda r: r.score, reverse=True)

    reranked = _finalize_reranked_results(scored_results, top_k)
    logger.debug("Reranked %d results to top %d", len(scored_results), len(reranked))
    return reranked


def _extract_query_terms(query: str) -> list[str]:
    """Keep deterministic lexical anchors for fallback ranking."""
    phrase_tokens_seen: set[str] = set()
    token_terms: list[str] = []
    seen: set[str] = set()

    for phrase in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query):
        for part in phrase.split():
            lowered = part.casefold()
            if lowered not in _RERANK_STOPWORDS:
                phrase_tokens_seen.add(lowered)

    for token in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_-]+", query):
        lowered = token.casefold()
        if (
            len(lowered) <= 2
            or lowered in _RERANK_STOPWORDS
            or lowered in phrase_tokens_seen
            or lowered in seen
        ):
            continue
        token_terms.append(lowered)
        seen.add(lowered)
    if token_terms:
        return token_terms
    return sorted(phrase_tokens_seen)


def rerank_lexical_semantic(
    query: str,
    query_embedding: list[float],
    results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Fallback rerank that blends cosine similarity with exact lexical coverage."""
    if not results:
        return []

    query_terms = _extract_query_terms(query)
    query_vec = np.array(query_embedding) if query_embedding else None
    query_norm = float(np.linalg.norm(query_vec)) if query_vec is not None else 0.0

    rescored: list[SearchResult] = []
    for result in results:
        content = result.chunk.enriched_content or result.chunk.content or ""
        lowered_content = content.casefold()
        content_terms = set(re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_-]+", lowered_content))
        lexical_ratio = (
            sum(1 for term in query_terms if term in content_terms) / len(query_terms)
            if query_terms
            else 0.0
        )
        positions = [lowered_content.find(term) for term in query_terms if lowered_content.find(term) >= 0]
        proximity_score = 0.0
        if len(positions) >= 2:
            span = max(positions) - min(positions)
            proximity_score = 1.0 / (1.0 + (span / 200.0))
        elif len(positions) == 1:
            proximity_score = 0.5

        cosine_score = 0.0
        if query_vec is not None and query_norm > 0.0 and result.chunk.embedding:
            chunk_vec = np.array(result.chunk.embedding)
            chunk_norm = float(np.linalg.norm(chunk_vec))
            if chunk_norm > 0.0:
                cosine_score = float(np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm))

        combined_score = (
            (0.45 * cosine_score)
            + (0.25 * lexical_ratio)
            + (0.30 * proximity_score)
        )
        rescored.append(result.model_copy(update={"score": combined_score}))

    rescored.sort(key=lambda item: item.score, reverse=True)
    return _finalize_reranked_results(rescored, top_k)


def _load_cross_encoder(model_name: str):
    """Load and cache a CrossEncoder model if the dependency is available."""
    if model_name in _CROSS_ENCODERS:
        return _CROSS_ENCODERS[model_name]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning(
            "sentence-transformers is not installed; falling back to cosine rerank"
        )
        _CROSS_ENCODERS[model_name] = None
        return None

    try:
        encoder = CrossEncoder(model_name, trust_remote_code=False)
    except Exception as exc:  # pragma: no cover - defensive around model bootstrap
        logger.warning(
            "Failed to load reranker model '%s'; falling back to cosine rerank: %s",
            model_name,
            exc,
        )
        _CROSS_ENCODERS[model_name] = None
        return None

    _CROSS_ENCODERS[model_name] = encoder
    return encoder


def rerank_cross_encoder(
    query: str,
    results: list[SearchResult],
    top_k: int,
    model_name: str | None = None,
) -> list[SearchResult]:
    """Re-rank results with a CrossEncoder over query-document pairs."""
    if not results:
        return []

    cfg = get_settings()
    encoder = _load_cross_encoder(model_name or cfg.retrieval.reranker_model)
    if encoder is None:
        return results[:top_k]

    pairs = [(query, result.chunk.content) for result in results]
    scores = encoder.predict(pairs)

    rescored = [
        result.model_copy(update={"score": float(score)})
        for result, score in zip(results, scores, strict=False)
    ]
    rescored.sort(key=lambda item: item.score, reverse=True)
    return _finalize_reranked_results(rescored, top_k)


def rerank(
    query: str | list[float],
    results: list[SearchResult],
    top_k: int | None = None,
    query_embedding: list[float] | None = None,
) -> list[SearchResult]:
    """Re-rank search results with cross-encoder or cosine fallback.

    Args:
        query: Query text or query embedding vector.
        results: Search results to re-rank.
        top_k: Number of top results to return.
        query_embedding: Optional query embedding used for cosine fallback.
    """
    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_final

    if isinstance(query, list):
        return rerank_cosine(query, results, top_k)

    if cfg.retrieval.reranker_backend == "cross_encoder":
        encoder = _load_cross_encoder(cfg.retrieval.reranker_model)
        if encoder is None:
            if query_embedding:
                return rerank_lexical_semantic(query, query_embedding, results, top_k)
            logger.warning("Cross-encoder unavailable and no embedding fallback provided")
            return results[:top_k]
        reranked = rerank_cross_encoder(
            query,
            results,
            top_k=top_k,
            model_name=cfg.retrieval.reranker_model,
        )
        if reranked:
            return reranked

    if query_embedding:
        return rerank_cosine(query_embedding, results, top_k)

    logger.warning("No query embedding available for cosine fallback, returning top results")
    return results[:top_k]
