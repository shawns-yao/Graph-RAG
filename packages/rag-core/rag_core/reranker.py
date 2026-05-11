"""Cross-encoder result re-ranking."""

from __future__ import annotations

import logging
from typing import Any

from rag_core.config import get_settings
from rag_core.models import SearchResult

logger = logging.getLogger(__name__)
_CROSS_ENCODERS: dict[str, Any] = {}


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


def _load_cross_encoder(model_name: str):
    """Load and cache a CrossEncoder model if the dependency is available."""
    if model_name in _CROSS_ENCODERS:
        return _CROSS_ENCODERS[model_name]

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning("sentence-transformers is not installed; cross-encoder rerank skipped")
        _CROSS_ENCODERS[model_name] = None
        return None

    try:
        encoder = CrossEncoder(model_name, trust_remote_code=False)
    except Exception as exc:  # pragma: no cover - defensive around model bootstrap
        logger.warning(
            "Failed to load cross-encoder reranker model '%s'; rerank skipped: %s",
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
) -> list[SearchResult]:
    """Re-rank search results with the configured cross-encoder only.

    Args:
        query: Query text. Embedding-list input is accepted only for legacy
            call compatibility and returns the original top-k results.
        results: Search results to re-rank.
        top_k: Number of top results to return.
    """
    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_final

    if isinstance(query, list):
        logger.warning("Embedding-only rerank input is unsupported; returning top results")
        return results[:top_k]

    return rerank_cross_encoder(
        query,
        results,
        top_k=top_k,
        model_name=cfg.retrieval.reranker_model,
    )
