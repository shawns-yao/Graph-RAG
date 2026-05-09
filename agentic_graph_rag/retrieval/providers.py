"""Retrieval providers for vector, sparse, and graph channels."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_core.config import get_settings
from rag_core.models import Chunk, GraphContext, SearchResult
from rag_core.neo4j_utils import open_neo4j_session
from rag_core.vector_store import VectorStore

from agentic_graph_rag.text_signals import build_tfidf_profile, rank_keywords
from agentic_graph_rag.retrieval.vector_cypher import search as vector_cypher_search

_PASSAGE_FULLTEXT_INDEX_READY = False

_BM25_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "described", "did", "do",
    "does", "during", "events", "for", "from", "had", "has", "have", "he", "her",
    "him", "his", "how", "in", "into", "is", "it", "its", "of", "on", "or",
    "recorded", "that", "the", "their", "them", "there", "these", "this", "those",
    "to", "was", "were", "what", "when", "where", "which", "who", "with",
    "diary", "daily", "account",
}


@dataclass(slots=True)
class RetrievalRequest:
    """Normalized request contract for all retrieval providers."""

    query: str
    top_k: int
    query_embedding: list[float] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)


class RetrievalProvider(Protocol):
    """Pluggable retrieval provider interface."""

    name: str

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        """Return normalized SearchResult objects for a retrieval request."""


def build_bm25_focus_query(
    query: str,
    corpus_texts: list[str],
) -> str:
    """Compress natural-language query text into high-IDF lexical anchors."""
    if not query.strip():
        return query

    cfg = get_settings().retrieval
    profile = build_tfidf_profile(corpus_texts)
    ranked = rank_keywords(
        query,
        profile,
        min_idf=float(getattr(cfg, "tfidf_query_min_idf", 1.2)),
        max_keywords=max(1, int(getattr(cfg, "tfidf_query_max_keywords", 6))),
    )
    if not ranked:
        return query
    return " ".join(item.term for item in ranked)


def _resolve_bm25_query_text(query: str, corpus_texts: list[str]) -> str:
    """Use TF-IDF query cleanup only when it produces enough lexical anchors."""
    focus_query = build_bm25_focus_query(query, corpus_texts)
    focus_required, _focus_optional = _extract_bm25_anchors(focus_query)
    original_required, _original_optional = _extract_bm25_anchors(query)
    if len(focus_required) >= max(1, min(2, len(original_required) or 1)):
        return focus_query
    return query


def _extract_bm25_anchors(query: str) -> tuple[list[str], list[str]]:
    """Extract required lexical tokens plus optional phrase anchors."""
    if not query.strip():
        return [], []

    phrase_anchors: list[str] = []
    token_anchors: list[str] = []
    seen: set[str] = set()

    for phrase in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query):
        phrase_tokens = [part for part in phrase.split() if part.casefold() not in _BM25_STOPWORDS]
        if not phrase_tokens:
            continue
        normalized_phrase = " ".join(phrase_tokens)
        normalized = normalized_phrase.casefold()
        if normalized not in seen:
            phrase_anchors.append(normalized_phrase)
            seen.add(normalized)
            for part in phrase_tokens:
                seen.add(part.casefold())

    for token in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_-]+", query):
        lowered = token.casefold()
        if len(lowered) <= 2 or lowered in _BM25_STOPWORDS or lowered in seen:
            continue
        token_anchors.append(lowered)
        seen.add(lowered)

    required_anchors = token_anchors[:4] if token_anchors else phrase_anchors[:2]
    optional_anchors = phrase_anchors[:2] if token_anchors else []
    return required_anchors, optional_anchors


def _build_bm25_search_text(query: str) -> str:
    """Build a Lucene query string from deterministic lexical anchors."""
    required_anchors, optional_anchors = _extract_bm25_anchors(query)

    lucene_terms = []
    for anchor in required_anchors:
        if " " in anchor:
            lucene_terms.append(f'+\"{anchor}\"')
        else:
            lucene_terms.append(f"+{anchor}")
    for anchor in optional_anchors:
        if " " in anchor:
            lucene_terms.append(f'"{anchor}"')
        elif anchor not in required_anchors:
            lucene_terms.append(anchor)
    return " ".join(lucene_terms) or query


def _bm25_local_sort_key(
    content: str,
    *,
    required_anchors: list[str],
    base_score: float,
    base_rank: int,
) -> tuple[float, float, float, float]:
    """Favor candidates where required lexical anchors co-occur tightly."""
    if not required_anchors:
        return (0.0, 0.0, base_score, -base_rank)

    lowered = content.casefold()
    positions = [lowered.find(anchor.casefold()) for anchor in required_anchors]
    matched_positions = [position for position in positions if position >= 0]
    coverage = len(matched_positions) / len(required_anchors)

    proximity = 0.0
    if len(matched_positions) >= 2:
        span = max(matched_positions) - min(matched_positions)
        proximity = 1.0 / (1.0 + (span / 200.0))

    return (coverage, proximity, base_score, -base_rank)


def graph_context_to_search_results(
    ctx: GraphContext,
    source: str,
    include_graph_structure: bool = False,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Convert GraphContext into normalized SearchResult candidates."""
    results: list[SearchResult] = []
    graph_prefix = _graph_context_prefix(ctx) if include_graph_structure else ""

    if ctx.passages:
        passages = ctx.passages if top_k is None else ctx.passages[:top_k]
        source_ids = ctx.source_ids if top_k is None else ctx.source_ids[:top_k]
        for index, passage in enumerate(passages):
            chunk_id = source_ids[index] if index < len(source_ids) else ""
            content = passage
            if graph_prefix:
                content = f"{graph_prefix}\n\nEvidence:\n{passage}"
            results.append(
                SearchResult(
                    chunk=Chunk(id=chunk_id, content=content),
                    score=1.0 / (index + 1),
                    rank=index + 1,
                    source=source,
                )
            )
        return results

    if graph_prefix:
        virtual_id = hashlib.md5(graph_prefix.encode("utf-8")).hexdigest()[:12]
        return [
            SearchResult(
                chunk=Chunk(id=f"graph-{virtual_id}", content=graph_prefix),
                score=1.0,
                rank=1,
                source=source,
            )
        ]

    return []


def _graph_context_prefix(ctx: GraphContext) -> str:
    """Serialize graph entities and paths into human-readable text."""
    sections: list[str] = []
    if ctx.triplets:
        path_lines = [
            f"{triplet.get('source', '')} -[{triplet.get('relation', '')}]-> {triplet.get('target', '')}"
            for triplet in ctx.triplets[:8]
        ]
        sections.append("Graph paths:\n" + "\n".join(path_lines))

    if ctx.entities:
        entity_lines = []
        for entity in ctx.entities[:8]:
            label = entity.entity_type or "Entity"
            entity_lines.append(f"{entity.name} ({label})")
        sections.append("Entities:\n" + "\n".join(entity_lines))

    return "\n\n".join(section for section in sections if section).strip()


class VectorRetrievalProvider:
    """Dense semantic retrieval provider."""

    name = "vector"

    def __init__(self, driver) -> None:
        self._driver = driver

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        results = VectorStore(driver=self._driver).search(
            request.query_embedding,
            top_k=request.top_k,
        )
        for item in results:
            item.source = self.name
        return results


class GraphRetrievalProvider:
    """Graph traversal provider that serializes paths into virtual documents."""

    name = "graph"

    def __init__(self, driver) -> None:
        self._driver = driver

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        cfg = get_settings()
        max_hops = int(request.filters.get("max_hops", cfg.retrieval.max_hops))
        entry_top_k = int(request.filters.get("entry_top_k", cfg.retrieval.graph_entry_top_k))
        ctx = vector_cypher_search(
            request.query_embedding,
            self._driver,
            top_k=entry_top_k,
            max_hops=max_hops,
        )
        return graph_context_to_search_results(
            ctx,
            source=self.name,
            include_graph_structure=True,
            top_k=request.top_k,
        )


class BM25RetrievalProvider:
    """Sparse lexical retrieval provider via Neo4j full-text index."""

    name = "bm25"

    def __init__(self, driver) -> None:
        self._driver = driver

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        cfg = get_settings()
        _ensure_passage_fulltext_index(self._driver)
        index_name = cfg.retrieval.fulltext_index_name
        focus_query = _resolve_bm25_query_text(request.query, _sample_passage_texts(self._driver))
        required_anchors, _optional_anchors = _extract_bm25_anchors(focus_query)
        search_text = _build_bm25_search_text(focus_query)

        with open_neo4j_session(self._driver) as session:
            result = session.run(
                """
                CALL db.index.fulltext.queryNodes($index_name, $search_text, {limit: $top_k})
                YIELD node, score
                RETURN node.id AS id,
                       node.chunk_id AS chunk_id,
                       node.text AS text,
                       score
                ORDER BY score DESC
                """,
                index_name=index_name,
                search_text=search_text,
                top_k=request.top_k,
            )

            results: list[SearchResult] = []
            for rank, record in enumerate(result, start=1):
                chunk_id = record["chunk_id"] or record["id"] or ""
                content = record["text"] or ""
                if not content:
                    continue
                results.append(
                    SearchResult(
                        chunk=Chunk(id=chunk_id, content=content),
                        score=float(record["score"] or 0.0),
                        rank=rank,
                        source=self.name,
                    )
                )
        results.sort(
            key=lambda item: _bm25_local_sort_key(
                item.chunk.enriched_content,
                required_anchors=required_anchors,
                base_score=item.score,
                base_rank=item.rank,
            ),
            reverse=True,
        )
        return [
            SearchResult(
                chunk=item.chunk,
                score=item.score,
                rank=index + 1,
                source=item.source,
            )
            for index, item in enumerate(results)
        ]


def _sample_passage_texts(driver, limit: int = 256) -> list[str]:
    """Sample passage text to estimate corpus-level IDF for BM25 query cleanup."""
    with open_neo4j_session(driver) as session:
        result = session.run(
            """
            MATCH (p:PassageNode)
            RETURN p.text AS text
            LIMIT $limit
            """,
            limit=limit,
        )
        return [str(record["text"] or "") for record in result if record["text"]]


def fetch_passage_embeddings(
    chunk_ids: list[str],
    driver,
) -> dict[str, list[float]]:
    """Fetch PassageNode embeddings for normalized result chunks."""
    from agentic_graph_rag.indexing.dual_node import PASSAGE_LABEL

    if not chunk_ids:
        return {}

    emb_map: dict[str, list[float]] = {}
    with open_neo4j_session(driver) as session:
        result = session.run(
            f"""
            MATCH (pa:{PASSAGE_LABEL})
            WHERE pa.chunk_id IN $chunk_ids
            RETURN pa.chunk_id AS chunk_id, pa.embedding AS embedding
            """,
            chunk_ids=chunk_ids,
        )
        for record in result:
            chunk_id = record["chunk_id"]
            embedding = record["embedding"]
            if chunk_id and embedding:
                emb_map[chunk_id] = list(embedding)
    return emb_map


def attach_passage_embeddings(
    results: list[SearchResult],
    driver,
) -> list[SearchResult]:
    """Attach stored embeddings onto normalized results for reranking."""
    chunk_ids = [result.chunk.id for result in results if result.chunk.id]
    emb_map = fetch_passage_embeddings(chunk_ids, driver)
    for result in results:
        chunk_id = result.chunk.id
        if chunk_id and not result.chunk.embedding and chunk_id in emb_map:
            result.chunk.embedding = emb_map[chunk_id]
    return results


def _ensure_passage_fulltext_index(driver) -> None:
    """Create PassageNode full-text index if needed."""
    global _PASSAGE_FULLTEXT_INDEX_READY  # noqa: PLW0603
    if _PASSAGE_FULLTEXT_INDEX_READY:
        return

    from agentic_graph_rag.indexing.dual_node import PASSAGE_LABEL

    index_name = get_settings().retrieval.fulltext_index_name
    with open_neo4j_session(driver) as session:
        session.run(
            f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (n:{PASSAGE_LABEL})
            ON EACH [n.text]
            """
        )
    _PASSAGE_FULLTEXT_INDEX_READY = True
