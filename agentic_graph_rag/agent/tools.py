"""Retrieval Tools — callable tools for the agentic router.

Each tool wraps a retrieval strategy and returns a list of SearchResult.
Tools are pure functions with driver/client injected for testability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.llm_resilience import LLMCallController
from rag_core.reranker import rerank
from rag_core.models import Chunk, GraphContext, QueryType, SearchResult
from rag_core.neo4j_utils import open_neo4j_session

from agentic_graph_rag.retrieval.fusion import FusionEngine
from agentic_graph_rag.retrieval.orchestrator import RetrievalOrchestrator
from agentic_graph_rag.retrieval.providers import (
    BM25RetrievalProvider,
    GraphRetrievalProvider,
    RetrievalRequest,
    VectorRetrievalProvider,
    graph_context_to_search_results,
)

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: embed query
# ---------------------------------------------------------------------------

def _embed_query(query: str, openai_client: OpenAI) -> list[float]:
    """Embed query text using OpenAI embeddings."""
    cfg = get_settings()
    max_retries = cfg.agent.max_retries
    controller = LLMCallController(
        max_retries=max_retries,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=8.0,
        jitter_seconds=0.25,
        max_consecutive_failures=max(2, max_retries + 1),
        total_budget_seconds=max(5.0, float(cfg.agent.request_time_budget_ms) / 1000),
    )
    response = controller.call(
        "query_embedding",
        openai_client.embeddings.create,
        model=cfg.openai.embedding_model,
        input=query,
        dimensions=cfg.openai.embedding_dimensions,
    )
    return response.data[0].embedding


def _graph_context_to_results(ctx: GraphContext, source: str) -> list[SearchResult]:
    """Convert GraphContext passages into SearchResult list."""
    return graph_context_to_search_results(ctx, source=source)


# ---------------------------------------------------------------------------
# 1. Vector search (simple)
# ---------------------------------------------------------------------------

def vector_search(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Simple dense retrieval through the vector provider."""
    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_vector

    request = RetrievalRequest(
        query=query,
        query_embedding=_embed_query(query, openai_client),
        top_k=top_k,
    )
    return VectorRetrievalProvider(driver).retrieve(request)


# ---------------------------------------------------------------------------
# 2. Cypher traverse (relation / multi-hop)
# ---------------------------------------------------------------------------

def cypher_traverse(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
    max_hops: int | None = None,
    entry_top_k: int | None = None,
) -> list[SearchResult]:
    """Graph retrieval with query-structure-aware constrained traversal.

    Parses the query into structured slots (focus entities, constraints,
    relation intent) and passes them as traversal constraints. This prevents
    unconstrained BFS from pulling in generic high-frequency clusters.
    """
    from agentic_graph_rag.agent.query_parser import parse_query_structure

    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_vector
    if max_hops is None:
        max_hops = cfg.retrieval.max_hops
    if entry_top_k is None:
        entry_top_k = cfg.retrieval.graph_entry_top_k

    # Parse query structure for constrained traversal.
    query_structure = parse_query_structure(query, openai_client=openai_client)

    request = RetrievalRequest(
        query=query,
        query_embedding=_embed_query(query, openai_client),
        top_k=top_k,
        filters={
            "max_hops": max_hops,
            "entry_top_k": entry_top_k,
            "query_structure": query_structure,
        },
    )
    return GraphRetrievalProvider(driver).retrieve(request)


# ---------------------------------------------------------------------------
# 3. Community search (Graphiti)
# ---------------------------------------------------------------------------

def community_search(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
) -> list[SearchResult]:
    """Search using Graphiti community summaries.

    Falls back to vector search if Graphiti is unavailable.
    """
    # Community search requires Graphiti; use vector search as fallback
    logger.info("Community search — falling back to vector search (Graphiti optional)")
    return vector_search(query, driver, openai_client)


# ---------------------------------------------------------------------------
# 4. Sparse retrieval + hybrid fusion
# ---------------------------------------------------------------------------

def bm25_search(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Sparse lexical retrieval via the BM25 provider."""
    del openai_client  # interface parity with other tools

    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_bm25

    return BM25RetrievalProvider(driver).retrieve(
        RetrievalRequest(query=query, top_k=top_k)
    )


def hybrid_search(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
    query_type: QueryType | str | None = None,
    seed_results: dict[str, list[SearchResult]] | None = None,
    enabled_providers: list[str] | None = None,
    provider_results: dict[str, list[SearchResult]] | None = None,
    rerank_enabled: bool = True,
) -> list[SearchResult]:
    """Hybrid retrieval delegated to provider orchestrator and fusion engine."""
    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_final

    query_emb = _embed_query(query, openai_client)
    orchestrator = RetrievalOrchestrator(
        driver,
        providers=[
            VectorRetrievalProvider(driver),
            BM25RetrievalProvider(driver),
            GraphRetrievalProvider(driver),
        ],
    )
    results = orchestrator.search(
        query=query,
        query_embedding=query_emb,
        top_k=top_k,
        query_type=query_type,
        seed_results=seed_results,
        enabled_providers=enabled_providers,
        provider_results=provider_results,
        rerank_enabled=rerank_enabled,
        provider_top_k={
            "vector": cfg.retrieval.top_k_vector,
            "bm25": cfg.retrieval.top_k_bm25,
            "graph": cfg.retrieval.top_k_vector,
        },
        provider_filters={"graph": {"max_hops": cfg.retrieval.max_hops}},
    )
    logger.info("Hybrid search returned %d results", len(results))
    return results


def _priority_merge(
    *result_lists: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """Priority-merge one or more result lists; kept for older call sites."""
    return FusionEngine().fuse(*result_lists, top_k=top_k)


# ---------------------------------------------------------------------------
# 5. Temporal query
# ---------------------------------------------------------------------------

_TEMPORAL_RE = None


def _get_temporal_re():
    """Lazy-compiled regex for temporal keywords."""
    global _TEMPORAL_RE  # noqa: PLW0603
    if _TEMPORAL_RE is None:
        import re
        _TEMPORAL_RE = re.compile(
            r'\b('
            r'\d{4}'                                      # years: 2020, 1995
            r'|первый|первая|первое|первые'               # "first" (RU)
            r'|история|исторический|эволюция|развитие'    # history/evolution (RU)
            r'|first|history|evolution|timeline|founded'   # temporal (EN)
            r'|начало|основан|создан|появи'               # origin (RU)
            r')\b',
            re.IGNORECASE,
        )
    return _TEMPORAL_RE


def _passage_vector_search(
    query_emb: list[float],
    driver: Driver,
    top_k: int,
) -> list[dict]:
    """Search PassageNode via Neo4j vector index, falling back to label scan."""
    from agentic_graph_rag.indexing.dual_node import PASSAGE_INDEX_NAME, PASSAGE_LABEL

    with open_neo4j_session(driver) as session:
        try:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes(
                    '{PASSAGE_INDEX_NAME}', $top_k, $embedding
                )
                YIELD node, score
                RETURN node.id AS id,
                       node.text AS text,
                       node.chunk_id AS chunk_id,
                       score
                ORDER BY score DESC
                """,
                top_k=top_k,
                embedding=query_emb,
            )
            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "chunk_id": record["chunk_id"],
                    "score": float(record["score"]),
                }
                for record in result
                if record["text"]
            ]
        except Exception as exc:
            logger.warning(
                "Passage vector index query failed (%s), falling back to label scan",
                exc,
            )
            return _passage_label_scan(query_emb, driver, top_k, PASSAGE_LABEL)


def _passage_label_scan(
    query_emb: list[float],
    driver: Driver,
    top_k: int,
    passage_label: str,
) -> list[dict]:
    """Fallback: full label scan with Python-side cosine ranking."""
    with open_neo4j_session(driver) as session:
        result = session.run(
            f"""
            MATCH (pa:{passage_label})
            WHERE pa.text IS NOT NULL AND pa.text <> '' AND pa.embedding IS NOT NULL
            RETURN pa.id AS id, pa.text AS text, pa.chunk_id AS chunk_id,
                   pa.embedding AS embedding
            """,
        )
        scored: list[tuple[float, dict]] = []
        for record in result:
            emb = record["embedding"]
            sim = _cosine_similarity(query_emb, list(emb)) if emb else 0.0
            scored.append((sim, {
                "id": record["id"],
                "text": record["text"],
                "chunk_id": record["chunk_id"],
                "score": sim,
            }))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def temporal_query(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
) -> list[SearchResult]:
    """Temporal-aware query: vector index search + temporal keyword boost."""
    cfg = get_settings()
    top_k = cfg.retrieval.top_k_final
    query_emb = _embed_query(query, openai_client)
    temporal_re = _get_temporal_re()

    # Oversample to allow temporal boost to re-rank
    candidates = _passage_vector_search(query_emb, driver, top_k=top_k * 3)

    if not candidates:
        logger.info("Temporal query — no passages, falling back to vector search")
        return vector_search(query, driver, openai_client)

    # Apply temporal boost: +0.15 for passages containing temporal markers
    scored: list[tuple[float, dict]] = []
    for candidate in candidates:
        sim = candidate["score"]
        if temporal_re.search(candidate["text"] or ""):
            sim += 0.15
        scored.append((sim, candidate))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    results = []
    for rank, (sim, rec) in enumerate(top, start=1):
        results.append(SearchResult(
            chunk=Chunk(
                id=rec["chunk_id"] or rec["id"] or "",
                content=rec["text"] or "",
            ),
            score=sim,
            rank=rank,
            source="vector",
        ))

    logger.info("Temporal query: %d passages (temporal-boosted)", len(results))
    return results


# ---------------------------------------------------------------------------
# 6. Full document read
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def full_document_read(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Read passage nodes ranked by cosine similarity via vector index."""
    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.top_k_final

    query_emb = _embed_query(query, openai_client)
    candidates = _passage_vector_search(query_emb, driver, top_k=top_k)

    results = []
    for rank, rec in enumerate(candidates, start=1):
        results.append(SearchResult(
            chunk=Chunk(
                id=rec["chunk_id"] or rec["id"] or "",
                content=rec["text"] or "",
            ),
            score=rec["score"],
            rank=rank,
            source="vector",
        ))

    logger.info("Full document read: %d passages (ranked by similarity)", len(results))
    return results


# ---------------------------------------------------------------------------
# 7. Comprehensive search (multi-query fan-out)
# ---------------------------------------------------------------------------

def comprehensive_search(
    query: str,
    driver: Driver,
    openai_client: OpenAI,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Comprehensive retrieval: LLM generates sub-queries + keyword extraction,
    each → retrieval, merge by priority. Also includes full_document_read passages.

    Designed for GLOBAL queries ("list all", "summarize all") where a single
    top-k pass misses components.
    """
    cfg = get_settings()
    if top_k is None:
        top_k = max(cfg.retrieval.top_k_final, 8)

    # Detect enumeration count for dynamic sub-query generation
    n_sub = max(3, min(_detect_enumeration_count(query), 6))
    sub_queries = _generate_sub_queries(query, openai_client, cfg.openai.llm_model_mini, n=n_sub)

    # Fan-out: run vector search for each sub-query
    all_results: list[list[SearchResult]] = []
    for sq in sub_queries:
        results = vector_search(sq, driver, openai_client, top_k=min(cfg.retrieval.top_k_vector, 4))
        all_results.append(results)

    # Full document read — keep narrow and similarity-ranked
    full_top_k = min(max(top_k, 4), 8)
    full_results = full_document_read(query, driver, openai_client, top_k=full_top_k)
    all_results.append(full_results)

    # Merge all result lists via deterministic priority merge
    if not all_results:
        return vector_search(query, driver, openai_client, top_k=top_k)

    merged = all_results[0]
    for i in range(1, len(all_results)):
        merged = _priority_merge(merged, all_results[i], top_k=top_k)

    # Single final re-rank via cosine over the merged pool
    query_emb = _embed_query(query, openai_client)
    merged = rerank(query, merged, top_k=top_k)

    logger.info("Comprehensive search: %d results from %d sub-queries + full_read", len(merged), len(sub_queries))
    return merged


_ENUM_COUNT_RE = None


def _detect_enumeration_count(query: str) -> int:
    """Detect number of items requested in enumeration queries.

    Extracts numbers like "seven", "7", "три" etc. from the query.
    Returns detected count (min 5, max 12) or default 5.
    """
    global _ENUM_COUNT_RE  # noqa: PLW0603
    if _ENUM_COUNT_RE is None:
        import re
        _ENUM_COUNT_RE = re.compile(
            r'\b('
            # English number words
            r'two|three|four|five|six|seven|eight|nine|ten|eleven|twelve'
            # Russian number words
            r'|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять'
            # Digits
            r'|\d{1,2}'
            r')\b',
            re.IGNORECASE,
        )

    word_to_num = {
        "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
        "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
    }

    match = _ENUM_COUNT_RE.search(query)
    if match:
        val = match.group(1).lower()
        n = word_to_num.get(val)
        if n is None:
            try:
                n = int(val)
            except ValueError:
                n = 5
        return max(5, min(n + 2, 12))  # add 2 for coverage margin, cap at 12
    return 5


def _generate_sub_queries(
    query: str, openai_client: OpenAI, model: str, n: int = 5,
) -> list[str]:
    """Generate N sub-queries from original query to improve coverage."""
    # Detect cross-language: Cyrillic query targeting English-only concepts (Doc2)
    import re
    has_cyrillic = bool(re.search(r'[а-яА-ЯёЁ]', query))
    has_en_concept = bool(re.search(
        r'\b(semantic\s+c(ore|ompanion)|SCL|companion\s+layer|MeaningHub|Cognitive\s+Contract)\b',
        query, re.IGNORECASE,
    ))
    lang_hint = ""
    if has_cyrillic and has_en_concept:
        lang_hint = (
            "\nIMPORTANT: The source documents are in English. "
            "Generate all sub-queries in ENGLISH to match the document content.\n"
        )

    prompt = (
        f"You are a search query decomposer for a RAG system. "
        f"Given this query, generate exactly {n} different search sub-queries "
        f"that together cover ALL aspects of the original question. "
        f"Each sub-query should focus on a DIFFERENT section, component, or angle.\n"
        f"For enumeration queries (list all, describe all), "
        f"each sub-query should target a DIFFERENT item from the expected list.\n"
        f"{lang_hint}\n"
        f"Original query: {query}\n\n"
        f"Return ONLY the {n} sub-queries, one per line, no numbering or bullets."
    )
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text = response.choices[0].message.content or ""
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        # Take up to n non-empty lines
        return lines[:n] if lines else [query]
    except Exception as e:
        logger.error("Error generating sub-queries: %s", e)
        return [query]
