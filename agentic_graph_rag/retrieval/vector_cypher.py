"""VectorCypher Retrieval — hybrid vector entry + Cypher graph traversal.

Uses Neo4j vector index to find entry-point PhraseNodes, then traverses
the graph via Cypher to collect related PhraseNodes and PassageNodes,
assembling a rich GraphContext for answer generation.

Pipeline: query_embedding → vector entry (+ query-anchor rerank) → graph traversal → context assembly.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.models import Entity, GraphContext
from rag_core.neo4j_utils import open_neo4j_session

from agentic_graph_rag.agent.query_parser import QueryStructure
from agentic_graph_rag.indexing.dual_node import PASSAGE_LABEL, PHRASE_LABEL, RELATED_TO_LABEL

if TYPE_CHECKING:
    from neo4j import Driver

logger = logging.getLogger(__name__)

# Vector index on PhraseNode embeddings (created during indexing)
PHRASE_INDEX_NAME = "phrase_node_index"

# Entry-point oversampling factor: fetch N× candidates then rerank by anchor hits.
_ENTRY_OVERSAMPLE_FACTOR = 3

# Regex for extracting query anchor terms (CJK runs + Latin tokens).
_ANCHOR_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9][A-Za-z0-9_.-]*")
_ANCHOR_STOPWORDS = {
    "什么", "如何", "怎么", "是否", "哪些", "多少", "为什么",
    "what", "how", "which", "when", "where", "does", "should",
    "the", "and", "for", "with", "from",
}

_RELATION_INTENT_TERMS: dict[str, tuple[str, ...]] = {
    "诊断": ("诊断", "标准", "符合", "提示"),
    "推荐": ("推荐", "首选", "适用", "用于"),
    "禁忌": ("禁忌", "禁用", "避免", "慎用"),
    "替代": ("替代", "换用", "改用"),
    "疗程": ("疗程", "持续", "时长", "至少"),
    "剂量": ("剂量", "用量", "滴定"),
    "副作用": ("不良反应", "副作用", "风险"),
    "比较": ("比较", "优于", "劣于", "区别"),
    "目标值": ("目标", "目标值", "控制目标", "阈值"),
    "处理": ("处理", "应对", "管理"),
    "机制": ("机制", "作用", "原理"),
    "预后": ("预后", "结局", "风险"),
}

_RELATION_INTENT_ENTITY_HINTS: dict[str, tuple[str, ...]] = {
    "诊断": ("Disease", "Test", "Biomarker", "Threshold"),
    "推荐": ("Drug", "DrugClass", "Therapy", "Procedure", "Device"),
    "禁忌": ("Drug", "DrugClass", "Therapy", "Threshold", "Disease"),
    "替代": ("Drug", "DrugClass", "Therapy", "Device"),
    "疗程": ("Therapy", "Procedure", "Threshold", "Device"),
    "剂量": ("Drug", "DrugClass", "Therapy", "Threshold"),
    "副作用": ("Drug", "DrugClass", "Therapy", "Disease", "Symptom"),
    "比较": ("Drug", "DrugClass", "Therapy", "Procedure", "Device"),
    "目标值": ("Threshold", "Test", "Biomarker"),
    "处理": ("Drug", "DrugClass", "Therapy", "Procedure"),
    "机制": ("Drug", "DrugClass", "Therapy", "Procedure", "Biomarker"),
    "预后": ("Disease", "Threshold", "Biomarker", "Population"),
}


def _extract_query_anchors(query: str) -> list[str]:
    """Extract high-signal anchor terms from a query for entry-point reranking.

    Keeps entity-like tokens (drug names, lab values, abbreviations) and
    drops generic question words. These anchors are used to boost
    PhraseNodes whose name/aliases contain them.
    """
    tokens: list[str] = []
    for match in _ANCHOR_TOKEN_RE.findall(query):
        token = match.strip()
        if not token or token.lower() in _ANCHOR_STOPWORDS:
            continue
        if len(token) < 2:
            continue
        if token.lower() not in tokens:
            tokens.append(token.lower())
    return tokens


def _anchor_hits(entry: dict, anchors: list[str]) -> int:
    """Count how many query anchors appear in a PhraseNode's name or aliases."""
    if not anchors:
        return 0
    # Build a searchable surface from name + aliases
    name = (entry.get("name") or "").lower()
    aliases_raw = entry.get("aliases") or []
    aliases_text = " ".join(str(a).lower() for a in aliases_raw) if aliases_raw else ""
    surface = f"{name} {aliases_text}"
    return sum(1 for anchor in anchors if anchor in surface)


def _entry_sort_key(entry: dict, anchors: list[str], focus_terms: tuple[str, ...]) -> tuple[int, int, float, float]:
    """Prefer query-specific entry points over generic hubs."""
    focus_hits = 0
    if focus_terms:
        surface = f"{entry.get('name', '')} {' '.join(str(a) for a in entry.get('aliases') or [])}".lower()
        focus_hits = sum(1 for term in focus_terms if term.lower() in surface)
    return (
        focus_hits,
        _anchor_hits(entry, anchors),
        float(entry.get("pagerank_score") or 0.0),
        float(entry.get("score") or 0.0),
    )


def _normalize_surface(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").casefold())


def _query_entity_signatures(query_structure: QueryStructure | None) -> set[str]:
    if query_structure is None:
        return set()
    values = [
        *query_structure.focus_entities,
        *query_structure.background_entities,
        *query_structure.constraints,
    ]
    return {
        signature
        for signature in (_normalize_surface(value) for value in values)
        if len(signature) >= 2
    }


def _matches_query_structure(name: str, query_structure: QueryStructure | None) -> bool:
    if query_structure is None:
        return False
    normalized_name = _normalize_surface(name)
    if not normalized_name:
        return False
    return any(
        signature in normalized_name or normalized_name in signature
        for signature in _query_entity_signatures(query_structure)
    )


def _preferred_entity_types(query_structure: QueryStructure | None) -> tuple[str, ...]:
    if query_structure is None or not query_structure.relation_intent:
        return ()
    return _RELATION_INTENT_ENTITY_HINTS.get(query_structure.relation_intent, ())


def _select_relation_label(query_structure: QueryStructure | None) -> str:
    if query_structure is None or not query_structure.relation_intent:
        return "CO_OCCURS_WITH"
    return query_structure.relation_intent


def _sentence_splits(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    return [part.strip() for part in parts if part.strip()]


def _sentence_supports_relation(
    sentence: str,
    *,
    seed_name: str,
    neighbor_name: str,
    expansion_terms: list[str],
    intent_terms: list[str],
) -> bool:
    lowered = sentence.casefold()
    if seed_name.casefold() not in lowered or neighbor_name.casefold() not in lowered:
        return False
    if intent_terms and not any(term.casefold() in lowered for term in intent_terms):
        return False
    matched_terms = sum(1 for term in expansion_terms if term.casefold() in lowered)
    return matched_terms >= 2 if expansion_terms else True


def _build_weak_triplet(
    *,
    seed_name: str,
    neighbor_name: str,
    relation_label: str,
    sentence: str,
) -> dict[str, str]:
    return {
        "source": seed_name,
        "relation": relation_label,
        "target": neighbor_name,
        "evidence": sentence.strip(),
        "strength": "weak",
    }


def _entity_type_rank(entity_type: str, preferred_types: tuple[str, ...]) -> int:
    if not preferred_types:
        return 0
    normalized = str(entity_type or "").casefold()
    return 1 if any(normalized == item.casefold() for item in preferred_types) else 0


def _collect_rule_entry_points(
    driver: Driver,
    *,
    top_k: int,
    query_structure: QueryStructure | None,
) -> list[dict]:
    """Find explicit entity entry points from PhraseNode name/aliases."""
    if query_structure is None:
        return []

    lookup_terms = list(query_structure.focus_entities or query_structure.all_entities)
    if not lookup_terms:
        return []

    with open_neo4j_session(driver) as session:
        result = session.run(
            f"""
            UNWIND $terms AS term
            MATCH (p:{PHRASE_LABEL})
            WHERE toLower(p.name) CONTAINS toLower(term)
               OR any(alias IN coalesce(p.aliases, []) WHERE toLower(alias) CONTAINS toLower(term))
            RETURN DISTINCT
                p.id AS id,
                p.name AS name,
                p.entity_type AS entity_type,
                p.pagerank_score AS pagerank_score,
                coalesce(p.aliases, []) AS aliases,
                term AS matched_term
            LIMIT $top_k
            """,
            terms=lookup_terms,
            top_k=max(top_k * 2, len(lookup_terms)),
        )
        candidates: list[dict] = []
        for record in result:
            candidates.append({
                "id": record["id"],
                "name": record["name"] or "",
                "entity_type": record["entity_type"] or "",
                "pagerank_score": record["pagerank_score"] or 0.0,
                "aliases": record["aliases"] or [],
                "matched_term": record["matched_term"] or "",
                "score": 1.0,
            })
    return candidates


def _collect_passage_seed_entry_points(
    driver: Driver,
    *,
    top_k: int,
    query_structure: QueryStructure | None,
) -> list[dict]:
    """Find entry points from passages that lexically match the query structure."""
    if query_structure is None:
        return []

    seed_terms = list(query_structure.seed_terms)
    background_terms = list(query_structure.background_entities)
    if not seed_terms:
        return []

    with open_neo4j_session(driver) as session:
        result = session.run(
            f"""
            MATCH (p:{PASSAGE_LABEL})<-[:MENTIONED_IN]-(ph:{PHRASE_LABEL})
            WITH p, ph,
                 size([term IN $seed_terms WHERE toLower(p.text) CONTAINS toLower(term)]) AS seed_hits,
                 size([term IN $background_terms WHERE toLower(p.text) CONTAINS toLower(term)]) AS background_hits
            WHERE seed_hits >= 2 OR (seed_hits >= 1 AND background_hits >= 1)
            RETURN
                ph.id AS id,
                ph.name AS name,
                ph.entity_type AS entity_type,
                ph.pagerank_score AS pagerank_score,
                coalesce(ph.aliases, []) AS aliases,
                max(seed_hits) AS hit_count,
                max(background_hits) AS background_hits
            ORDER BY hit_count DESC, background_hits DESC, ph.pagerank_score DESC
            LIMIT $top_k
            """,
            seed_terms=seed_terms,
            background_terms=background_terms,
            top_k=max(top_k * 3, len(seed_terms) * 2),
        )
        candidates: list[dict] = []
        for record in result:
            candidates.append({
                "id": record["id"],
                "name": record["name"] or "",
                "entity_type": record["entity_type"] or "",
                "pagerank_score": record["pagerank_score"] or 0.0,
                "aliases": record["aliases"] or [],
                "score": float(record["hit_count"] or 0.0),
            })
    return candidates


# ---------------------------------------------------------------------------
# 1. Find entry points via vector search + query-anchor rerank
# ---------------------------------------------------------------------------

def find_entry_points(
    query_embedding: list[float],
    driver: Driver,
    top_k: int | None = None,
    threshold: float | None = None,
    query: str = "",
    query_structure: QueryStructure | None = None,
) -> list[dict]:
    """Find nearest PhraseNodes via Neo4j vector index, then rerank by
    query-anchor hits on name/aliases.

    The oversample-then-rerank strategy ensures that PhraseNodes whose name
    exactly matches a query entity (e.g. "二甲双胍") are promoted above
    generic high-frequency nodes (e.g. "2型糖尿病") that happen to have
    similar embeddings.

    Returns list of dicts with keys: id, name, entity_type, score.
    """
    cfg = get_settings()
    if top_k is None:
        top_k = cfg.retrieval.graph_entry_top_k
    if threshold is None:
        threshold = cfg.retrieval.vector_threshold

    # Oversample: fetch more candidates than needed so reranking has room.
    fetch_k = top_k * _ENTRY_OVERSAMPLE_FACTOR

    with open_neo4j_session(driver) as session:
        result = session.run(
            f"""
            CALL db.index.vector.queryNodes(
                '{PHRASE_INDEX_NAME}', $top_k, $embedding
            )
            YIELD node, score
            WHERE score >= $threshold
            RETURN node.id AS id,
                   node.name AS name,
                   node.entity_type AS entity_type,
                   node.pagerank_score AS pagerank_score,
                   coalesce(node.aliases, []) AS aliases,
                   score
            ORDER BY score DESC
            """,
            top_k=fetch_k,
            embedding=query_embedding,
            threshold=threshold,
        )

        candidates = []
        for record in result:
            try:
                aliases = record["aliases"] or []
            except (KeyError, TypeError):
                aliases = []
            candidates.append({
                "id": record["id"],
                "name": record["name"] or "",
                "entity_type": record["entity_type"] or "",
                "pagerank_score": record["pagerank_score"] or 0.0,
                "aliases": aliases,
                "score": record["score"],
            })

    rule_candidates = _collect_rule_entry_points(
        driver,
        top_k=top_k,
        query_structure=query_structure,
    )
    passage_seed_candidates = _collect_passage_seed_entry_points(
        driver,
        top_k=top_k,
        query_structure=query_structure,
    )
    merged_candidates: dict[str, dict] = {}
    for candidate in [*rule_candidates, *passage_seed_candidates, *candidates]:
        candidate_id = str(candidate.get("id") or "")
        if not candidate_id:
            continue
        existing = merged_candidates.get(candidate_id)
        if existing is None or float(candidate.get("score") or 0.0) > float(existing.get("score") or 0.0):
            merged_candidates[candidate_id] = candidate

    # Rerank by query-anchor hits using lexicographic priority:
    # 1st key: anchor hit count (more hits = more relevant entry point)
    # 2nd key: cosine score (tiebreaker among same hit count)
    # No magic numbers — just priority ordering.
    anchors = _extract_query_anchors(query)
    focus_terms = tuple(query_structure.focus_entities) if query_structure is not None else ()
    reranked = list(merged_candidates.values())
    if anchors or focus_terms:
        reranked.sort(
            key=lambda e: _entry_sort_key(e, anchors, focus_terms),
            reverse=True,
        )
    else:
        reranked.sort(key=lambda e: float(e.get("score") or 0.0), reverse=True)

    entries = reranked[:top_k]

    logger.info(
        "Found %d entry points (vector=%d, rule=%d, passage_seed=%d, top_k=%d, threshold=%.2f, anchors=%s)",
        len(entries),
        len(candidates),
        len(rule_candidates),
        len(passage_seed_candidates),
        top_k,
        threshold,
        anchors[:5],
    )
    return entries


# ---------------------------------------------------------------------------
# 2. Graph traversal from entry points
# ---------------------------------------------------------------------------

def traverse_graph(
    entry_ids: list[str],
    driver: Driver,
    max_hops: int | None = None,
    cooccurrence_limit: int | None = None,
    passage_limit: int | None = None,
    query_structure: QueryStructure | None = None,
) -> dict:
    """Traverse graph from entry PhraseNodes with optional constraints.

    When `focus_entities` or `relation_intent` are provided, the traversal
    is constrained: only nodes/edges that match the query structure are
    kept. This prevents unconstrained BFS from pulling in generic clusters.

    Returns dict with keys:
      - phrase_nodes: list of {id, name, entity_type}
      - passage_nodes: list of {id, text, chunk_id}
      - relationships: list of {source, relation, target}
    """
    if not entry_ids:
        return {"phrase_nodes": [], "passage_nodes": [], "relationships": []}

    cfg = get_settings()
    if max_hops is None:
        max_hops = cfg.retrieval.max_hops
    if cooccurrence_limit is None:
        cooccurrence_limit = cfg.retrieval.graph_cooccurrence_limit
    if passage_limit is None:
        passage_limit = cfg.retrieval.graph_passage_limit

    expansion_terms = list(query_structure.expansion_terms) if query_structure is not None else []
    intent_terms = list(_RELATION_INTENT_TERMS.get(query_structure.relation_intent, ())) if query_structure else []
    preferred_entity_types = _preferred_entity_types(query_structure)
    relation_label = _select_relation_label(query_structure)

    phrase_nodes: dict[str, dict] = {}
    passage_nodes: dict[str, dict] = {}
    relationships: list[dict[str, str]] = []
    weak_relationships: list[dict[str, str]] = []

    with open_neo4j_session(driver) as session:
        # Step 1: Traverse inter-PhraseNode RELATED_TO edges up to max_hops
        result = session.run(
            f"""
            MATCH (start:{PHRASE_LABEL})
            WHERE start.id IN $entry_ids
            MATCH path = (start)-[r:{RELATED_TO_LABEL}*1..{max_hops}]-(connected:{PHRASE_LABEL})
            UNWIND relationships(path) AS rel
            WITH path, start, connected, rel,
                 startNode(rel) AS src, endNode(rel) AS tgt
            WHERE (
                size($expansion_terms) = 0
                OR any(term IN $expansion_terms WHERE toLower(connected.name) CONTAINS toLower(term))
                OR any(term IN $expansion_terms WHERE toLower(src.name) CONTAINS toLower(term))
                OR any(term IN $expansion_terms WHERE toLower(tgt.name) CONTAINS toLower(term))
            )
            RETURN DISTINCT
                connected.id AS connected_id,
                connected.name AS connected_name,
                connected.entity_type AS connected_type,
                coalesce(connected.pagerank_score, 0.0) AS connected_pagerank,
                src.id AS src_id, src.name AS src_name,
                coalesce(rel.relation_type, type(rel)) AS rel_type,
                tgt.id AS tgt_id, tgt.name AS tgt_name,
                length(path) AS path_length
            ORDER BY path_length ASC, connected_pagerank DESC
            """,
            entry_ids=entry_ids,
            expansion_terms=expansion_terms,
            intent_terms=intent_terms,
        )

        for record in result:
            cid = record["connected_id"]
            connected_name = record["connected_name"] or ""
            rel_type = record["rel_type"] or ""

            if cid and cid not in phrase_nodes:
                phrase_nodes[cid] = {
                    "id": cid,
                    "name": connected_name,
                    "entity_type": record["connected_type"] or "",
                    "pagerank_score": float(record["connected_pagerank"] or 0.0),
                }

            relation_row = {
                "source": record["src_name"] or record["src_id"] or "",
                "relation": rel_type,
                "target": record["tgt_name"] or record["tgt_id"] or "",
            }
            relation_surface = " ".join(str(value) for value in relation_row.values())
            if (
                intent_terms
                and not any(term.casefold() in relation_surface.casefold() for term in intent_terms)
                and not _matches_query_structure(connected_name, query_structure)
            ):
                continue
            relationships.append(relation_row)

        # Also include entry nodes themselves
        for eid in entry_ids:
            if eid not in phrase_nodes:
                r = session.run(
                    f"""
                    MATCH (p:{PHRASE_LABEL} {{id: $id}})
                    RETURN p.id AS id,
                           p.name AS name,
                           p.entity_type AS entity_type,
                           coalesce(p.pagerank_score, 0.0) AS pagerank_score
                    """,
                    id=eid,
                )
                rec = r.single()
                if rec:
                    phrase_nodes[eid] = {
                        "id": rec["id"],
                        "name": rec["name"] or "",
                        "entity_type": rec["entity_type"] or "",
                        "pagerank_score": float(rec["pagerank_score"] or 0.0),
                    }

        # Step 2: Expand via MENTIONED_IN co-occurrence (1-hop).
        # Two PhraseNodes that appear in the same PassageNode are related
        # even without an explicit RELATED_TO edge. This recovers the
        # breadth that RELATED_TO-only traversal misses (46% of nodes
        # are isolated). Hybrid reranks the final candidate pool with a cross-encoder.
        cooccur_seed_ids = list(phrase_nodes.keys())
        if cooccur_seed_ids:
            result = session.run(
                f"""
                MATCH (seed:{PHRASE_LABEL})-[:MENTIONED_IN]->(pa:{PASSAGE_LABEL})
                      <-[:MENTIONED_IN]-(neighbor:{PHRASE_LABEL})
                WHERE seed.id IN $seed_ids AND NOT neighbor.id IN $seed_ids
                  AND (
                    size($expansion_terms) = 0
                    OR any(term IN $expansion_terms WHERE toLower(neighbor.name) CONTAINS toLower(term))
                    OR any(term IN $expansion_terms WHERE toLower(pa.text) CONTAINS toLower(term))
                  )
                RETURN DISTINCT
                    neighbor.id AS id,
                    neighbor.name AS name,
                    neighbor.entity_type AS entity_type,
                    coalesce(neighbor.pagerank_score, 0.0) AS pagerank_score,
                    pa.text AS passage_text
                ORDER BY pagerank_score DESC
                LIMIT $cooccurrence_limit
                """,
                seed_ids=cooccur_seed_ids,
                cooccurrence_limit=cooccurrence_limit,
                expansion_terms=expansion_terms,
            )
            for record in result:
                nid = record["id"]
                neighbor_name = record["name"] or ""
                neighbor_type = record["entity_type"] or ""
                neighbor_pagerank = float(record["pagerank_score"] or 0.0)
                passage_text = record["passage_text"] or ""
                if (
                    preferred_entity_types
                    and not _entity_type_rank(neighbor_type, preferred_entity_types)
                    and not any(term.casefold() in passage_text.casefold() for term in intent_terms)
                    and not _matches_query_structure(neighbor_name, query_structure)
                ):
                    continue
                if nid and nid not in phrase_nodes:
                    phrase_nodes[nid] = {
                        "id": nid,
                        "name": neighbor_name,
                        "entity_type": neighbor_type,
                        "pagerank_score": neighbor_pagerank,
                    }
                if neighbor_name:
                    for seed_id in cooccur_seed_ids:
                        seed = phrase_nodes.get(seed_id)
                        if seed is None:
                            continue
                        seed_name = seed.get("name", "")
                        if not seed_name or seed_name == neighbor_name:
                            continue
                        supporting_sentence = next(
                            (
                                sentence
                                for sentence in _sentence_splits(passage_text)
                                if _sentence_supports_relation(
                                    sentence,
                                    seed_name=seed_name,
                                    neighbor_name=neighbor_name,
                                    expansion_terms=expansion_terms,
                                    intent_terms=intent_terms,
                                )
                            ),
                            "",
                        )
                        if not supporting_sentence:
                            continue
                        weak_relationships.append(
                            _build_weak_triplet(
                                seed_name=seed_name,
                                neighbor_name=neighbor_name,
                                relation_label=relation_label,
                                sentence=supporting_sentence,
                            )
                        )

        # Step 3: Collect PassageNodes linked to all discovered PhraseNodes
        all_phrase_ids = list(phrase_nodes.keys())
        if all_phrase_ids:
            result = session.run(
                f"""
                MATCH (ph:{PHRASE_LABEL})-[:MENTIONED_IN]->(pa:{PASSAGE_LABEL})
                WHERE ph.id IN $phrase_ids
                  AND (
                    size($expansion_terms) = 0
                    OR any(term IN $expansion_terms WHERE toLower(pa.text) CONTAINS toLower(term))
                )
                WITH DISTINCT
                    pa.id AS id,
                    pa.text AS text,
                    pa.chunk_id AS chunk_id,
                    max(coalesce(ph.pagerank_score, 0.0)) AS phrase_pagerank
                WITH id, text, chunk_id, phrase_pagerank,
                     CASE
                       WHEN size($expansion_terms) = 0 THEN 0
                       ELSE size([term IN $expansion_terms WHERE toLower(text) CONTAINS toLower(term)])
                     END AS term_hits
                RETURN id, text, chunk_id, phrase_pagerank
                ORDER BY term_hits DESC, phrase_pagerank DESC
                LIMIT $passage_limit
                """,
                phrase_ids=all_phrase_ids,
                passage_limit=passage_limit,
                expansion_terms=expansion_terms,
            )

            for record in result:
                pid = record["id"]
                if pid and pid not in passage_nodes:
                    text = record["text"] or ""
                    if (
                        intent_terms
                        and not any(term.casefold() in text.casefold() for term in intent_terms)
                        and not any(term.casefold() in text.casefold() for term in expansion_terms)
                    ):
                        continue
                    passage_nodes[pid] = {
                        "id": pid,
                        "text": text,
                        "chunk_id": record["chunk_id"] or "",
                        "pagerank_score": float(record["phrase_pagerank"] or 0.0),
                    }

    logger.info(
        "Traversal from %d entries: %d phrases, %d passages, %d relationships, %d weak relationships",
        len(entry_ids), len(phrase_nodes), len(passage_nodes), len(relationships), len(weak_relationships),
    )
    return {
        "phrase_nodes": list(phrase_nodes.values()),
        "passage_nodes": list(passage_nodes.values()),
        "relationships": relationships,
        "weak_relationships": weak_relationships,
    }


# ---------------------------------------------------------------------------
# 3. Collect and assemble context
# ---------------------------------------------------------------------------

def collect_context(traversal_result: dict) -> GraphContext:
    """Assemble GraphContext from traversal results.

    Combines triplets (relationships) with passage texts and entity info.
    """
    triplets = []
    for rel in traversal_result.get("relationships", []):
        triplets.append({
            "source": rel.get("source", ""),
            "relation": rel.get("relation", ""),
            "target": rel.get("target", ""),
        })
    weak_triplets = []
    for rel in traversal_result.get("weak_relationships", []):
        weak_triplets.append({
            "source": rel.get("source", ""),
            "relation": rel.get("relation", ""),
            "target": rel.get("target", ""),
            "evidence": rel.get("evidence", ""),
            "strength": rel.get("strength", "weak"),
        })

    # Deduplicate triplets
    seen = set()
    unique_triplets = []
    for t in triplets:
        key = (t["source"], t["relation"], t["target"])
        if key not in seen:
            seen.add(key)
            unique_triplets.append(t)

    passages = [
        p["text"] for p in traversal_result.get("passage_nodes", [])
        if p.get("text")
    ]

    entities = [
        Entity(
            id=p.get("id", ""),
            name=p.get("name", ""),
            entity_type=p.get("entity_type", ""),
        )
        for p in traversal_result.get("phrase_nodes", [])
    ]

    source_ids = [
        p["chunk_id"] for p in traversal_result.get("passage_nodes", [])
        if p.get("chunk_id")
    ]

    return GraphContext(
        triplets=unique_triplets,
        weak_triplets=weak_triplets,
        passages=passages,
        entities=entities,
        source_ids=source_ids,
    )


# ---------------------------------------------------------------------------
# 4. Full VectorCypher search pipeline
# ---------------------------------------------------------------------------

def search(
    query_embedding: list[float],
    driver: Driver,
    top_k: int | None = None,
    max_hops: int | None = None,
    threshold: float | None = None,
    cooccurrence_limit: int | None = None,
    passage_limit: int | None = None,
    query: str = "",
    query_structure: QueryStructure | None = None,
) -> GraphContext:
    """Full VectorCypher retrieval pipeline.

    1. Find entry PhraseNodes via vector similarity + query-anchor rerank
    2. Traverse graph from entries with optional constraints
    3. Collect and assemble context

    When `focus_entities` / `relation_intent` are provided (from QueryStructure),
    the traversal is constrained to only follow relevant edges/nodes.

    Returns GraphContext with triplets, passages, entities, source_ids.
    """
    # Step 1: Vector entry with query-anchor reranking
    entries = find_entry_points(
        query_embedding,
        driver,
        top_k=top_k,
        threshold=threshold,
        query=query,
        query_structure=query_structure,
    )

    if not entries:
        logger.warning("No entry points found for query")
        return GraphContext()

    entry_ids = [e["id"] for e in entries]

    # Step 2: Constrained graph traversal
    traversal = traverse_graph(
        entry_ids,
        driver,
        max_hops=max_hops,
        cooccurrence_limit=cooccurrence_limit,
        passage_limit=passage_limit,
        query_structure=query_structure,
    )

    # Step 3: Assemble context
    context = collect_context(traversal)

    logger.info(
        "VectorCypher search: %d entries → %d triplets, %d passages",
        len(entries), len(context.triplets), len(context.passages),
    )
    return context
