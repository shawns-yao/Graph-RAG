"""Dual-node graph structure for HippoRAG 2.

Creates and manages PhraseNode (entity-level) and PassageNode (full-text)
nodes in Neo4j, linked via MENTIONED_IN relationships.

Also provides Personalized PageRank (PPR) for query-focused retrieval.

Reference: HippoRAG 2 (ICML 2025) — F1 +7.1 on MuSiQue, 12x fewer tokens.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

import networkx as nx
from rag_core.config import get_settings
from rag_core.models import Chunk, Entity, PassageNode, PhraseNode, Relationship
from rag_core.neo4j_utils import open_neo4j_session

from agentic_graph_rag.text_signals import TfidfProfile, best_term_idf, build_tfidf_profile, extract_terms

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

logger = logging.getLogger(__name__)

PHRASE_LABEL = "PhraseNode"
PASSAGE_LABEL = "PassageNode"
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_DEFAULT_PHRASE_EMBED_BATCH_SIZE = 64
_MAX_MENTIONED_IN_PER_ENTITY = 3
_MAX_MENTION_SCORE_OCCURRENCES = 3
_ALIAS_FUZZY_MATCH_THRESHOLD = 0.88
_COMMON_ENTITY_NAMES = {
    "patient",
    "patients",
    "treatment",
    "treatments",
    "method",
    "methods",
    "result",
    "results",
    "study",
    "studies",
    "data",
    "analysis",
    "conclusion",
    "conclusions",
    "患者",
    "治疗",
    "方法",
    "结果",
    "研究",
    "数据",
    "分析",
    "结论",
    "目的",
}


def _coerce_confidence(value: object) -> float | None:
    """Convert optional confidence metadata to a bounded float."""
    if value is None or value == "":
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


# ---------------------------------------------------------------------------
# 1. Create PhraseNodes in Neo4j
# ---------------------------------------------------------------------------

def create_phrase_nodes(
    entities: list[Entity],
    driver: Driver,
    pagerank_scores: dict[str, float] | None = None,
) -> list[PhraseNode]:
    """Create PhraseNode nodes in Neo4j from extracted entities.

    Args:
        entities: Entities to create as graph nodes.
        driver: Neo4j driver.
        pagerank_scores: Optional mapping entity_id → pagerank score.

    Returns list of created PhraseNode objects.
    """
    if not entities:
        return []

    entities = _merge_entities_by_explicit_aliases(entities)
    entities = _canonicalize_entities_against_existing_nodes(entities, driver)
    phrase_nodes: list[PhraseNode] = []
    scores = pagerank_scores or {}

    with open_neo4j_session(driver) as session:
        for entity in entities:
            eid = entity.id or hashlib.md5(entity.name.lower().encode()).hexdigest()[:8]
            pr_score = scores.get(eid, 0.0)
            confidence = _coerce_confidence(
                getattr(entity, "entity_confidence", entity.metadata.get("confidence"))
            ) or 0.0

            session.run(
                f"""
                MERGE (p:{PHRASE_LABEL} {{id: $id}})
                SET p.name = $name,
                    p.entity_type = $entity_type,
                    p.description = $description,
                    p.pagerank_score = $pagerank_score,
                    p.confidence = $confidence,
                    p.confidence_count = $confidence_count,
                    p.confidence_max = $confidence_max,
                    p.aliases = $aliases
                """,
                id=eid,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                pagerank_score=pr_score,
                confidence=confidence,
                confidence_count=int(entity.metadata.get("confidence_count", 1)),
                confidence_max=_coerce_confidence(
                    entity.metadata.get("confidence_max", confidence)
                ) or confidence,
                aliases=_medical_aliases(entity),
            )

            phrase_nodes.append(PhraseNode(
                id=eid,
                name=entity.name,
                entity_type=entity.entity_type,
                pagerank_score=pr_score,
                confidence=confidence,
            ))

    logger.info("Created %d PhraseNodes in Neo4j", len(phrase_nodes))
    return phrase_nodes


def _merge_entities_by_explicit_aliases(entities: list[Entity]) -> list[Entity]:
    """Merge same-type entities when explicit name/alias surfaces overlap."""
    merged: list[Entity] = []
    for entity in entities:
        target: Entity | None = None
        entity_type = entity.entity_type.strip().casefold()
        entity_signatures = _entity_alias_signatures(entity)
        for candidate in merged:
            if candidate.entity_type.strip().casefold() != entity_type:
                continue
            if _entity_alias_signatures(candidate) & entity_signatures:
                target = candidate
                break
        if target is None:
            merged.append(entity.model_copy(deep=True))
            continue

        aliases = _sorted_aliases([
            target.name,
            *(str(alias) for alias in target.metadata.get("aliases", [])),
            entity.name,
            *(str(alias) for alias in entity.metadata.get("aliases", [])),
        ])
        metadata = dict(target.metadata)
        metadata["aliases"] = aliases
        merged[merged.index(target)] = target.model_copy(
            update={
                "name": _preferred_canonical_name(target.name, entity.name),
                "description": target.description or entity.description,
                "metadata": metadata,
                "entity_confidence": max(target.entity_confidence, entity.entity_confidence),
            }
        )
    return merged


def _entity_alias_signatures(entity: Entity) -> set[str]:
    signatures: set[str] = set()
    for surface in [entity.name, *entity.metadata.get("aliases", [])]:
        for signature in _relationship_alias_signatures(str(surface)):
            signatures.add(signature)
    return signatures


def _preferred_canonical_name(left: str, right: str) -> str:
    return max([left, right], key=_canonical_name_rank)


def _canonical_name_rank(name: str) -> tuple[int, int, int]:
    text = name.strip()
    acronym = 1 if re.fullmatch(r"[A-Z0-9]{2,10}", text) else 0
    cjk = 1 if _CJK_RE.search(text) else 0
    return (cjk, -acronym, len(text))


def _canonicalize_entities_against_existing_nodes(
    entities: list[Entity],
    driver: Driver,
) -> list[Entity]:
    """Reuse existing PhraseNodes when explicit aliases identify the same type."""
    if not entities:
        return []

    with open_neo4j_session(driver) as session:
        rows = session.run(
            f"""
            MATCH (p:{PHRASE_LABEL})
            RETURN p.id AS id,
                   p.name AS name,
                   p.entity_type AS entity_type,
                   p.description AS description,
                   p.aliases AS aliases
            """
        ).data()

    if not rows:
        return entities

    by_type_signature: dict[tuple[str, str], dict] = {}
    for row in rows:
        entity_type = str(row.get("entity_type") or "").strip().casefold()
        surfaces = [
            str(row.get("name") or ""),
            *(str(alias) for alias in (row.get("aliases") or [])),
        ]
        for surface in surfaces:
            for signature in _relationship_alias_signatures(surface):
                by_type_signature[(entity_type, signature)] = row

    canonicalized: list[Entity] = []
    for entity in entities:
        entity_type = entity.entity_type.strip().casefold()
        surfaces = [entity.name, *entity.metadata.get("aliases", [])]
        match: dict | None = None
        for surface in surfaces:
            for signature in _relationship_alias_signatures(str(surface)):
                match = by_type_signature.get((entity_type, signature))
                if match:
                    break
            if match:
                break
        if match is None:
            canonicalized.append(entity)
            continue

        aliases = _sorted_aliases([
            str(match.get("name") or ""),
            *(str(alias) for alias in (match.get("aliases") or [])),
            entity.name,
            *(str(alias) for alias in entity.metadata.get("aliases", [])),
        ])
        metadata = dict(entity.metadata)
        metadata["aliases"] = aliases
        canonicalized.append(
            entity.model_copy(
                update={
                    "id": str(match.get("id") or entity.id),
                    "name": str(match.get("name") or entity.name),
                    "entity_type": str(match.get("entity_type") or entity.entity_type),
                    "description": entity.description or str(match.get("description") or ""),
                    "metadata": metadata,
                }
            )
        )
    return canonicalized


def persist_entity_alias_metadata(
    entities: list[Entity],
    driver: Driver,
) -> None:
    """Persist learned alias metadata onto PhraseNode records."""
    if not entities:
        return

    with open_neo4j_session(driver) as session:
        for entity in entities:
            eid = entity.id or hashlib.md5(entity.name.lower().encode()).hexdigest()[:8]
            session.run(
                f"""
                MATCH (p:{PHRASE_LABEL} {{id: $id}})
                SET p.aliases = $aliases
                """,
                id=eid,
                aliases=_medical_aliases(entity),
            )


# ---------------------------------------------------------------------------
# 2. Create PassageNodes in Neo4j
# ---------------------------------------------------------------------------

def create_passage_nodes(
    chunks: list[Chunk],
    driver: Driver,
) -> list[PassageNode]:
    """Create PassageNode nodes in Neo4j from text chunks.

    Each passage stores full text + embedding for later retrieval.
    """
    if not chunks:
        return []

    passage_nodes: list[PassageNode] = []

    with open_neo4j_session(driver) as session:
        for chunk in chunks:
            pid = chunk.id or hashlib.md5(chunk.content.encode()).hexdigest()[:8]
            embedding = list(chunk.embedding or [])
            if embedding:
                session.run(
                    f"""
                    MERGE (p:{PASSAGE_LABEL} {{id: $id}})
                    SET p.text = $text,
                        p.chunk_id = $chunk_id,
                        p.embedding = $embedding
                    """,
                    id=pid,
                    text=chunk.enriched_content,
                    chunk_id=chunk.id,
                    embedding=embedding,
                )
            else:
                logger.warning(
                    "Chunk %s has no embedding, skipping passage node creation",
                    chunk.id,
                )
                continue

            passage_nodes.append(PassageNode(
                id=pid,
                text=chunk.enriched_content,
                chunk_id=chunk.id,
                embedding=embedding,
            ))

    logger.info("Created %d PassageNodes in Neo4j", len(passage_nodes))
    return passage_nodes


# ---------------------------------------------------------------------------
# 3. Link PhraseNode → PassageNode via MENTIONED_IN
# ---------------------------------------------------------------------------

def link_phrase_to_passage(
    phrase_id: str,
    passage_id: str,
    driver: Driver,
) -> None:
    """Create MENTIONED_IN relationship between PhraseNode and PassageNode."""
    with open_neo4j_session(driver) as session:
        session.run(
            f"""
            MATCH (ph:{PHRASE_LABEL} {{id: $phrase_id}})
            MATCH (pa:{PASSAGE_LABEL} {{id: $passage_id}})
            MERGE (ph)-[:MENTIONED_IN]->(pa)
            """,
            phrase_id=phrase_id,
            passage_id=passage_id,
        )


def link_entities_to_passages(
    entities: list[Entity],
    chunks: list[Chunk],
    driver: Driver,
) -> int:
    """Link all entities to chunks where they're mentioned.

    Uses case-insensitive substring matching.
    Returns number of links created.
    """
    if not entities or not chunks:
        return 0

    tfidf_profile = _build_cross_document_profile(chunks)
    count = 0
    for entity in entities:
        if _should_skip_entity_linking(entity, tfidf_profile):
            continue
        surface_forms = _entity_surface_forms(entity)
        if not surface_forms:
            continue
        eid = entity.id or hashlib.md5(entity.name.lower().encode()).hexdigest()[:8]

        scored_passages: list[tuple[float, str]] = []
        for chunk in chunks:
            score = _chunk_entity_match_score(chunk.enriched_content, surface_forms)
            if score > 0:
                pid = chunk.id or hashlib.md5(chunk.content.encode()).hexdigest()[:8]
                scored_passages.append((score, pid))

        scored_passages.sort(key=lambda item: (-item[0], item[1]))
        seen_passage_ids: set[str] = set()
        for _score, pid in scored_passages:
            if pid in seen_passage_ids:
                continue
            link_phrase_to_passage(eid, pid, driver)
            count += 1
            seen_passage_ids.add(pid)
            if len(seen_passage_ids) >= _MAX_MENTIONED_IN_PER_ENTITY:
                break

    logger.info("Created %d MENTIONED_IN links", count)
    return count


def _should_skip_entity_linking(
    entity: Entity,
    tfidf_profile: TfidfProfile | None = None,
) -> bool:
    """Drop low-signal entities before building broad mention edges."""
    cfg = get_settings().indexing
    name = entity.name.strip()
    if len(name) < 2:
        return True

    low_idf_threshold = float(getattr(cfg, "tfidf_low_idf_threshold", 1.2))
    if tfidf_profile is not None:
        best_idf = best_term_idf(name, tfidf_profile)
        if (
            tfidf_profile.document_count >= 3
            and
            best_idf > 0
            and best_idf < low_idf_threshold
            and _entity_document_frequency_ratio(name, tfidf_profile) >= 0.8
        ):
            return True
    if name.casefold() in _COMMON_ENTITY_NAMES:
        return True

    confidence = entity.metadata.get("confidence")
    if confidence is None:
        return False

    try:
        return float(confidence) < 0.7
    except (TypeError, ValueError):
        return False


def _entity_document_frequency_ratio(
    name: str,
    tfidf_profile: TfidfProfile,
) -> float:
    surface_terms = _entity_surface_terms(name)
    if not surface_terms:
        return 0.0
    max_df = max(tfidf_profile.df(term) for term in surface_terms)
    return max_df / max(1, tfidf_profile.document_count)


def _entity_surface_terms(name: str) -> list[str]:
    return [term for term in best_term_candidates(name) if term]


def best_term_candidates(text: str) -> list[str]:
    terms = extract_terms(text)
    if terms:
        return terms
    normalized = text.strip().casefold()
    return [normalized] if normalized else []


def _build_cross_document_profile(chunks: list[Chunk]) -> TfidfProfile | None:
    """Build IDF statistics only when chunk metadata proves multi-document coverage."""
    grouped_texts: dict[str, list[str]] = {}
    for chunk in chunks:
        document_key = _chunk_document_key(chunk)
        if not document_key:
            return None
        grouped_texts.setdefault(document_key, []).append(chunk.enriched_content)

    if len(grouped_texts) < 3:
        return None
    return build_tfidf_profile(["\n".join(parts) for parts in grouped_texts.values()])


def _chunk_document_key(chunk: Chunk) -> str:
    metadata = chunk.metadata
    for key in ("document_id", "source", "source_file", "file_path"):
        value = str(metadata.get(key, "")).strip()
        if value:
            return value
    return ""


def _entity_surface_forms(entity: Entity) -> list[str]:
    """Collect entity surface forms from the canonical name plus aliases."""
    forms: list[str] = []
    values = [entity.name]
    values.extend(
        alias
        for alias in entity.metadata.get("aliases", [])
        if not _is_cross_language_alias(str(alias), entity.name)
    )
    for value in values:
        text = str(value).strip()
        if len(text) >= 2 and text.casefold() not in {item.casefold() for item in forms}:
            forms.append(text)
        normalized = _normalize_alias_text(text)
        if len(normalized) >= 2 and normalized.casefold() not in {item.casefold() for item in forms}:
            forms.append(normalized)
    return forms


def _chunk_mentions_entity(text: str, surface_forms: list[str]) -> bool:
    """Match entity mentions using alias-aware, language-safe checks."""
    return _chunk_entity_match_score(text, surface_forms) > 0


def _chunk_entity_match_score(text: str, surface_forms: list[str]) -> float:
    """Score how confidently a chunk mentions an entity."""
    lowered = text.casefold()
    normalized_text = _normalize_alias_text(text)
    text_tokens = _token_signature(text)
    best_score = 0.0
    for form in surface_forms:
        normalized = form.strip()
        if not normalized:
            continue
        if _CJK_RE.search(normalized):
            occurrences = lowered.count(normalized.casefold())
            if occurrences > 0:
                best_score = max(
                    best_score,
                    min(occurrences, _MAX_MENTION_SCORE_OCCURRENCES) * max(1.0, len(normalized) / 4.0),
                )
            continue

        escaped = re.escape(normalized)
        whole_word_matches = re.findall(
            rf"(?<![0-9A-Za-z_]){escaped}(?![0-9A-Za-z_])",
            text,
            re.IGNORECASE,
        )
        if whole_word_matches:
            best_score = max(
                best_score,
                min(len(whole_word_matches), _MAX_MENTION_SCORE_OCCURRENCES) * max(1.0, len(normalized) / 5.0),
            )
            continue
        if len(form) > 4 and form.lower() in lowered:
            best_score = max(best_score, max(0.5, len(normalized) / 12.0))
            continue

        canonical_form = _normalize_alias_text(normalized)
        if canonical_form and len(canonical_form) >= 5 and canonical_form in normalized_text:
            best_score = max(best_score, max(0.8, len(canonical_form) / 10.0))
            continue

        abbreviation = _abbreviation_signature(normalized)
        if abbreviation and abbreviation in text_tokens:
            best_score = max(best_score, 0.95)
            continue

        fuzzy_score = _alias_fuzzy_similarity(normalized, text_tokens)
        if fuzzy_score >= _ALIAS_FUZZY_MATCH_THRESHOLD:
            best_score = max(best_score, fuzzy_score)
    return best_score


def _normalize_alias_text(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"[\s\-_/.]+", "", text)
    return re.sub(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", "", text)


def _sorted_aliases(values: list[str]) -> list[str]:
    aliases = [str(value).strip() for value in values if str(value).strip()]
    return sorted(dict.fromkeys(aliases), key=str.casefold)


def _medical_aliases(entity: Entity) -> list[str]:
    return _sorted_aliases([str(alias) for alias in entity.metadata.get("aliases", [])])


def _json_dump(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _token_signature(text: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9_-]*", text)
        if len(token.strip()) >= 2
    }
    normalized_tokens = {_normalize_alias_text(token) for token in tokens}
    return {token for token in tokens | normalized_tokens if token}


def _abbreviation_signature(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    if len(words) < 2:
        return ""
    return "".join(word[0] for word in words).casefold()


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def _is_medical_abbreviation_alias(alias: str, name: str) -> bool:
    alias_text = alias.strip()
    if not re.fullmatch(r"[A-Za-z0-9]{2,10}", alias_text):
        return False
    name_abbr = _abbreviation_signature(name)
    return bool(name_abbr and alias_text.casefold() == name_abbr)


def _is_cross_language_alias(alias: str, name: str) -> bool:
    if _is_medical_abbreviation_alias(alias, name):
        return False
    return (
        _contains_cjk(alias) != _contains_cjk(name)
        and _contains_latin(alias) != _contains_latin(name)
    )


def _alias_fuzzy_similarity(alias: str, text_tokens: set[str]) -> float:
    normalized_alias = _normalize_alias_text(alias)
    if len(normalized_alias) < 5 or not text_tokens:
        return 0.0
    return max(
        SequenceMatcher(None, normalized_alias, token).ratio()
        for token in text_tokens
    )


# ---------------------------------------------------------------------------
# 4. Create inter-PhraseNode relationships (RELATED_TO)
# ---------------------------------------------------------------------------

RELATED_TO_LABEL = "RELATED_TO"


def create_phrase_relationships(
    relationships: list[Relationship],
    driver: Driver,
) -> int:
    """Create RELATED_TO edges between PhraseNodes from extracted relationships.

    Matches source/target by case-insensitive PhraseNode name.
    Returns number of edges created.
    """
    if not relationships:
        return 0

    count = 0
    with open_neo4j_session(driver) as session:
        for rel in relationships:
            src = rel.source.strip()
            tgt = rel.target.strip()
            if not src or not tgt or src.lower() == tgt.lower():
                continue

            result = session.run(
                f"""
                MATCH (a:{PHRASE_LABEL})
                WHERE toLower(a.name) = toLower($src)
                MATCH (b:{PHRASE_LABEL})
                WHERE toLower(b.name) = toLower($tgt)
                MERGE (a)-[r:{RELATED_TO_LABEL}]->(b)
                SET r.relation_type = $rel_type
                RETURN count(r) AS cnt
                """,
                src=src,
                tgt=tgt,
                rel_type=rel.relation_type,
            )
            rec = result.single()
            if rec and rec["cnt"] > 0:
                count += 1

    logger.info("Created %d RELATED_TO edges between PhraseNodes", count)
    return count


# ---------------------------------------------------------------------------
# 5. Personalized PageRank (PPR)
# ---------------------------------------------------------------------------

def compute_ppr(
    graph: nx.Graph,
    query_nodes: list[int | str],
    alpha: float | None = None,
) -> dict[int | str, float]:
    """Compute Personalized PageRank from query starting nodes.

    Args:
        graph: NetworkX graph (can be directed or undirected).
        query_nodes: Starting node IDs for personalization.
        alpha: Restart probability (default from settings).

    Returns mapping node → PPR score.
    """
    if alpha is None:
        alpha = get_settings().retrieval.ppr_alpha

    if graph.number_of_nodes() == 0 or not query_nodes:
        return {}

    # Build personalization vector: uniform over query nodes
    personalization: dict[int | str, float] = {}
    valid_query = [n for n in query_nodes if n in graph]
    if not valid_query:
        return {}

    weight = 1.0 / len(valid_query)
    for node in graph.nodes():
        personalization[node] = weight if node in valid_query else 0.0

    scores: dict[int | str, float] = nx.pagerank(
        graph,
        alpha=alpha,
        personalization=personalization,
        weight="weight",
    )

    logger.debug("PPR computed: %d nodes, %d query nodes", len(scores), len(valid_query))
    return scores


# ---------------------------------------------------------------------------
# 5. Build dual-node graph from entities + chunks
# ---------------------------------------------------------------------------

def build_dual_graph(
    entities: list[Entity],
    chunks: list[Chunk],
    driver: Driver,
    pagerank_scores: dict[str, float] | None = None,
    relationships: list[Relationship] | None = None,
) -> tuple[list[PhraseNode], list[PassageNode], int]:
    """Build complete dual-node graph in Neo4j.

    1. Create PhraseNodes from entities
    2. Create PassageNodes from chunks
    3. Link entities to passages via MENTIONED_IN
    4. Create inter-PhraseNode edges from relationships (RELATED_TO)

    Returns (phrase_nodes, passage_nodes, link_count).
    """
    phrase_nodes = create_phrase_nodes(entities, driver, pagerank_scores)
    passage_nodes = create_passage_nodes(chunks, driver)
    link_count = link_entities_to_passages(entities, chunks, driver)

    # Create inter-PhraseNode edges from extracted relationships
    canonical_relationships = _canonicalize_relationships(relationships or [], entities)
    rel_count = create_phrase_relationships(canonical_relationships, driver)

    logger.info(
        "Dual graph built: %d phrases, %d passages, %d MENTIONED_IN, %d RELATED_TO",
        len(phrase_nodes), len(passage_nodes), link_count, rel_count,
    )
    return phrase_nodes, passage_nodes, link_count


def _canonicalize_relationships(
    relationships: list[Relationship],
    entities: list[Entity],
) -> list[Relationship]:
    """Map relationship endpoints onto canonical entity names using aliases."""
    if not relationships or not entities:
        return relationships

    alias_to_name: dict[str, str] = {}
    for entity in entities:
        canonical = entity.name.strip()
        if not canonical:
            continue
        for alias in [canonical, *entity.metadata.get("aliases", [])]:
            alias_text = str(alias).strip()
            if alias_text:
                for signature in _relationship_alias_signatures(alias_text):
                    alias_to_name[signature] = canonical

    canonicalized: list[Relationship] = []
    for rel in relationships:
        src = _resolve_canonical_entity_name(rel.source, alias_to_name, entities)
        tgt = _resolve_canonical_entity_name(rel.target, alias_to_name, entities)
        canonicalized.append(
            rel.model_copy(
                update={
                    "source": src,
                    "target": tgt,
                }
            )
        )
    return canonicalized


def _normalized_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _relationship_alias_signatures(text: str) -> set[str]:
    signatures = {_normalized_name(text)}
    compact = _normalize_alias_text(text)
    if compact:
        signatures.add(compact)
    abbreviation = _abbreviation_signature(text)
    if abbreviation:
        signatures.add(abbreviation)
    return {signature for signature in signatures if signature}


def _resolve_canonical_entity_name(
    text: str,
    alias_to_name: dict[str, str],
    entities: list[Entity],
) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped

    for signature in _relationship_alias_signatures(stripped):
        canonical = alias_to_name.get(signature)
        if canonical:
            return canonical

    best_name = stripped
    best_score = 0.0
    for entity in entities:
        for alias in [entity.name, *entity.metadata.get("aliases", [])]:
            score = _surface_form_similarity(stripped, str(alias))
            if score > best_score:
                best_score = score
                best_name = entity.name
    if best_score >= _ALIAS_FUZZY_MATCH_THRESHOLD:
        return best_name
    return stripped


def _surface_form_similarity(left: str, right: str) -> float:
    left_norm = _normalize_alias_text(left)
    right_norm = _normalize_alias_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if _abbreviation_signature(left) and _abbreviation_signature(left) == right_norm:
        return 0.95
    if _abbreviation_signature(right) and _abbreviation_signature(right) == left_norm:
        return 0.95
    return SequenceMatcher(None, left_norm, right_norm).ratio()


# ---------------------------------------------------------------------------
# 6. Embed PhraseNodes and create vector index
# ---------------------------------------------------------------------------

PHRASE_INDEX_NAME = "phrase_node_index"
PASSAGE_INDEX_NAME = "passage_node_index"


def embed_phrase_nodes(
    phrase_nodes: list[PhraseNode],
    driver: Driver,
    openai_client: OpenAI | None = None,
) -> int:
    """Add embeddings to PhraseNodes by embedding their name + description.

    Returns count of nodes updated.
    """
    cfg = get_settings()
    if openai_client is None:
        from rag_core.config import make_openai_client
        openai_client = make_openai_client(cfg)

    if not phrase_nodes:
        return 0

    batch_size = max(
        1,
        getattr(cfg.ingest, "embedding_batch_size", _DEFAULT_PHRASE_EMBED_BATCH_SIZE),
    )
    with open_neo4j_session(driver) as session:
        updated = 0
        for offset in range(0, len(phrase_nodes), batch_size):
            batch = phrase_nodes[offset : offset + batch_size]
            texts = [f"{pn.name}: {pn.entity_type}" for pn in batch]
            response = openai_client.embeddings.create(
                model=cfg.openai.embedding_model,
                input=texts,
                dimensions=cfg.openai.embedding_dimensions,
            )
            for i, pn in enumerate(batch):
                emb = response.data[i].embedding
                session.run(
                    f"""
                    MATCH (p:{PHRASE_LABEL} {{id: $id}})
                    SET p.embedding = $embedding
                    """,
                    id=pn.id,
                    embedding=emb,
                )
                updated += 1

    logger.info("Added embeddings to %d PhraseNodes", updated)
    return updated


def init_phrase_index(driver: Driver) -> None:
    """Create vector index on PhraseNode embeddings if not exists."""
    cfg = get_settings()
    try:
        with open_neo4j_session(driver) as session:
            session.run(
                f"""
                CREATE VECTOR INDEX {PHRASE_INDEX_NAME} IF NOT EXISTS
                FOR (n:{PHRASE_LABEL})
                ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $dimensions,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                dimensions=cfg.openai.embedding_dimensions,
            )
        logger.info("Phrase vector index '%s' initialized", PHRASE_INDEX_NAME)
    except Exception as exc:
        logger.warning(
            "Failed to create phrase vector index '%s': %s — graph traversal will fall back to label scan",
            PHRASE_INDEX_NAME,
            exc,
        )


def init_passage_index(driver: Driver) -> None:
    """Create vector index on PassageNode embeddings if not exists."""
    cfg = get_settings()
    try:
        with open_neo4j_session(driver) as session:
            session.run(
                f"""
                CREATE VECTOR INDEX {PASSAGE_INDEX_NAME} IF NOT EXISTS
                FOR (n:{PASSAGE_LABEL})
                ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $dimensions,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                dimensions=cfg.openai.embedding_dimensions,
            )
        logger.info("Passage vector index '%s' initialized", PASSAGE_INDEX_NAME)
    except Exception as exc:
        logger.warning(
            "Failed to create passage vector index '%s': %s — "
            "temporal_query and full_document_read will fall back to label scan",
            PASSAGE_INDEX_NAME,
            exc,
        )
