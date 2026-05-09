"""KET-RAG Skeleton Indexer with adaptive sampling and hybrid extraction."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from rag_core.chunker import estimate_chunk_entity_count
from rag_core.config import get_settings
from rag_core.llm_resilience import LLMFatalError
from rag_core.models import Chunk, Entity, Relationship

from agentic_graph_rag.indexing.dual_node import (
    _chunk_entity_match_score,
    _entity_surface_forms,
    _normalize_alias_text as dual_normalize_alias_text,
    persist_entity_alias_metadata,
)
from agentic_graph_rag.text_signals import build_tfidf_profile, text_signal_score

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)
_DEFAULT_EXTRACTION_BATCH_SIZE = 1
_MIN_KNN_SIMILARITY = 0.7
_MAX_EXTRACTION_TEXT_CHARS = 1200
_MAX_CANDIDATE_ENTITIES = 6

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "and", "or", "but",
    "if", "then", "than", "that", "this", "it", "its", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "into", "about",
    "not", "no", "so", "up", "out", "all", "also", "very", "just", "how",
})

_PAPER_HINTS = (
    "abstract", "introduction", "methodology", "experiment", "conclusion",
    "references", "appendix", "theorem", "proof",
)
_TECHNICAL_HINTS = (
    "architecture", "service", "module", "api", "endpoint", "config",
    "deployment", "pipeline", "class", "function", "component",
)
_MEDICAL_HINTS = (
    "diagnosis", "diagnostic", "disease", "symptom", "sign", "treatment",
    "therapy", "drug", "medication", "dose", "adverse", "prognosis", "risk",
    "complication", "biomarker", "laboratory", "test", "screening", "imaging",
    "procedure", "syndrome", "infection", "cancer", "tumor", "mutation",
    "patient", "clinical", "患者", "症状", "体征", "诊断", "治疗", "药物",
    "剂量", "并发症", "预后", "风险", "检查", "检验", "影像", "手术", "感染",
    "肿瘤", "癌", "综合征", "生物标志物",
)
_MEDICAL_ENTITY_TYPES = (
    "Disease", "Symptom", "Drug", "Test", "Biomarker", "Anatomy",
    "Procedure", "RiskFactor", "Pathogen", "Population",
)
_ALIAS_FUZZY_MATCH_THRESHOLD = 0.88
_ALIAS_PROMOTION_MIN_COUNT = 2


def _normalized_name(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalized_alias_signature(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"[\s\-_/.]+", "", text)
    return re.sub(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", "", text)


def _abbreviation_signature(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    if len(words) < 2:
        return ""
    return "".join(word[0] for word in words).casefold()


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))


def _contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def _is_medical_abbreviation_alias(candidate: str, name: str) -> bool:
    candidate_text = candidate.strip()
    if not re.fullmatch(r"[A-Za-z0-9]{2,10}", candidate_text):
        return False
    name_abbr = _abbreviation_signature(name)
    return bool(name_abbr and candidate_text.casefold() == name_abbr)


def _is_cross_language_alias(candidate: str, name: str) -> bool:
    return (
        _contains_cjk(candidate) != _contains_cjk(name)
        and _contains_latin(candidate) != _contains_latin(name)
    )


def _candidate_matches_entity_name(candidate: str, name: str) -> bool:
    if _is_cross_language_alias(candidate, name) and not _is_medical_abbreviation_alias(candidate, name):
        return False
    candidate_norm = _normalized_alias_signature(candidate)
    name_norm = _normalized_alias_signature(name)
    if not candidate_norm or not name_norm:
        return False
    if candidate_norm == name_norm:
        return True
    if candidate_norm in name_norm or name_norm in candidate_norm:
        return True
    candidate_abbr = _abbreviation_signature(candidate)
    name_abbr = _abbreviation_signature(name)
    if candidate_abbr and candidate_abbr == name_norm:
        return True
    if name_abbr and name_abbr == candidate_norm:
        return True
    return SequenceMatcher(None, candidate_norm, name_norm).ratio() >= _ALIAS_FUZZY_MATCH_THRESHOLD


def _resolve_aliases_for_entity(name: str, candidates: list[str]) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_text = str(candidate).strip()
        if not candidate_text:
            continue
        if not _candidate_matches_entity_name(candidate_text, name):
            continue
        seen_key = candidate_text.casefold()
        if seen_key in seen:
            continue
        seen.add(seen_key)
        aliases.append(candidate_text)
    return aliases


def _float_setting(settings: object, name: str, default: float) -> float:
    value = getattr(settings, name, default)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _int_setting(settings: object, name: str, default: int) -> int:
    value = getattr(settings, name, default)
    if isinstance(value, int):
        return value
    return default


@lru_cache(maxsize=1)
def _load_spacy_model():
    try:
        import spacy
    except ImportError:
        return None

    for model_name in ("en_core_web_sm", "xx_ent_wiki_sm"):
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    return None


def _iter_chunk_hints(chunks: list[Chunk]) -> str:
    parts: list[str] = []
    for chunk in chunks[:8]:
        section = str(chunk.metadata.get("section_title", "")).strip()
        source = str(chunk.metadata.get("source", "")).strip()
        doc_type = str(chunk.metadata.get("document_type", "")).strip()
        if section:
            parts.append(section.lower())
        if source:
            parts.append(source.lower())
        if doc_type:
            parts.append(doc_type.lower())
    return " ".join(parts)


def infer_document_type(chunks: list[Chunk]) -> str:
    """Infer a coarse document type for adaptive PageRank tuning."""
    hints = _iter_chunk_hints(chunks)
    paper_hits = sum(1 for word in _PAPER_HINTS if word in hints)
    technical_hits = sum(1 for word in _TECHNICAL_HINTS if word in hints)
    medical_hits = sum(1 for word in _MEDICAL_HINTS if word in hints)

    if medical_hits >= max(paper_hits, technical_hits) and medical_hits > 0:
        return "medical"
    if paper_hits > technical_hits and paper_hits > 0:
        return "paper"
    if technical_hits > 0:
        return "technical"
    return "generic"


def resolve_skeleton_beta(chunks: list[Chunk], beta: float | None = None) -> float:
    """Choose skeleton beta dynamically based on document length."""
    if beta is not None:
        return beta

    cfg = get_settings().indexing
    default_beta = _float_setting(cfg, "skeleton_beta", 0.25)
    short_beta = _float_setting(cfg, "skeleton_beta_short_doc", default_beta)
    medium_beta = _float_setting(cfg, "skeleton_beta_medium_doc", default_beta)
    long_beta = _float_setting(cfg, "skeleton_beta_long_doc", default_beta)
    short_max = _int_setting(cfg, "skeleton_short_doc_max_chunks", 8)
    medium_max = _int_setting(cfg, "skeleton_medium_doc_max_chunks", 24)

    chunk_count = len(chunks)
    if chunk_count <= max(1, short_max):
        # Short documents do not benefit from full-width skeletal extraction.
        return min(short_beta, 0.5)
    if chunk_count <= max(1, medium_max):
        return medium_beta
    return long_beta


def resolve_pagerank_damping(chunks: list[Chunk], damping: float | None = None) -> float:
    """Choose PageRank damping dynamically from inferred document type."""
    if damping is not None:
        return damping

    cfg = get_settings().indexing
    default_damping = _float_setting(cfg, "pagerank_damping", 0.85)
    doc_type = infer_document_type(chunks)
    if doc_type == "paper":
        return _float_setting(cfg, "pagerank_damping_paper", default_damping)
    if doc_type == "technical":
        return _float_setting(cfg, "pagerank_damping_technical", default_damping)
    return default_damping


def extract_candidate_entities(text: str, max_candidates: int = 12) -> list[str]:
    """Extract cheap entity candidates via optional spaCy, else heuristics."""
    seen: dict[str, str] = {}

    model = _load_spacy_model()
    if model is not None:
        try:
            doc = model(text[:4000])
            for ent in doc.ents:
                candidate = ent.text.strip()
                norm = _normalized_name(candidate)
                if len(candidate) >= 3 and norm not in seen:
                    seen[norm] = candidate
        except Exception as exc:
            logger.debug("spaCy candidate extraction failed: %s", exc)

    patterns = (
        r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b",
        r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b",
        r"\b(?:[A-Za-z]+[A-Z][A-Za-z0-9_-]+|[A-Z][a-z]+RAG|Neo4j|PageRank|GraphRAG)\b",
        r"\b(?:[A-Za-z]+(?:itis|emia|oma|osis|pathy|plasia|penia|cytosis|genic))\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = match.group(0).strip(" ,.;:()[]{}")
            norm = _normalized_name(candidate)
            if len(candidate) < 3 or norm in _STOP_WORDS or norm in seen:
                continue
            seen[norm] = candidate

    candidates = list(seen.values())
    candidates.sort(key=lambda value: (-len(value.split()), value.lower()))
    return candidates[:max_candidates]


# ---------------------------------------------------------------------------
# 1. KNN graph construction
# ---------------------------------------------------------------------------

def build_knn_graph(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    k: int | None = None,
) -> nx.DiGraph:
    """Build a directed KNN graph over chunks using cosine similarity."""
    if k is None:
        k = get_settings().indexing.knn_k

    n = len(chunks)
    if n == 0:
        return nx.DiGraph()

    emb_matrix = np.array(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = emb_matrix / norms
    sim_matrix = normed @ normed.T

    graph = nx.DiGraph()
    for i in range(n):
        graph.add_node(i, chunk_id=chunks[i].id)

    effective_k = min(k, n - 1)
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1.0
        if effective_k > 0:
            top_indices = np.argsort(sims)[-effective_k:][::-1]
            for j_idx in top_indices:
                j = int(j_idx)
                similarity = float(sims[j])
                if similarity <= _MIN_KNN_SIMILARITY:
                    continue
                graph.add_edge(i, j, weight=similarity)

    logger.info("Built KNN graph: %d nodes, %d edges (k=%d)", n, graph.number_of_edges(), effective_k)
    return graph


# ---------------------------------------------------------------------------
# 2. PageRank computation
# ---------------------------------------------------------------------------

def compute_pagerank(
    knn_graph: nx.DiGraph,
    damping: float | None = None,
) -> dict[int, float]:
    """Compute PageRank scores for chunk nodes."""
    if damping is None:
        damping = get_settings().indexing.pagerank_damping

    if knn_graph.number_of_nodes() == 0:
        return {}

    scores: dict[int, float] = nx.pagerank(knn_graph, alpha=damping, weight="weight")
    logger.debug("PageRank computed for %d nodes", len(scores))
    return scores


# ---------------------------------------------------------------------------
# 3. Skeletal chunk selection
# ---------------------------------------------------------------------------

def select_skeletal_chunks(
    chunks: list[Chunk],
    pagerank_scores: dict[int, float],
    beta: float | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Split chunks into skeletal and peripheral subsets."""
    beta = resolve_skeleton_beta(chunks, beta)

    if not chunks or not pagerank_scores:
        return [], list(chunks)

    ranked_scores = _rank_chunks_for_skeleton_selection(chunks, pagerank_scores)
    ranked = sorted(ranked_scores.items(), key=lambda kv: kv[1], reverse=True)
    n_skeletal = max(1, int(len(chunks) * beta))
    if len(chunks) <= 10:
        n_skeletal = min(n_skeletal, max(2, (len(chunks) + 1) // 2))
    skeletal_indices = {idx for idx, _ in ranked[:n_skeletal]}

    skeletal: list[Chunk] = []
    peripheral: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        if i in skeletal_indices:
            skeletal.append(chunk)
        else:
            peripheral.append(chunk)

    logger.info(
        "Selected %d skeletal + %d peripheral chunks (beta=%.2f)",
        len(skeletal), len(peripheral), beta,
    )
    return skeletal, peripheral


def filter_low_information_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> tuple[list[Chunk], list[list[float]], list[Chunk]]:
    """Drop low-information chunks before KNN + PageRank graph construction."""
    if not chunks or not embeddings or len(chunks) != len(embeddings):
        return chunks, embeddings, []

    cfg = get_settings().indexing
    profile = build_tfidf_profile([chunk.enriched_content for chunk in chunks])
    min_idf = _float_setting(cfg, "tfidf_low_idf_threshold", 1.2)
    score_threshold = _float_setting(cfg, "tfidf_low_info_chunk_score_threshold", 0.6)
    max_keywords = _int_setting(cfg, "tfidf_max_keywords", 8)
    normalized_text_counts: dict[str, int] = {}
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk.enriched_content.strip())
        if normalized:
            normalized_text_counts[normalized] = normalized_text_counts.get(normalized, 0) + 1

    kept_chunks: list[Chunk] = []
    kept_embeddings: list[list[float]] = []
    dropped_chunks: list[Chunk] = []

    for chunk, embedding in zip(chunks, embeddings, strict=False):
        normalized = re.sub(r"\s+", " ", chunk.enriched_content.strip())
        if (
            normalized
            and len(normalized) <= 12
            and normalized_text_counts.get(normalized, 0) >= 2
        ):
            chunk.metadata["tfidf_signal_score"] = 0.0
            chunk.metadata["low_information_chunk"] = True
            dropped_chunks.append(chunk)
            continue
        score = text_signal_score(
            chunk.enriched_content,
            profile,
            min_idf=min_idf,
            max_keywords=max_keywords,
        )
        if score < score_threshold:
            chunk.metadata["tfidf_signal_score"] = score
            chunk.metadata["low_information_chunk"] = True
            dropped_chunks.append(chunk)
            continue

        chunk.metadata["tfidf_signal_score"] = score
        chunk.metadata["low_information_chunk"] = False
        kept_chunks.append(chunk)
        kept_embeddings.append(embedding)

    if not kept_chunks:
        return chunks, embeddings, []

    logger.info(
        "Filtered %d low-information chunks before skeleton graph build",
        len(dropped_chunks),
    )
    return kept_chunks, kept_embeddings, dropped_chunks


def _rank_chunks_for_skeleton_selection(
    chunks: list[Chunk],
    pagerank_scores: dict[int, float],
) -> dict[int, float]:
    """Blend graph centrality with entity density for skeleton selection."""
    cfg = get_settings().indexing
    density_weight = _float_setting(cfg, "skeleton_entity_density_weight", 0.35)
    density_weight = min(max(density_weight, 0.0), 1.0)
    if not chunks:
        return {}

    entity_counts = {
        index: estimate_chunk_entity_count(chunk)
        for index, chunk in enumerate(chunks)
    }
    max_pagerank = max(pagerank_scores.values(), default=0.0) or 1.0
    max_entity_count = max(entity_counts.values(), default=0) or 1
    ranked_scores: dict[int, float] = {}

    for index, chunk in enumerate(chunks):
        pagerank_score = pagerank_scores.get(index, 0.0) / max_pagerank
        entity_density = entity_counts.get(index, 0) / max_entity_count
        graph_chunk_type = str(chunk.metadata.get("graph_chunk_type", ""))
        type_boost = 1.0 if graph_chunk_type == "skeleton_candidate" else 0.0
        density_signal = max(entity_density, type_boost)
        ranked_scores[index] = ((1.0 - density_weight) * pagerank_score) + (
            density_weight * density_signal
        )
    return ranked_scores


# ---------------------------------------------------------------------------
# 4. Hybrid entity extraction (candidate NER + LLM validation)
# ---------------------------------------------------------------------------

def extract_entities_full(
    skeletal_chunks: list[Chunk],
    openai_client: OpenAI | None = None,
) -> tuple[list[Entity], list[Relationship]]:
    """Extract entities and relationships from skeletal chunks using candidate NER + LLM."""
    if not skeletal_chunks:
        return [], []

    cfg = get_settings()
    if openai_client is None:
        from rag_core.config import make_openai_client
        openai_client = make_openai_client(cfg)

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    system_prompt = (
        "You are validating and completing entity extraction for Graph RAG.\n"
        "A lightweight NER step already produced candidate entities.\n"
        "Use those candidates as hints, remove noise, merge aliases, add important missing entities,\n"
        "and extract high-value relationships grounded in the text.\n"
        "Return valid JSON only. Do not wrap it in markdown. If nothing is found, return empty arrays.\n"
        "The JSON shape must be exactly:\n"
        "{\n"
        '  "entities": [{"chunk_id": "...", "name": "...", "type": "...", "confidence": 0.9}],\n'
        '  "relationships": [{"chunk_id": "...", "from": "...", "to": "...", "type": "...", "confidence": 0.8}]\n'
        "}\n"
        "Use only these entity types when possible: "
        + ", ".join(_MEDICAL_ENTITY_TYPES)
        + ".\n"
        "Keep entity names concise and canonical. Keep only medically meaningful relations.\n"
        "Entity confidence and relationship confidence must be numbers between 0 and 1.\n"
    )

    extraction_batch_size = _DEFAULT_EXTRACTION_BATCH_SIZE
    for offset in range(0, len(skeletal_chunks), extraction_batch_size):
        batch = skeletal_chunks[offset : offset + extraction_batch_size]
        batch_candidates: dict[str, list[str]] = {}
        prompt_blocks: list[str] = []
        for chunk in batch:
            candidates = extract_candidate_entities(
                chunk.enriched_content,
                max_candidates=_MAX_CANDIDATE_ENTITIES,
            )
            batch_candidates[chunk.id] = candidates
            prompt_blocks.append(
                "\n".join(
                    [
                        f"Chunk ID: {chunk.id}",
                        f"Candidate entities: {', '.join(candidates) or 'none'}",
                        "Text:",
                        chunk.enriched_content[:_MAX_EXTRACTION_TEXT_CHARS],
                    ]
                )
            )
        user_prompt = (
            "Extract entities and relationships for each chunk independently.\n"
            "Preserve the chunk_id for every entity and relationship item.\n\n"
            + "\n\n---\n\n".join(prompt_blocks)
        )
        try:
            response = openai_client.chat.completions.create(
                model=cfg.openai.llm_model_mini,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            text = response.choices[0].message.content or ""
            entities, rels = _parse_extraction_response(
                text,
                candidate_entities_by_chunk=batch_candidates,
            )
            all_entities.extend(entities)
            all_relationships.extend(rels)
        except LLMFatalError:
            raise
        except Exception as exc:
            chunk_ids = ",".join(chunk.id for chunk in batch)
            logger.error("Entity extraction failed for chunks [%s]: %s", chunk_ids, exc)

    entities = _merge_entities(all_entities)
    logger.info(
        "Extracted %d entities, %d relationships from %d skeletal chunks",
        len(entities), len(all_relationships), len(skeletal_chunks),
    )
    return entities, all_relationships


def _parse_extraction_response(
    text: str,
    source_chunk_id: str | None = None,
    candidate_entities: list[str] | None = None,
    candidate_entities_by_chunk: dict[str, list[str]] | None = None,
) -> tuple[list[Entity], list[Relationship]]:
    """Parse LLM extraction output into Entity and Relationship objects."""
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    candidates = candidate_entities or []
    candidate_map = candidate_entities_by_chunk or {}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        for item in payload.get("entities", []):
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or source_chunk_id or "").strip()
            name = str(item.get("name") or "").strip()
            entity_type = str(item.get("type") or "").strip()
            if not name or not entity_type:
                continue
            ent_candidates = candidate_map.get(chunk_id, candidates)
            ent_id = hashlib.md5(name.lower().encode()).hexdigest()[:8]
            aliases = _resolve_aliases_for_entity(name, ent_candidates)
            entities.append(Entity(
                id=ent_id,
                name=name,
                entity_type=entity_type,
                description=str(item.get("description") or "").strip(),
                entity_confidence=float(item.get("confidence") or 0.0),
                metadata={
                    "source_chunk": chunk_id,
                    "aliases": aliases,
                    "candidate_entities": ent_candidates,
                    "confidence": item.get("confidence"),
                },
            ))
        for item in payload.get("relationships", []):
            if not isinstance(item, dict):
                continue
            src = str(item.get("from") or "").strip()
            tgt = str(item.get("to") or "").strip()
            rel_type = str(item.get("type") or "").strip()
            if not src or not tgt or not rel_type:
                continue
            chunk_id = str(item.get("chunk_id") or source_chunk_id or "").strip()
            rel_id = hashlib.md5(
                f"{src}:{rel_type}:{tgt}".lower().encode()
            ).hexdigest()[:8]
            relationships.append(Relationship(
                id=rel_id,
                source=src,
                target=tgt,
                relation_type=rel_type,
                description=str(item.get("description") or "").strip(),
                metadata={
                    "source_chunk": chunk_id,
                    "confidence": item.get("confidence"),
                },
            ))
        return entities, relationships

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("ENTITY:"):
            parts = [p.strip() for p in line[len("ENTITY:"):].split("|")]
            if len(parts) >= 2:
                name = parts[0]
                ent_id = hashlib.md5(name.lower().encode()).hexdigest()[:8]
                aliases = _resolve_aliases_for_entity(name, candidates)
                entities.append(Entity(
                    id=ent_id,
                    name=name,
                    entity_type=parts[1] if len(parts) > 1 else "",
                    description=parts[2] if len(parts) > 2 else "",
                    entity_confidence=0.0,
                    metadata={
                        "source_chunk": source_chunk_id,
                        "aliases": aliases,
                        "candidate_entities": candidates,
                    },
                ))
        elif line.startswith("RELATIONSHIP:"):
            parts = [p.strip() for p in line[len("RELATIONSHIP:"):].split("|")]
            if len(parts) >= 3:
                rel_id = hashlib.md5(
                    f"{parts[0]}:{parts[1]}:{parts[2]}".lower().encode()
                ).hexdigest()[:8]
                relationships.append(Relationship(
                    id=rel_id,
                    source=parts[0],
                    target=parts[2],
                    relation_type=parts[1],
                    metadata={"source_chunk": source_chunk_id},
                ))

    return entities, relationships


def _merge_entities(entities: list[Entity]) -> list[Entity]:
    """Merge duplicate entities and accumulate aliases / provenance."""
    merged: list[Entity] = []
    for entity in entities:
        norm = f"{_normalized_name(entity.name)}::{entity.entity_type.strip().casefold()}"
        target: Entity | None = None
        for candidate in merged:
            candidate_norm = (
                f"{_normalized_name(candidate.name)}::"
                f"{candidate.entity_type.strip().casefold()}"
            )
            if candidate_norm == norm:
                target = candidate
                break
        if target is None:
            target = entity.model_copy(deep=True)
            target.metadata.setdefault("aliases", [])
            target.metadata.setdefault("source_chunks", [])
            target.metadata["confidence"] = target.entity_confidence
            merged.append(target)
            continue

        if entity.description and not target.description:
            target.description = entity.description
        if entity.entity_type and not target.entity_type:
            target.entity_type = entity.entity_type

        aliases = set(target.metadata.get("aliases", []))
        aliases.add(entity.name)
        aliases.update(entity.metadata.get("aliases", []))
        target.metadata["aliases"] = sorted(alias for alias in aliases if alias)

        source_chunks = set(target.metadata.get("source_chunks", []))
        source_chunk = entity.metadata.get("source_chunk")
        if source_chunk:
            source_chunks.add(source_chunk)
        target.metadata["source_chunks"] = sorted(source_chunks)

        if entity.entity_confidence > target.entity_confidence:
            target.entity_confidence = entity.entity_confidence
            target.metadata["confidence"] = entity.entity_confidence
    return merged


# ---------------------------------------------------------------------------
# 5. Keyword-based peripheral linking (cheap, no LLM)
# ---------------------------------------------------------------------------

def link_peripheral_keywords(
    peripheral_chunks: list[Chunk],
    existing_entities: list[Entity],
) -> list[Relationship]:
    """Link peripheral chunks to existing entities via keyword matching."""
    if not peripheral_chunks or not existing_entities:
        return []

    relationships: list[Relationship] = []
    alias_observations: dict[str, dict[str, set[str]]] = {
        entity.id or entity.name: {} for entity in existing_entities
    }

    for chunk in peripheral_chunks:
        for entity in existing_entities:
            if not _is_medically_salient_entity(entity):
                continue
            score = _chunk_entity_match_score(
                chunk.enriched_content,
                _entity_surface_forms(entity),
            )
            if score <= 0:
                continue

            rel_id = hashlib.md5(
                f"{entity.id}:mentioned_in:{chunk.id}".encode()
            ).hexdigest()[:8]
            relationships.append(Relationship(
                id=rel_id,
                source=entity.name,
                target=chunk.id,
                relation_type="MENTIONED_IN",
                metadata={"method": "keyword", "score": score},
            ))
            _collect_peripheral_alias_observations(
                entity,
                chunk,
                score,
                alias_observations,
            )

    _promote_observed_aliases(existing_entities, alias_observations)

    logger.info(
        "Linked %d peripheral mentions across %d chunks",
        len(relationships), len(peripheral_chunks),
    )
    return relationships


def _collect_peripheral_alias_observations(
    entity: Entity,
    chunk: Chunk,
    score: float,
    alias_observations: dict[str, dict[str, set[str]]],
) -> None:
    if score < 0.95:
        return

    entity_key = entity.id or entity.name
    seen_aliases = {
        dual_normalize_alias_text(alias)
        for alias in [entity.name, *entity.metadata.get("aliases", [])]
        if str(alias).strip()
    }

    candidates = set(entity.metadata.get("candidate_entities", []))
    candidates.update(extract_candidate_entities(chunk.enriched_content, max_candidates=16))
    for candidate in candidates:
        candidate_text = str(candidate).strip()
        if len(candidate_text) < 2:
            continue
        if not _candidate_matches_entity_name(candidate_text, entity.name):
            continue
        normalized = dual_normalize_alias_text(candidate_text)
        if not normalized or normalized in seen_aliases:
            continue
        alias_observations.setdefault(entity_key, {}).setdefault(candidate_text, set()).add(chunk.id)


def _promote_observed_aliases(
    entities: list[Entity],
    alias_observations: dict[str, dict[str, set[str]]],
) -> None:
    min_count = max(1, _ALIAS_PROMOTION_MIN_COUNT)
    for entity in entities:
        entity_key = entity.id or entity.name
        observations = alias_observations.get(entity_key, {})
        if not observations:
            continue

        aliases = list(entity.metadata.get("aliases", []))
        alias_signatures = {dual_normalize_alias_text(alias) for alias in aliases}
        promoted_aliases: list[str] = []
        for alias_text, chunk_ids in observations.items():
            if len(chunk_ids) < min_count:
                continue
            signature = dual_normalize_alias_text(alias_text)
            if not signature or signature in alias_signatures:
                continue
            aliases.append(alias_text)
            promoted_aliases.append(alias_text)
            alias_signatures.add(signature)

        if promoted_aliases:
            entity.metadata["aliases"] = aliases


def _is_medically_salient_entity(entity: Entity) -> bool:
    entity_type = entity.entity_type.strip().casefold()
    if entity_type and entity_type in {item.casefold() for item in _MEDICAL_ENTITY_TYPES}:
        return True
    name = entity.name.casefold()
    return any(hint.casefold() in name for hint in _MEDICAL_HINTS)


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract keywords from text (simple tokenizer, no LLM)."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS]
    freq: dict[str, int] = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return [word for word, _ in ranked[:max_keywords]]


# ---------------------------------------------------------------------------
# 6. Orchestrator
# ---------------------------------------------------------------------------

def build_skeleton_index(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    openai_client: OpenAI | None = None,
    driver=None,
) -> tuple[list[Entity], list[Relationship], list[Chunk], list[Chunk]]:
    """Full KET-RAG skeleton indexing pipeline."""
    if not chunks or not embeddings:
        return [], [], [], []

    doc_type = infer_document_type(chunks)
    damping = resolve_pagerank_damping(chunks)
    beta = resolve_skeleton_beta(chunks)

    graph_chunks, graph_embeddings, low_information_chunks = filter_low_information_chunks(
        chunks,
        embeddings,
    )

    knn_graph = build_knn_graph(graph_chunks, graph_embeddings)
    pagerank_scores = compute_pagerank(knn_graph, damping=damping)
    skeletal, peripheral = select_skeletal_chunks(graph_chunks, pagerank_scores, beta=beta)
    peripheral.extend(low_information_chunks)
    entities, relationships = extract_entities_full(skeletal, openai_client)
    relationships.extend(link_peripheral_keywords(peripheral, entities))
    if driver is not None:
        persist_entity_alias_metadata(entities, driver)

    logger.info(
        "Skeleton index built: %d entities, %d relationships "
        "(%d skeletal + %d peripheral, doc_type=%s, beta=%.2f, damping=%.2f)",
        len(entities), len(relationships), len(skeletal), len(peripheral), doc_type, beta, damping,
    )
    return entities, relationships, skeletal, peripheral
