"""Semantic text chunker with markdown-aware splitting.

Merged from RAG 2.0 (chunk_text → list[Chunk]) and TKB (table-aware
chunking, sanitize_for_graphiti, split_large_content for KG episodes).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from rag_core.config import get_settings
from rag_core.models import Chunk, DocumentBlock

# ── KG episode utilities (from TKB) ─────────────────────────────

MAX_EPISODE_CHARS = 8_000

_LUCENE_SPECIAL_RE = re.compile(r'[+\-&|!(){}[\]^"~*?:\\/]')
_MEDICAL_SIGNAL_TERMS = (
    "diagnosis", "diagnostic", "disease", "symptom", "sign", "treatment",
    "therapy", "drug", "medication", "dose", "adverse", "prognosis", "risk",
    "complication", "biomarker", "laboratory", "test", "screening", "imaging",
    "procedure", "syndrome", "infection", "cancer", "tumor", "mutation",
    "diagnose", "treat", "clinical", "患者", "症状", "体征", "诊断", "治疗", "药物",
    "剂量", "并发症", "预后", "风险", "检查", "检验", "影像", "手术", "感染", "肿瘤",
    "癌", "综合征", "生物标志物",
)


def sanitize_for_graphiti(text: str) -> str:
    """Remove Lucene special characters that break Neo4j fulltext queries."""
    return _LUCENE_SPECIAL_RE.sub(" ", text)


def split_large_content(
    text: str,
    source: str,
    max_chars: int = MAX_EPISODE_CHARS,
) -> list[tuple[str, str]]:
    """Split large text into episode-sized pieces for Graphiti.

    Returns list of (content, source_name) tuples.
    """
    if len(text) <= max_chars:
        return [(text, source)]

    paragraphs = re.split(r"\n\s*\n", text)
    parts: list[tuple[str, str]] = []
    current = ""
    part_num = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > max_chars:
            if current:
                parts.append((current.strip(), f"{source}_part_{part_num}"))
                part_num += 1
                current = ""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 > max_chars:
                    if current:
                        parts.append((current.strip(), f"{source}_part_{part_num}"))
                        part_num += 1
                    current = sent
                else:
                    current = f"{current} {sent}".strip() if current else sent
            continue

        if len(current) + len(para) + 2 > max_chars:
            if current:
                parts.append((current.strip(), f"{source}_part_{part_num}"))
                part_num += 1
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        parts.append((current.strip(), f"{source}_part_{part_num}"))

    return parts


# ── Main chunk_text function (from RAG 2.0) ─────────────────────

def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk text semantically using markdown structure.

    Strategy:
    1. Split by markdown headers (##, ###) first
    2. Then by paragraphs
    3. If still too large, split by sentences
    4. Tables (lines starting with |) kept as atomic units

    Each chunk gets auto-generated id (md5) and metadata.
    """
    cfg = get_settings()
    if chunk_size is None:
        chunk_size = cfg.indexing.chunk_size
    if chunk_overlap is None:
        chunk_overlap = cfg.indexing.chunk_overlap

    if not text.strip():
        return []

    chunks: list[Chunk] = []
    sections = _split_by_headers(text)

    for section_title, section_content in sections:
        section_chunks = _chunk_section(section_content, chunk_size, chunk_overlap, section_title)
        chunks.extend(section_chunks)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def chunk_document(
    document: Any,
    *,
    parent_chunk_size: int | None = None,
    child_chunk_size: int | None = None,
    child_chunk_overlap: int | None = None,
    context_chars: int | None = None,
    hierarchical: bool | None = None,
) -> list[Chunk]:
    """Chunk a loaded document, preserving structure when available."""
    cfg = get_settings()
    if parent_chunk_size is None:
        parent_chunk_size = cfg.indexing.parent_chunk_size
    if child_chunk_size is None:
        child_chunk_size = cfg.indexing.chunk_size
    if child_chunk_overlap is None:
        child_chunk_overlap = cfg.indexing.chunk_overlap
    if context_chars is None:
        context_chars = cfg.indexing.context_window_chars
    if hierarchical is None:
        hierarchical = cfg.indexing.hierarchical_chunking_enabled

    markdown = getattr(document, "markdown", "") or ""
    blocks = list(getattr(document, "blocks", []) or [])
    if not cfg.indexing.structured_chunking_enabled or not blocks:
        return chunk_text(markdown, child_chunk_size, child_chunk_overlap)
    return _chunk_structured_blocks(
        blocks,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        context_chars=context_chars,
        hierarchical=hierarchical,
    )


def chunk_document_for_graph(document: Any) -> list[Chunk]:
    """Build Graph RAG oriented chunks for relation extraction and graph linking."""
    cfg = get_settings().indexing
    markdown = getattr(document, "markdown", "") or ""
    blocks = list(getattr(document, "blocks", []) or [])
    if markdown and len(markdown) <= max(
        cfg.graph_skeleton_chunk_size,
        cfg.graph_peripheral_chunk_size * 2,
    ):
        return chunk_text(markdown, cfg.graph_skeleton_chunk_size, 0)
    if not cfg.graph_chunking_enabled or not blocks:
        return chunk_text(markdown, cfg.graph_skeleton_chunk_size, 0)

    graph_blocks = [
        block for block in blocks
        if block.block_type not in {"table", "code"}
    ]
    if not graph_blocks:
        return []

    return _chunk_graph_blocks(
        graph_blocks,
        skeleton_chunk_size=cfg.graph_skeleton_chunk_size,
        skeleton_chunk_max_size=cfg.graph_skeleton_chunk_max_size,
        peripheral_chunk_size=cfg.graph_peripheral_chunk_size,
        min_entities_per_chunk=cfg.graph_min_entities_per_chunk,
        sentence_boundary_only=cfg.graph_sentence_boundary_only,
    )


# ── Internal helpers ─────────────────────────────────────────────

def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split text by markdown headers (## or ###)."""
    header_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_content: list[str] = []

    for line in text.split("\n"):
        match = header_pattern.match(line)
        if match:
            if current_content:
                sections.append((current_title, "\n".join(current_content)))
            current_title = match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append((current_title, "\n".join(current_content)))

    if not sections:
        sections.append(("", text))

    return sections


def _chunk_structured_blocks(
    blocks: list[DocumentBlock],
    *,
    parent_chunk_size: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
    context_chars: int,
    hierarchical: bool,
) -> list[Chunk]:
    parent_groups = _assemble_parent_groups(blocks, parent_chunk_size)
    chunks: list[Chunk] = []

    for parent_index, parent in enumerate(parent_groups):
        child_segments = (
            _assemble_child_segments(
                parent["blocks"],
                child_chunk_size,
                child_chunk_overlap,
            )
            if hierarchical
            else [{
                "text": parent["text"],
                "block_types": sorted({block.block_type for block in parent["blocks"]}),
            }]
        )
        for child_index, segment in enumerate(child_segments):
            child_text = segment["text"].strip()
            if not child_text:
                continue
            chunk_id = hashlib.md5(
                f"{parent['id']}:{child_index}:{child_text}".encode("utf-8")
            ).hexdigest()[:8]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=child_text,
                    context=_build_child_context(
                        parent["text"],
                        child_text,
                        parent["heading_path"],
                        context_chars,
                    ),
                    metadata={
                        "chunking_strategy": (
                            "structured_hierarchical" if hierarchical else "structured"
                        ),
                        "parent_chunk_id": parent["id"],
                        "parent_index": parent_index,
                        "child_index_within_parent": child_index,
                        "heading_path": list(parent["heading_path"]),
                        "block_types": segment["block_types"],
                    },
                )
            )

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def _chunk_graph_blocks(
    blocks: list[DocumentBlock],
    *,
    skeleton_chunk_size: int,
    skeleton_chunk_max_size: int,
    peripheral_chunk_size: int,
    min_entities_per_chunk: int,
    sentence_boundary_only: bool,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for block in blocks:
        block_text = block.text.strip()
        if not block_text:
            continue
        heading_path = list(block.heading_path)
        entity_count = _estimate_entity_count(block_text)
        medical_score = _estimate_medical_signal_score(block_text, heading_path)
        is_core_candidate = (
            entity_count >= min_entities_per_chunk
            or medical_score >= 2
        )
        target_size = skeleton_chunk_size if is_core_candidate else peripheral_chunk_size
        max_size = skeleton_chunk_max_size if is_core_candidate else peripheral_chunk_size
        pieces = _split_for_graph_strategy(
            block_text,
            target_size=target_size,
            max_size=max_size,
            sentence_boundary_only=sentence_boundary_only,
        )
        for piece_index, piece in enumerate(pieces):
            piece_text = piece.strip()
            if not piece_text:
                continue
            piece_entities = _estimate_entity_count(piece_text)
            piece_medical_score = _estimate_medical_signal_score(piece_text, heading_path)
            chunk_id = hashlib.md5(
                f"graph:{block.order_index}:{piece_index}:{piece_text}".encode("utf-8")
            ).hexdigest()[:8]
            chunk_type = (
                "skeleton_candidate"
                if piece_entities >= min_entities_per_chunk or piece_medical_score >= 2
                else "peripheral_candidate"
            )
            metadata = {
                "section_title": heading_path[-1] if heading_path else "",
                "heading_path": heading_path,
                "block_type": block.block_type,
                "graph_chunk_type": chunk_type,
                "graph_entity_count": piece_entities,
                "graph_medical_score": piece_medical_score,
                "source_order_index": block.order_index,
                "chunk_index": chunk_index,
            }
            context = ""
            if heading_path:
                context = "Section: " + " > ".join(heading_path)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=piece_text,
                    context=context,
                    metadata=metadata,
                )
            )
            chunk_index += 1

    return chunks


def _chunk_section(
    text: str, chunk_size: int, chunk_overlap: int, section_title: str,
) -> list[Chunk]:
    """Chunk a single section into Chunk objects."""
    if not text.strip():
        return []

    lines = text.split("\n")
    is_table = all(line.strip().startswith("|") or not line.strip() for line in lines)

    if is_table and text.strip():
        return [_create_chunk(text, section_title)]

    paragraphs = text.split("\n\n")
    chunks: list[Chunk] = []
    current_chunk_text = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk_text) + len(para) + 2 <= chunk_size:
            current_chunk_text = f"{current_chunk_text}\n\n{para}".strip() if current_chunk_text else para
        else:
            if current_chunk_text:
                chunks.append(_create_chunk(current_chunk_text, section_title))
                if chunk_overlap > 0:
                    current_chunk_text = current_chunk_text[-chunk_overlap:] + "\n\n" + para
                else:
                    current_chunk_text = para
            else:
                sentence_chunks = _split_by_sentences(para, chunk_size, chunk_overlap)
                for sc in sentence_chunks:
                    chunks.append(_create_chunk(sc, section_title))
                current_chunk_text = ""

    if current_chunk_text:
        chunks.append(_create_chunk(current_chunk_text, section_title))

    return chunks


def _assemble_parent_groups(
    blocks: list[DocumentBlock],
    target_size: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: list[DocumentBlock] = []
    current_chars = 0
    current_section: tuple[str, ...] = ()
    split_threshold = max(1, target_size // 2)

    for block in blocks:
        block_text = block.text.strip()
        if not block_text:
            continue
        block_len = len(block_text)
        block_section = tuple(block.heading_path[:2] or block.heading_path)
        should_split = bool(current) and current_chars + block_len > target_size
        if (
            current
            and not should_split
            and block_section != current_section
            and current_chars >= split_threshold
        ):
            should_split = True
        if should_split:
            groups.append(_finalize_parent_group(current, len(groups)))
            current = []
            current_chars = 0
            current_section = ()
        if not current:
            current_section = block_section
        current.append(block)
        current_chars += block_len + 2

    if current:
        groups.append(_finalize_parent_group(current, len(groups)))
    return groups


def _split_for_graph_strategy(
    text: str,
    *,
    target_size: int,
    max_size: int,
    sentence_boundary_only: bool,
) -> list[str]:
    if len(text) <= target_size:
        return [text]

    if not sentence_boundary_only:
        return _split_long_text(text, target_size, 0)

    sentences = [sentence.strip() for sentence in _split_by_sentences(text, max_size, 0) if sentence.strip()]
    if len(sentences) <= 1:
        return [text[:max_size].strip()] if len(text) > max_size else [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        projected = current_len + sentence_len + (1 if current else 0)
        if current and projected > target_size:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = sentence_len
            continue
        if not current and sentence_len > max_size:
            chunks.append(sentence[:max_size].strip())
            current = []
            current_len = 0
            continue
        current.append(sentence)
        current_len = projected

    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _estimate_entity_count(text: str) -> int:
    candidates = set()
    for token in re.findall(r"\b[A-Z][A-Za-z0-9_-]{1,}\b", text):
        if len(token) >= 2:
            candidates.add(token)
    return len(candidates)


def _estimate_medical_signal_score(text: str, heading_path: list[str] | None = None) -> int:
    haystack = f"{' '.join(heading_path or [])}\n{text}".casefold()
    score = 0
    for term in _MEDICAL_SIGNAL_TERMS:
        if term.casefold() in haystack:
            score += 1
    return score


def estimate_chunk_entity_count(chunk: Chunk) -> int:
    """Estimate entity count for a chunk using metadata when available."""
    metadata_count = chunk.metadata.get("graph_entity_count")
    if isinstance(metadata_count, int) and metadata_count >= 0:
        return metadata_count
    return _estimate_entity_count(chunk.enriched_content)


def _finalize_parent_group(blocks: list[DocumentBlock], index: int) -> dict[str, Any]:
    heading_path = _common_heading_path([block.heading_path for block in blocks])
    parts: list[str] = []
    if heading_path:
        parts.append("Section: " + " > ".join(heading_path))
    parts.extend(block.text.strip() for block in blocks if block.text.strip())
    text = "\n\n".join(parts).strip()
    parent_id = hashlib.md5(f"parent:{index}:{text}".encode("utf-8")).hexdigest()[:8]
    return {
        "id": parent_id,
        "text": text,
        "heading_path": heading_path,
        "blocks": blocks,
    }


def _common_heading_path(paths: list[list[str]]) -> list[str]:
    if not paths:
        return []
    common = list(paths[0])
    for path in paths[1:]:
        limit = min(len(common), len(path))
        cursor = 0
        while cursor < limit and common[cursor] == path[cursor]:
            cursor += 1
        common = common[:cursor]
        if not common:
            break
    return common


def _assemble_child_segments(
    blocks: list[DocumentBlock],
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current_parts: list[str] = []
    current_types: set[str] = set()
    current_chars = 0

    def flush_current() -> None:
        nonlocal current_parts, current_types, current_chars
        if not current_parts:
            return
        segments.append({
            "text": "\n\n".join(current_parts).strip(),
            "block_types": sorted(current_types),
        })
        current_parts = []
        current_types = set()
        current_chars = 0

    for block in blocks:
        block_text = block.text.strip()
        if not block_text:
            continue
        pieces = _split_block_for_children(block, child_chunk_size, child_chunk_overlap)
        for piece in pieces:
            piece_text = piece.strip()
            if not piece_text:
                continue
            piece_len = len(piece_text)
            is_atomic = block.block_type in {"table", "code"}
            if is_atomic:
                flush_current()
                segments.append({"text": piece_text, "block_types": [block.block_type]})
                continue
            if current_parts and current_chars + piece_len > child_chunk_size:
                flush_current()
            current_parts.append(piece_text)
            current_types.add(block.block_type)
            current_chars += piece_len + 2

    flush_current()
    return segments


def _split_block_for_children(
    block: DocumentBlock,
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> list[str]:
    text = block.text.strip()
    if len(text) <= child_chunk_size or block.block_type in {"table", "code"}:
        return [text]
    if block.block_type in {"paragraph", "list", "list_item"}:
        return _split_text_with_overlap(text, child_chunk_size, child_chunk_overlap)
    return [text]


def _split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    sentence_chunks = _split_by_sentences(text, chunk_size, chunk_overlap)
    if len(sentence_chunks) == 1 and len(sentence_chunks[0]) > chunk_size:
        return _split_long_text(sentence_chunks[0], chunk_size, chunk_overlap)
    normalized = [chunk.strip() for chunk in sentence_chunks if chunk.strip()]
    return normalized or _split_long_text(text, chunk_size, chunk_overlap)


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - max(0, chunk_overlap))
    return [text[start:start + chunk_size].strip() for start in range(0, len(text), step)]


def _build_child_context(
    parent_text: str,
    child_text: str,
    heading_path: list[str],
    context_chars: int,
) -> str:
    parts: list[str] = []
    if heading_path:
        parts.append("Section: " + " > ".join(heading_path))
    parent_context = _compact_parent_context(parent_text, child_text, context_chars)
    if parent_context:
        parts.append(parent_context)
    return "\n\n".join(parts).strip()


def _compact_parent_context(parent_text: str, child_text: str, max_chars: int) -> str:
    candidate = parent_text.strip()
    if child_text and child_text in candidate:
        candidate = candidate.replace(child_text, "", 1).strip()
    if not candidate or max_chars <= 0:
        return ""
    if len(candidate) <= max_chars:
        return candidate
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars - 5
    return f"{candidate[:head_chars].strip()}\n...\n{candidate[-tail_chars:].strip()}"


def _split_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text by sentence boundaries when paragraph is too large."""
    sentence_pattern = re.compile(r"([.!?]+\s+)")
    parts = sentence_pattern.split(text)

    sentences: list[str] = []
    current = ""
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)

    chunks: list[str] = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
                if chunk_overlap > 0:
                    current_chunk = current_chunk[-chunk_overlap:] + sent
                else:
                    current_chunk = sent
            else:
                chunks.append(sent)
                current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _create_chunk(content: str, section_title: str) -> Chunk:
    """Create Chunk with auto-generated id and metadata."""
    chunk_id = hashlib.md5(content.encode()).hexdigest()[:8]
    return Chunk(
        id=chunk_id,
        content=content,
        metadata={"section_title": section_title} if section_title else {},
    )
