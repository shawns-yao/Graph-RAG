"""LLM answer generation from retrieved chunks.

From RAG 2.0 — generates answers using OpenAI chat completions
with context from retrieved search results.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from rag_core.config import get_settings, make_openai_client
from rag_core.models import QAResult, SearchResult

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_ENUM_RE = None
_GRAPH_SECTION_SPLIT_RE = re.compile(r"\n\s*\n(?=(?:Graph paths:|Entities:|Evidence:))")


def _retrieval_number(cfg: object, field: str, default: float) -> float:
    retrieval_cfg = getattr(cfg, "retrieval", None)
    value = getattr(retrieval_cfg, field, default)
    return float(value) if isinstance(value, int | float) else default


def _is_enumeration_query(query: str) -> bool:
    """Detect enumeration/global queries that need comprehensive listing."""
    global _ENUM_RE  # noqa: PLW0603
    if _ENUM_RE is None:
        import re
        _ENUM_RE = re.compile(
            r'\b('
            r'все\b|всех\b|всё\b|перечисл|опиши все|резюмируй все|обзор\b'
            r'|list all|describe all|summarize all|overview|every\b'
            r'|все компоненты|все методы|все слои|все решения|семь\b|seven\b'
            r'|all components|all layers|all methods|all decisions'
            r')\b',
            re.IGNORECASE,
        )
    return bool(_ENUM_RE.search(query))


def _order_results_for_prompt(results: list[SearchResult]) -> list[SearchResult]:
    """Place the strongest evidence at the prompt edges to reduce mid-context loss."""
    ranked = sorted(
        enumerate(results),
        key=lambda item: (-item[1].score, item[0]),
    )
    ordered = [result for _, result in ranked]
    if len(ordered) <= 2:
        return ordered

    head = ordered[::2]
    tail = list(reversed(ordered[1::2]))
    return head + tail


def _limit_results_for_generation(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Apply hard prompt bounds before context assembly."""
    cfg = get_settings()
    max_chunks = max(1, cfg.retrieval.prompt_max_chunks)
    max_chars = max(1, cfg.retrieval.prompt_max_chars)
    normalized_query = " ".join(query.strip().lower().split())

    if _is_enumeration_query(query):
        max_chunks = min(max_chunks, 5)
        max_chars = min(max_chars, 8_000)
    elif len(normalized_query) <= 64:
        max_chunks = min(max_chunks, 3)
        max_chars = min(max_chars, 4_000)

    ranked = sorted(
        results,
        key=lambda item: (-item.score, item.rank if item.rank > 0 else 10**9),
    )
    selected: list[SearchResult] = []
    total_chars = 0
    for result in ranked:
        chunk_chars = len(result.chunk.enriched_content)
        if len(selected) >= max_chunks:
            break
        if selected and total_chars + chunk_chars > max_chars:
            continue
        selected.append(result)
        total_chars += chunk_chars
        if total_chars >= max_chars:
            break

    if not selected:
        selected = [ranked[0]]

    if len(selected) < len(results):
        logger.info(
            "Prompt evidence capped for query '%s': %s -> %s chunks, %s chars",
            query,
            len(results),
            len(selected),
            total_chars,
        )
    return selected


def _compress_graph_result(result: SearchResult, max_chars: int) -> SearchResult:
    """Shrink graph-derived evidence before prompt assembly."""
    content = result.chunk.enriched_content.strip()
    if len(content) <= max_chars:
        return result

    sections = [
        section.strip()
        for section in _GRAPH_SECTION_SPLIT_RE.split(content)
        if section.strip()
    ]
    if not sections:
        shortened = content[:max_chars].strip()
        return result.model_copy(
            update={"chunk": result.chunk.model_copy(update={"content": shortened, "context": ""})}
        )

    kept: list[str] = []
    total_chars = 0
    for section in sections:
        if section.startswith("Graph paths:"):
            section = "\n".join(section.splitlines()[:5]).strip()
        elif section.startswith("Entities:"):
            section = "\n".join(section.splitlines()[:7]).strip()
        elif section.startswith("Evidence:"):
            evidence_body = section[len("Evidence:"):].strip()
            section = ("Evidence:\n" + evidence_body[: max_chars // 2]).strip()

        projected = total_chars + len(section) + (2 if kept else 0)
        if kept and projected > max_chars:
            break
        kept.append(section[:max_chars].strip())
        total_chars = projected
        if total_chars >= max_chars:
            break

    compressed = "\n\n".join(part for part in kept if part).strip()
    if not compressed:
        compressed = content[:max_chars].strip()
    return result.model_copy(
        update={
            "chunk": result.chunk.model_copy(
                update={
                    "content": compressed,
                    "context": "",
                }
            )
        }
    )


def _compress_results_for_generation(results: list[SearchResult]) -> list[SearchResult]:
    """Apply source-aware prompt compression before message assembly."""
    cfg = get_settings()
    graph_char_budget = max(
        600,
        cfg.retrieval.prompt_max_chars // max(1, cfg.retrieval.prompt_max_chunks * 2),
    )
    compressed: list[SearchResult] = []
    for result in results:
        if result.source == "graph":
            compressed.append(_compress_graph_result(result, graph_char_budget))
        else:
            compressed.append(result)
    return compressed


def _clamp_confidence(value: float) -> float:
    cfg = get_settings()
    confidence_min = _retrieval_number(cfg, "confidence_min", 0.1)
    return round(min(1.0, max(confidence_min, value)), 3)


def _result_confidence_score(result: SearchResult) -> float:
    if result.score_normalized is not None:
        return result.score_normalized
    if 0.0 <= result.score <= 1.0:
        return result.score
    return 1.0


def _calculate_confidence(
    results: list[SearchResult],
    *,
    reflection_score: float | None = None,
) -> float:
    cfg = get_settings()
    avg_retrieval_score = sum(_result_confidence_score(result) for result in results) / len(results)
    if reflection_score is None:
        return _clamp_confidence(avg_retrieval_score)

    score_scale = _retrieval_number(cfg, "reflection_score_scale", 5.0)
    retrieval_weight = _retrieval_number(cfg, "retrieval_confidence_weight", 0.5)
    reflection_weight = _retrieval_number(cfg, "reflection_confidence_weight", 0.5)
    retrieval_score_normalized = avg_retrieval_score * score_scale
    fused_score = (
        (retrieval_weight * retrieval_score_normalized)
        + (reflection_weight * reflection_score)
    )
    return _clamp_confidence(fused_score / score_scale)


def calculate_answer_confidence(
    results: list[SearchResult],
    *,
    reflection_score: float | None = None,
) -> float:
    """Public confidence API shared by sync and streaming generation paths."""
    return _calculate_confidence(results, reflection_score=reflection_score)


def _build_context(results: list[SearchResult]) -> str:
    ordered_results = _order_results_for_prompt(results)
    context_chunks = []
    for i, result in enumerate(ordered_results, start=1):
        context_chunks.append(f"[Chunk {i}]\n{result.chunk.enriched_content}")
    return "\n\n".join(context_chunks)


def _build_system_prompt(query: str) -> str:
    if _is_enumeration_query(query):
        return (
            "You are an expert Q&A assistant specialized in comprehensive enumeration. "
            "Your task is to extract and list EVERY distinct item, component, decision, method, "
            "or concept mentioned across ALL provided context chunks.\n\n"
            "INSTRUCTIONS:\n"
            "1. Scan ALL chunks systematically — do not stop at the first few\n"
            "2. Create a NUMBERED LIST of every distinct item found\n"
            "3. For each item, provide a brief description (1-2 sentences)\n"
            "4. Combine information from multiple chunks about the same item\n"
            "5. Do NOT say 'the document does not list' — extract items even if "
            "they are discussed narratively rather than listed explicitly\n"
            "6. Answer in the same language as the query"
        )
    return (
        "You are a knowledgeable Q&A assistant. Synthesize information from ALL provided "
        "context chunks to give a comprehensive answer. Combine facts from different chunks "
        "when needed. If some details are missing, answer with what IS available rather than "
        "refusing. Cite chunk numbers used."
    )


def _build_messages(query: str, results: list[SearchResult]) -> list[dict[str, str]]:
    context = _build_context(results)
    system_prompt = _build_system_prompt(query)
    user_prompt = (
        f"Query: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Please provide an answer based on the above context."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _iter_stream_text(response_stream: Iterator[object]) -> Iterator[str]:
    for chunk in response_stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if content:
            yield str(content)


def generate_answer(
    query: str,
    results: list[SearchResult],
    openai_client: OpenAI | None = None,
    *,
    reflection_score: float | None = None,
) -> QAResult:
    """Generate answer from query and retrieved chunks using LLM."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    if not results:
        logger.warning("No results provided for answer generation")
        return QAResult(
            answer="I don't have enough context to answer this question.",
            sources=[],
            confidence=0.0,
            query=query,
        )

    selected_results = _compress_results_for_generation(
        _limit_results_for_generation(query, results)
    )
    messages = _build_messages(query, selected_results)
    prompt_chars = len(messages[1]["content"])
    evidence_chars = sum(len(result.chunk.enriched_content) for result in selected_results)

    logger.info(
        "Generating answer for query '%s' with %d/%d chunks, %d prompt chars, %d evidence chars",
        query,
        len(selected_results),
        len(results),
        prompt_chars,
        evidence_chars,
    )
    try:
        started = time.perf_counter()
        response = openai_client.chat.completions.create(
            model=cfg.openai.llm_model,
            messages=messages,
            temperature=cfg.openai.llm_temperature,
        )
        elapsed = time.perf_counter() - started

        answer_text = response.choices[0].message.content or ""
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        logger.info(
            "Generated answer in %.3fs (prompt_tokens=%s, completion_tokens=%s): %s",
            elapsed,
            prompt_tokens,
            completion_tokens,
            answer_text[:100],
        )

        confidence = calculate_answer_confidence(
            selected_results,
            reflection_score=reflection_score,
        )

        return QAResult(
            answer=answer_text,
            sources=selected_results,
            confidence=confidence,
            query=query,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return QAResult(
            answer=f"Error generating answer: {e}",
            sources=selected_results,
            confidence=0.0,
            query=query,
        )


def stream_answer(
    query: str,
    results: list[SearchResult],
    openai_client: OpenAI | None = None,
) -> Iterator[str]:
    """Stream answer tokens from the LLM using the same prompt assembly as sync generation."""
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    if not results:
        logger.warning("No results provided for streaming answer generation")
        return iter(())

    selected_results = _limit_results_for_generation(query, results)
    messages = _build_messages(query, selected_results)
    logger.info("Streaming answer for query: %s", query)

    def _stream() -> Iterator[str]:
        try:
            response = openai_client.chat.completions.create(
                model=cfg.openai.llm_model,
                messages=messages,
                temperature=cfg.openai.llm_temperature,
                stream=True,
            )
            yield from _iter_stream_text(response)
        except Exception as exc:
            logger.error("Error streaming answer: %s", exc)

    return _stream()
