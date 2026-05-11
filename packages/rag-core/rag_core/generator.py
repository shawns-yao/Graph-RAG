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


def derive_retrieval_status(results: list[SearchResult]) -> str:
    """Map retrieval output to a discrete status, not a user-facing score."""
    return "complete" if results else "empty"


def derive_answer_status(
    *,
    retrieval_status: str,
    reflection_verdict: str = "",
) -> str:
    """Classify answer state with enums only; no numeric confidence proxy."""
    if retrieval_status == "empty":
        return "failed"
    verdict = (reflection_verdict or "").strip().lower()
    if verdict in {"retry", "rerank"}:
        return "retry_required"
    if verdict == "stop":
        return "partial"
    return "unverified"


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
        "You are a precise Q&A assistant. Answer only the fields explicitly asked "
        "in the query. Use the provided context as evidence, but do not add related "
        "facts, background, classifications, timing, mechanisms, or caveats unless "
        "the query asks for them. If evidence contains extra facts, ignore them. "
        "Cite chunk numbers used."
    )


def _build_messages(query: str, results: list[SearchResult]) -> list[dict[str, str]]:
    context = _build_context(results)
    system_prompt = _build_system_prompt(query)
    user_prompt = (
        f"Query: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer the query directly. Do not include facts that are not needed to "
        "answer the exact question."
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
    reflection_verdict: str = "",
) -> QAResult:
    """Generate answer from query and retrieved chunks using LLM.

    The `reflection_verdict` parameter (one of "answer"/"retry"/"stop"/"rerank")
    is used to derive discrete answer status only.
    """
    cfg = get_settings()
    if openai_client is None:
        openai_client = make_openai_client(cfg)

    if not results:
        logger.warning("No results provided for answer generation")
        return QAResult(
            answer="I don't have enough context to answer this question.",
            sources=[],
            answer_status="failed",
            retrieval_status="empty",
            verification_status="skipped",
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

        retrieval_status = derive_retrieval_status(selected_results)

        return QAResult(
            answer=answer_text,
            sources=selected_results,
            answer_status=derive_answer_status(
                retrieval_status=retrieval_status,
                reflection_verdict=reflection_verdict,
            ),
            retrieval_status=retrieval_status,
            verification_status="skipped",
            query=query,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return QAResult(
            answer=f"Error generating answer: {e}",
            sources=selected_results,
            answer_status="failed",
            retrieval_status=derive_retrieval_status(selected_results),
            verification_status="skipped",
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
