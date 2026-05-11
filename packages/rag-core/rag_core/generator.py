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

from agentic_graph_rag.agent.query_signals import extract_query_signals
from rag_core.config import get_settings, make_openai_client
from rag_core.models import QAResult, SearchResult

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_ENUM_RE = None
_GRAPH_SECTION_SPLIT_RE = re.compile(r"\n\s*\n(?=(?:Graph paths:|Entities:|Evidence:))")
_ACRONYM_ANCHOR_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")
_PHRASE_ANCHOR_CLEAN_RE = re.compile(
    r"^(?:时|当|如果|若|患者|的患者|对于|关于|有关|在|对|和|与|及|、)+|"
    r"(?:怎么处理|如何处理|怎么办|是否正确|是否|可以用|需要注意|有哪些|是什么|是多少|吗|呢)$"
)


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


def _strong_anchor_texts(query: str) -> list[str]:
    signals = extract_query_signals(query)
    anchors: list[str] = []
    for anchor in signals.anchors:
        if anchor.kind not in {"numeric", "threshold", "symbolic", "quoted"}:
            continue
        text = anchor.text.casefold()
        if text and text not in anchors:
            anchors.append(text)
    return anchors


def _phrase_anchor_texts(query: str) -> list[str]:
    signals = extract_query_signals(query)
    anchors: list[str] = []
    for match in _ACRONYM_ANCHOR_RE.finditer(query):
        text = match.group(0).casefold()
        if text and text not in anchors:
            anchors.append(text)
    for anchor in signals.anchors:
        if anchor.kind != "phrase":
            continue
        candidates = [anchor.text]
        cleaned = anchor.text
        previous = ""
        while cleaned != previous:
            previous = cleaned
            cleaned = _PHRASE_ANCHOR_CLEAN_RE.sub("", cleaned).strip(" ，,。？?:：")
        if cleaned and cleaned != anchor.text:
            candidates.append(cleaned)
        for candidate in candidates:
            text = candidate.casefold()
            if text and text not in anchors:
                anchors.append(text)
    return anchors


def _contains_strong_anchor(result: SearchResult, anchors: list[str]) -> bool:
    if not anchors:
        return False
    content = (result.chunk.enriched_content or result.chunk.content or "").casefold()
    return any(anchor in content for anchor in anchors)


def _matched_phrase_anchors(result: SearchResult, anchors: list[str]) -> set[str]:
    if not anchors:
        return set()
    content = (result.chunk.enriched_content or result.chunk.content or "").casefold()
    return {anchor for anchor in anchors if anchor in content}


def _contains_phrase_anchor(result: SearchResult, anchors: list[str]) -> bool:
    return bool(_matched_phrase_anchors(result, anchors))


def _limit_results_for_generation(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Build an evidence pack without query-length prompt shortcuts."""
    cfg = get_settings()
    max_chunks = max(1, cfg.retrieval.prompt_max_chunks)
    max_chars = max(1, cfg.retrieval.prompt_max_chars)

    if _is_enumeration_query(query):
        max_chunks = min(max_chunks, 5)
        max_chars = min(max_chars, 8_000)

    anchors = _strong_anchor_texts(query)
    phrase_anchors = _phrase_anchor_texts(query)
    ranked = sorted(
        results,
        key=lambda item: (
            -len(_matched_phrase_anchors(item, phrase_anchors)),
            not _contains_strong_anchor(item, anchors),
            -item.score,
            item.rank if item.rank > 0 else 10**9,
        ),
    )
    selected: list[SearchResult] = []
    selected_ids: set[int] = set()
    covered_phrase_anchors: set[str] = set()
    total_chars = 0

    def try_select(result: SearchResult) -> bool:
        nonlocal total_chars
        if id(result) in selected_ids or len(selected) >= max_chunks:
            return False
        chunk_chars = len(result.chunk.enriched_content)
        if selected and total_chars + chunk_chars > max_chars:
            return False
        selected.append(result)
        selected_ids.add(id(result))
        covered_phrase_anchors.update(_matched_phrase_anchors(result, phrase_anchors))
        total_chars += chunk_chars
        return total_chars >= max_chars

    for anchor in phrase_anchors:
        if anchor in covered_phrase_anchors:
            continue
        for result in ranked:
            if anchor in _matched_phrase_anchors(result, phrase_anchors):
                if try_select(result):
                    break
                break
        if len(selected) >= max_chunks or total_chars >= max_chars:
            break

    if len(selected) < max_chunks and total_chars < max_chars:
        for result in ranked:
            if try_select(result):
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
