"""Context relevance metric adapted from GraphRAG-Benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmark.metrics.utils import JSONHandler

CONTEXT_RELEVANCE_PROMPT = """
### Instructions
You are a world class expert designed to evaluate the relevance score of a Context
in order to answer the Question. Use only the Context and the Question.

Scoring rules:
0. No relevant information.
1. Partial relevant information.
2. Fully contains relevant information.

Output strictly as JSON: {"score": 0|1|2}

Question: {question}
Context: {context}
/no_think
"""


async def compute_context_relevance(
    question: str,
    contexts: list[str],
    llm: Any,
    callbacks: Any = None,
    max_retries: int = 2,
) -> float:
    if not question.strip() or not contexts or not any(context.strip() for context in contexts):
        return 0.0
    context_str = "\n".join(contexts)
    if context_str.strip() == question.strip() or context_str.strip() in question:
        return 0.0
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context_str[:20000])
    rating1 = await _get_llm_rating(prompt, llm, callbacks, max_retries)
    rating2 = await _get_llm_rating(prompt, llm, callbacks, max_retries)
    scores = [rating / 2 for rating in (rating1, rating2) if rating is not None]
    if not scores:
        return float(np.nan)
    return float(sum(scores) / len(scores))


async def _get_llm_rating(prompt: str, llm: Any, callbacks: Any, max_retries: int) -> float | None:
    parser = JSONHandler(max_retries=max_retries)
    for _ in range(max_retries):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(response.content, callbacks=callbacks)
            return _normalize_rating(parsed)
        except Exception:
            continue
    return None


def _normalize_rating(parsed: Any) -> float | None:
    if isinstance(parsed, dict):
        score = parsed.get("rating", parsed.get("score"))
        if score in {0, 1, 2, 0.0, 1.0, 2.0}:
            return float(score)
    if isinstance(parsed, list) and len(parsed) == 1 and parsed[0] in {0, 1, 2, 0.0, 1.0, 2.0}:
        return float(parsed[0])
    if isinstance(parsed, str):
        stripped = parsed.strip()
        for token in stripped.replace("{", " ").replace("}", " ").split():
            try:
                value = float(token.strip('",:'))
            except ValueError:
                continue
            if value in {0.0, 1.0, 2.0}:
                return value
    return None

