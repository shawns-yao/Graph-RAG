"""Coverage score metric adapted from GraphRAG-Benchmark."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from benchmark.metrics.utils import JSONHandler

FACT_EXTRACTION_PROMPT = """
You are given a question and a reference answer. Break down the reference answer into
a list of distinct factual statements under the `facts` field.

Question: "{question}"
Reference Answer: "{reference}"
"""


FACT_COVERAGE_PROMPT = """
### Task
For each factual statement from the reference, determine if it is covered in the response.
Respond ONLY with a JSON object containing a "classifications" list.
Each item should have "statement" and "attributed" (1 or 0).

Question: "{question}"
Response: "{response}"
Reference Facts: {facts}
"""


async def compute_coverage_score(
    question: str,
    reference: str,
    response: str,
    llm: Any,
    callbacks: Any = None,
    max_retries: int = 2,
) -> float:
    if not reference.strip():
        return 1.0
    facts = await _extract_facts(question, reference, llm, callbacks, max_retries)
    if not facts:
        return float(np.nan)
    coverage = await _check_fact_coverage(question, facts, response, llm, callbacks, max_retries)
    if coverage:
        attributed = [item["attributed"] for item in coverage]
        return float(sum(attributed) / len(attributed))
    return float(np.nan)


async def _extract_facts(
    question: str,
    reference: str,
    llm: Any,
    callbacks: Any,
    max_retries: int,
) -> list[str]:
    parser = JSONHandler(max_retries=max_retries)
    prompt = FACT_EXTRACTION_PROMPT.format(question=question, reference=reference[:3000])
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(response.content, callbacks=callbacks)
            facts = parsed.get("facts", []) if isinstance(parsed, dict) else parsed
            return _validate_facts(facts)
        except Exception:
            continue
    return []


def _validate_facts(facts: list) -> list[str]:
    if not isinstance(facts, list):
        return []
    return [str(fact).strip() for fact in facts if str(fact).strip()]


async def _check_fact_coverage(
    question: str,
    facts: list[str],
    response_text: str,
    llm: Any,
    callbacks: Any,
    max_retries: int,
) -> list[dict]:
    parser = JSONHandler(max_retries=max_retries)
    prompt = FACT_COVERAGE_PROMPT.format(
        question=question,
        response=response_text[:3000],
        facts=json.dumps(facts),
    )
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(
                response.content,
                key="classifications",
                callbacks=callbacks,
            )
            return _validate_classifications(parsed)
        except Exception:
            continue
    return []


def _validate_classifications(classifications: list) -> list[dict]:
    valid: list[dict] = []
    for item in classifications:
        if isinstance(item, dict) and "statement" in item and "attributed" in item:
            try:
                attributed = int(item["attributed"])
            except (TypeError, ValueError):
                continue
            if attributed not in {0, 1}:
                continue
            valid.append(
                {
                    "statement": str(item["statement"]),
                    "attributed": attributed,
                }
            )
    return valid

