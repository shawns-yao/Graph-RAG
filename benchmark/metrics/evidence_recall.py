"""Evidence recall metric adapted from GraphRAG-Benchmark."""

from __future__ import annotations

import numpy as np

from benchmark.metrics.utils import JSONHandler

EVIDENCE_RECALL_PROMPT = """
### Task
You are given a list of evidences and a Context. For each evidence, determine whether
it can be attributed to the Context.

Respond ONLY with a JSON object containing a "classifications" list. Each item should include:
- "statement"
- "reason"
- "attributed": 1 or 0

Context: "{context}"
Evidence: {evidence}
Question: "{question}"
"""


async def compute_evidence_recall(
    question: str,
    contexts: list[str],
    reference_evidence: list[str],
    llm: object,
    callbacks: object = None,
    max_retries: int = 2,
) -> float:
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0
    prompt = EVIDENCE_RECALL_PROMPT.format(
        question=question,
        context=context_str[:20000],
        evidence=reference_evidence,
    )
    classifications = await _get_classifications(prompt, llm, callbacks, max_retries)
    if classifications:
        attributed = [item["attributed"] for item in classifications]
        return float(sum(attributed) / len(attributed))
    return float(np.nan)


async def _get_classifications(prompt: str, llm: object, callbacks: object, max_retries: int) -> list[dict]:
    parser = JSONHandler(max_retries=max_retries)
    for _ in range(max_retries):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            classifications = await parser.parse_with_fallbacks(
                response.content,
                key="classifications",
                callbacks=callbacks,
            )
            return _validate_classifications(classifications)
        except Exception:
            continue
    return []


def _validate_classifications(classifications: list) -> list[dict]:
    valid: list[dict] = []
    for item in classifications:
        try:
            if (
                isinstance(item, dict)
                and "statement" in item
                and "reason" in item
                and "attributed" in item
                and item["attributed"] in {0, 1}
            ):
                valid.append(
                    {
                        "statement": str(item["statement"]),
                        "reason": str(item["reason"]),
                        "attributed": int(item["attributed"]),
                    }
                )
        except (TypeError, ValueError):
            continue
    return valid

