"""Faithfulness metric adapted from GraphRAG-Benchmark."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from benchmark.metrics.utils import JSONHandler

STATEMENT_GENERATOR_PROMPT = """
Given a question and an answer, break the answer into atomic statements in JSON.

Question:{question}
Answer: {answer}

Generated Statements:
"""


FAITHFULNESS_EVALUATION_PROMPT = """
Your task is to judge the faithfulness of statements based on a context.
For each statement return verdict as 1 if directly supported by context, else 0.

Examples:
{examples}

Current Analysis:
Context: {context}
Statements: {statements}
"""


FAITHFULNESS_EXAMPLES = [
    {
        "input": {
            "context": "John studies Computer Science and often stays late in the library.",
            "statements": ["John studies Biology.", "John is a dedicated student."],
        },
        "output": [
            {"statement": "John studies Biology.", "reason": "Contradicted by context.", "verdict": 0},
            {"statement": "John is a dedicated student.", "reason": "Supported by his behavior.", "verdict": 1},
        ],
    }
]


async def compute_faithfulness_score(
    question: str,
    answer: str,
    contexts: list[str],
    llm: Any,
    callbacks: Any = None,
    max_retries: int = 2,
) -> float:
    statements = await _generate_statements(question, answer, llm, callbacks, max_retries)
    if not statements:
        return 1.0 if not answer.strip() else float(np.nan)
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0
    verdicts = await _evaluate_statements(statements, context_str, llm, callbacks, max_retries)
    if verdicts:
        supported = [item["verdict"] for item in verdicts]
        return float(sum(supported) / len(supported))
    return float(np.nan)


async def _generate_statements(
    question: str,
    answer: str,
    llm: Any,
    callbacks: Any,
    max_retries: int,
) -> list[str]:
    parser = JSONHandler(max_retries=max_retries)
    prompt = STATEMENT_GENERATOR_PROMPT.format(question=question, answer=answer)
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(response.content, callbacks=callbacks)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            if isinstance(parsed, dict):
                for key in ["statements", "answers", "items", "list", "output", "result"]:
                    value = parsed.get(key)
                    if isinstance(value, list):
                        return [str(item) for item in value]
            return []
        except Exception:
            continue
    return []


async def _evaluate_statements(
    statements: list[str],
    context: str,
    llm: Any,
    callbacks: Any,
    max_retries: int,
) -> list[dict]:
    parser = JSONHandler(max_retries=max_retries)
    examples = "\n".join(
        f"Input: {json.dumps(example['input'])}\nOutput: {json.dumps(example['output'])}"
        for example in FAITHFULNESS_EXAMPLES
    )
    prompt = FAITHFULNESS_EVALUATION_PROMPT.format(
        examples=examples,
        context=context[:10000],
        statements=json.dumps(statements)[:5000],
    )
    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(response.content, callbacks=callbacks)
            items = parsed if isinstance(parsed, list) else [parsed]
            return _validate_verdicts(items)
        except Exception:
            continue
    return []


def _validate_verdicts(items: list) -> list[dict]:
    valid: list[dict] = []
    for item in items:
        if isinstance(item, dict) and "statement" in item and "verdict" in item:
            try:
                verdict = int(item["verdict"])
            except (TypeError, ValueError):
                continue
            if verdict not in {0, 1}:
                continue
            valid.append(
                {
                    "statement": str(item["statement"]),
                    "verdict": verdict,
                    "reason": str(item.get("reason", "")),
                }
            )
    return valid

