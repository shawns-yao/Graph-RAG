"""Answer correctness metric adapted from GraphRAG-Benchmark."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
from pydantic import BaseModel

from benchmark.metrics.utils import JSONHandler


class StatementsWithReason(BaseModel):
    statement: str
    reason: str


class ClassificationWithReason(BaseModel):
    TP: list[StatementsWithReason] = []
    FP: list[StatementsWithReason] = []
    FN: list[StatementsWithReason] = []


def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)


STATEMENT_GENERATOR_PROMPT = """
Given a question and an answer, analyze the complexity of each sentence in the answer.
Break down each sentence into one or more fully understandable statements. Ensure that
no pronouns are used in any statement. Format the outputs in JSON.

Question:{question}
Answer: {answer}

Generated Statements:
"""


CORRECTNESS_PROMPT_TEMPLATE = """
Given a ground truth and answer statements, analyze each statement and classify them
as TP, FP, or FN. Provide a reason for each classification.

Examples:
{examples}

Current Analysis:
Question: {question}
Answer Statements: {answer}
Ground Truth Statements: {ground_truth}
"""


CORRECTNESS_EXAMPLES = [
    {
        "input": {
            "question": "What powers the sun and what is its primary function?",
            "answer": [
                "The sun is powered by nuclear fission.",
                "The primary function of the sun is to provide light.",
            ],
            "ground_truth": [
                "The sun is powered by nuclear fusion.",
                "The energy from the sun provides heat and light.",
            ],
        },
        "output": {
            "TP": [
                {
                    "statement": "The primary function of the sun is to provide light.",
                    "reason": "Supported by the ground truth mentioning light.",
                }
            ],
            "FP": [
                {
                    "statement": "The sun is powered by nuclear fission.",
                    "reason": "Contradicts the ground truth fusion statement.",
                }
            ],
            "FN": [
                {
                    "statement": "The sun is powered by nuclear fusion.",
                    "reason": "Missing from the answer.",
                }
            ],
        },
    }
]


async def compute_answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
    llm: Any,
    embeddings: Any,
    weights: list[float] = [0.75, 0.25],
    beta: float = 1.0,
    callbacks: Any = None,
) -> float:
    try:
        answer_statements = await generate_statements(llm, question, answer, callbacks)
        gt_statements = await generate_statements(llm, question, ground_truth, callbacks)
    except Exception:
        return 0.0
    factuality_score = 0.0
    similarity_score = 0.0
    if weights[0] != 0:
        try:
            factuality_score = await calculate_factuality(
                llm,
                question,
                answer_statements,
                gt_statements,
                callbacks,
                beta,
            )
        except Exception:
            factuality_score = 0.0
    if weights[1] != 0:
        try:
            similarity_score = await calculate_semantic_similarity(embeddings, answer, ground_truth)
        except Exception:
            similarity_score = 0.0
    return float(np.average([factuality_score, similarity_score], weights=weights))


async def generate_statements(llm: Any, question: str, answer: str, callbacks: Any) -> list[str]:
    handler = JSONHandler()
    prompt = STATEMENT_GENERATOR_PROMPT.format(question=question, answer=answer)
    response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
    parsed = await handler.parse_with_fallbacks(response.content)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, dict):
        for key in ["statements", "answers", "items", "list", "output", "result"]:
            value = parsed.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
        return [str(value) for value in parsed.values()]
    return [str(parsed)]


async def calculate_factuality(
    llm: Any,
    question: str,
    answer_stmts: list[str],
    gt_stmts: list[str],
    callbacks: Any,
    beta: float,
) -> float:
    if not answer_stmts and not gt_stmts:
        return 1.0
    examples = "\n".join(
        f"Input: {json.dumps(example['input'])}\nOutput: {json.dumps(example['output'])}"
        for example in CORRECTNESS_EXAMPLES
    )
    prompt = CORRECTNESS_PROMPT_TEMPLATE.format(
        examples=examples,
        question=question,
        answer=json.dumps(answer_stmts),
        ground_truth=json.dumps(gt_stmts),
    )
    response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
    try:
        classification = ClassificationWithReason(**json.loads(response.content))
        return fbeta_score(
            len(classification.TP),
            len(classification.FP),
            len(classification.FN),
            beta,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


async def calculate_semantic_similarity(embeddings: Any, answer: str, ground_truth: str) -> float:
    answer_embedding, gt_embedding = await asyncio.gather(
        embeddings.aembed_query(answer),
        embeddings.aembed_query(ground_truth),
    )
    cosine_sim = np.dot(answer_embedding, gt_embedding) / (
        np.linalg.norm(answer_embedding) * np.linalg.norm(gt_embedding)
    )
    return float((cosine_sim + 1) / 2)
