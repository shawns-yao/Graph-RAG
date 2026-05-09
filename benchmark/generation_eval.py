"""GraphRAG-Benchmark style generation evaluation for local runs."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from benchmark.metrics import (
    compute_answer_correctness,
    compute_coverage_score,
    compute_faithfulness_score,
    compute_rouge_score,
)


async def evaluate_sample(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    metrics: list[str],
    llm: Any,
    embeddings: Any,
) -> dict[str, float]:
    tasks: dict[str, Any] = {}
    if "rouge_score" in metrics:
        tasks["rouge_score"] = compute_rouge_score(answer, ground_truth)
    if "answer_correctness" in metrics:
        tasks["answer_correctness"] = compute_answer_correctness(question, answer, ground_truth, llm, embeddings)
    if "coverage_score" in metrics:
        tasks["coverage_score"] = compute_coverage_score(question, ground_truth, answer, llm)
    if "faithfulness" in metrics:
        tasks["faithfulness"] = compute_faithfulness_score(question, answer, contexts, llm)
    task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return {metric: task_results[index] for index, metric in enumerate(tasks.keys())}


async def evaluate_dataset(
    dataset: list[dict[str, Any]],
    metrics: list[str],
    llm: Any,
    embeddings: Any,
    max_concurrent: int = 3,
) -> dict[str, float]:
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[str, list[float]] = {metric: [] for metric in metrics}

    async def evaluate_row(row: dict[str, Any]) -> dict[str, float]:
        async with semaphore:
            return await evaluate_sample(
                question=row["question"],
                answer=row["answer"],
                contexts=row["contexts"],
                ground_truth=row["ground_truth"],
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
            )

    for row_result in await asyncio.gather(*(evaluate_row(row) for row in dataset)):
        for metric, score in row_result.items():
            if isinstance(score, Exception):
                continue
            if not np.isnan(score):
                results[metric].append(score)
    return {
        metric: float(np.nanmean(scores)) if scores else float("nan")
        for metric, scores in results.items()
    }
