"""GraphRAG-Benchmark style retrieval evaluation for local runs."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from benchmark.metrics import compute_context_relevance, compute_evidence_recall


async def evaluate_sample(
    question: str,
    contexts: list[str],
    evidences: list[str],
    llm: Any,
) -> dict[str, float]:
    context_relevancy, evidence_recall = await asyncio.gather(
        compute_context_relevance(question, contexts, llm),
        compute_evidence_recall(question, contexts, evidences, llm),
        return_exceptions=True,
    )
    return {
        "context_relevancy": context_relevancy,
        "evidence_recall": evidence_recall,
    }


async def evaluate_dataset(
    dataset: list[dict[str, Any]],
    llm: Any,
    max_concurrent: int = 3,
) -> dict[str, float]:
    semaphore = asyncio.Semaphore(max_concurrent)
    metrics: dict[str, list[float]] = {
        "context_relevancy": [],
        "evidence_recall": [],
    }

    async def evaluate_row(row: dict[str, Any]) -> dict[str, float]:
        async with semaphore:
            return await evaluate_sample(
                question=row["question"],
                contexts=row["contexts"],
                evidences=row["evidences"],
                llm=llm,
            )

    for row_result in await asyncio.gather(*(evaluate_row(row) for row in dataset)):
        for metric_name, score in row_result.items():
            if isinstance(score, Exception):
                continue
            if not np.isnan(score):
                metrics[metric_name].append(score)

    return {
        metric_name: float(np.nanmean(scores)) if scores else float("nan")
        for metric_name, scores in metrics.items()
    }
