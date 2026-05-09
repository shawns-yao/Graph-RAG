"""ROUGE metric from GraphRAG-Benchmark evaluation."""

from __future__ import annotations

from rouge_score import rouge_scorer


async def compute_rouge_score(
    answer: str,
    ground_truth: str,
    rouge_type: str = "rougeL",
    mode: str = "fmeasure",
) -> float:
    if not ground_truth.strip() or not answer.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(ground_truth, answer)
    return getattr(scores[rouge_type], mode)

