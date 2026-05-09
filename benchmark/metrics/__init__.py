"""GraphRAG-Benchmark metric exports."""

from benchmark.metrics.answer_accuracy import compute_answer_correctness
from benchmark.metrics.context_relevance import compute_context_relevance
from benchmark.metrics.coverage import compute_coverage_score
from benchmark.metrics.evidence_recall import compute_evidence_recall
from benchmark.metrics.faithfulness import compute_faithfulness_score
from benchmark.metrics.rouge import compute_rouge_score
from benchmark.metrics.utils import JSONHandler

__all__ = [
    "JSONHandler",
    "compute_answer_correctness",
    "compute_context_relevance",
    "compute_coverage_score",
    "compute_evidence_recall",
    "compute_faithfulness_score",
    "compute_rouge_score",
]

