"""Tests for assertion silver-set pre-labeling."""

from agentic_graph_rag.indexing.assertion_classifier import AssertionPrediction
from scripts.prelabel_assertion_silver_set import build_silver_rows


class _Predictor:
    def __init__(self, labels):
        self._labels = list(labels)

    def predict(self, text, entity):
        label = self._labels.pop(0)
        return AssertionPrediction(label=label, confidence=0.91, model="test-model")


def test_silver_prelabelling_prefers_high_confidence_non_affirmed_rule():
    rows = [
        {
            "text": "目前不能排除冠心病。",
            "entity": "冠心病",
            "label": "affirmed",
            "start": 6,
            "end": 9,
            "source": "test",
        }
    ]

    csv_rows, jsonl_rows, summary = build_silver_rows(
        rows,
        _Predictor(["affirmed"]),
        threshold=0.75,
    )

    assert csv_rows[0]["rule_label"] == "speculated"
    assert csv_rows[0]["suggested_gold_label"] == "speculated"
    assert csv_rows[0]["needs_review"] == "true"
    assert jsonl_rows[0]["label"] == "speculated"
    assert summary["labels"]["speculated"] == 1


def test_silver_prelabelling_keeps_weak_label_when_model_agrees():
    rows = [
        {
            "text": "患者存在高血压。",
            "entity": "高血压",
            "label": "affirmed",
            "start": 4,
            "end": 7,
            "source": "test",
        }
    ]

    csv_rows, jsonl_rows, summary = build_silver_rows(
        rows,
        _Predictor(["affirmed"]),
        threshold=0.75,
    )

    assert csv_rows[0]["suggested_gold_label"] == "affirmed"
    assert csv_rows[0]["needs_review"] == "false"
    assert jsonl_rows[0]["confidence"] == 0.8
    assert summary["agreements"]["weak_model"] == 1.0
