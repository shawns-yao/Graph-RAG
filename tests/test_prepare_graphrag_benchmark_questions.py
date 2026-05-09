"""Tests for GraphRAG-Benchmark question preparation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_graphrag_benchmark_questions import prepare_questions


def test_prepare_questions_maps_fields_for_runner(tmp_path: Path) -> None:
    source = tmp_path / "questions.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "Medical-1",
                    "question": "What is the most common type of skin cancer?",
                    "answer": "Basal cell carcinoma (BCC).",
                    "question_type": "Fact Retrieval",
                    "evidence": "Basal cell carcinoma (BCC) is common.",
                }
            ]
        ),
        encoding="utf-8",
    )

    target = tmp_path / "prepared.json"
    rows = prepare_questions(str(source), str(target))

    assert len(rows) == 1
    assert rows[0]["id"] == "Medical-1"
    assert rows[0]["type"] == "simple"
    assert rows[0]["reference_answer"] == "Basal cell carcinoma (BCC)."
    assert "basal" in rows[0]["keywords"]


def test_prepare_questions_applies_limit(tmp_path: Path) -> None:
    source = tmp_path / "questions.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "Question one",
                    "answer": "Answer one",
                    "question_type": "Fact Retrieval",
                    "evidence": "Evidence one",
                },
                {
                    "id": "q2",
                    "question": "Question two",
                    "answer": "Answer two",
                    "question_type": "Temporal Reasoning",
                    "evidence": "Evidence two",
                },
            ]
        ),
        encoding="utf-8",
    )

    target = tmp_path / "prepared.json"
    rows = prepare_questions(str(source), str(target), limit=1)

    assert len(rows) == 1
    assert rows[0]["id"] == "q1"
