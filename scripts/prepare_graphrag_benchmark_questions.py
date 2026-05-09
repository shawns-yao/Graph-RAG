#!/usr/bin/env python3
"""Prepare GraphRAG-Benchmark question files for the local benchmark runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

QUESTION_TYPE_MAP = {
    "Fact Retrieval": "simple",
    "Relation Reasoning": "relation",
    "Multi-hop Reasoning": "multi_hop",
    "Global Summarization": "global",
    "Temporal Reasoning": "temporal",
}


def _load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Question file must contain a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def _normalize_question_type(value: Any) -> str:
    if not isinstance(value, str):
        return "simple"
    return QUESTION_TYPE_MAP.get(value.strip(), "simple")


def _extract_keywords(question: str, answer: str, evidence: str) -> list[str]:
    raw_terms = f"{question} {answer} {evidence}".replace(";", " ").split()
    keywords: list[str] = []
    seen: set[str] = set()
    for token in raw_terms:
        cleaned = token.strip(".,:;!?()[]{}\"'").lower()
        if len(cleaned) < 4 or cleaned in seen:
            continue
        seen.add(cleaned)
        keywords.append(cleaned)
        if len(keywords) >= 12:
            break
    return keywords


def prepare_questions(
    source_path: str,
    output_path: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    source = Path(source_path)
    rows = _load_questions(source)
    prepared: list[dict[str, Any]] = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        evidence = str(row.get("evidence") or "").strip()
        if not question:
            continue
        prepared.append(
            {
                "id": row.get("id"),
                "question": question,
                "type": _normalize_question_type(row.get("question_type")),
                "reference_answer": answer,
                "keywords": _extract_keywords(question, answer, evidence),
            }
        )
        if limit is not None and len(prepared) >= limit:
            break
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GraphRAG-Benchmark questions for local runner")
    parser.add_argument("source_path", help="Path to benchmark question JSON file")
    parser.add_argument("output_path", help="Path to prepared output JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Optional max questions to emit")
    args = parser.parse_args()

    questions = prepare_questions(
        args.source_path,
        args.output_path,
        limit=args.limit,
    )
    print(f"prepared_questions={len(questions)}")


if __name__ == "__main__":
    main()
