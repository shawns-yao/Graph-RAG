#!/usr/bin/env python3
"""Evaluate a trained assertion-status classifier on JSONL data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_graph_rag.indexing.assertion_classifier import (  # noqa: E402
    AssertionClassifierConfig,
    TinyBertAssertionClassifier,
)
from agentic_graph_rag.indexing.assertion_rules import ASSERTION_LABELS  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_rows(rows: list[dict[str, Any]], model: TinyBertAssertionClassifier) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0
    for row in rows:
        gold = str(row["label"])
        pred = model.predict(str(row["text"]), str(row["entity"])).label
        confusion[gold][pred] += 1
        correct += int(gold == pred)

    per_label: dict[str, dict[str, float]] = {}
    for label in ASSERTION_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in ASSERTION_LABELS if other != label)
        fn = sum(confusion[label][other] for other in ASSERTION_LABELS if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values()),
        }
    return {
        "total": len(rows),
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "per_label": per_label,
        "confusion": {gold: dict(preds) for gold, preds in confusion.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/assertion_tinybert")
    parser.add_argument("--data", default="data/assertion/gold_reviewed_assertion.jsonl")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.data))
    model = TinyBertAssertionClassifier(
        AssertionClassifierConfig(
            model_path=args.model_dir,
            threshold=args.threshold,
            max_length=args.max_length,
        )
    )
    metrics = score_rows(rows, model)
    text = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
