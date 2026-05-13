#!/usr/bin/env python3
"""Create a pre-labeled silver assertion set for manual review and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_graph_rag.indexing.assertion_classifier import (  # noqa: E402
    AssertionClassifierConfig,
    AssertionPrediction,
    TinyBertAssertionClassifier,
)
from agentic_graph_rag.indexing.assertion_rules import (  # noqa: E402
    ASSERTION_LABELS,
    AssertionLabel,
    classify_assertion_by_rules,
)

CSV_FIELDS = (
    "id",
    "weak_label",
    "model_label",
    "model_confidence",
    "model_source",
    "rule_label",
    "rule_confidence",
    "suggested_gold_label",
    "needs_review",
    "entity",
    "answer_span",
    "start",
    "end",
    "cue",
    "text",
    "source",
    "notes",
)


class AssertionPredictor(Protocol):
    def predict(self, text: str, entity: str) -> AssertionPrediction:
        ...


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_offsets(row: dict[str, Any], line_number: int) -> tuple[int, int]:
    text = str(row["text"])
    entity = str(row["entity"])
    start = int(row["start"])
    end = int(row["end"])
    if start < 0 or end > len(text) or start >= end:
        raise ValueError(f"Invalid offsets at row {line_number}: {start}/{end}")
    if text[start:end] != entity:
        raise ValueError(
            f"Offset validation failed at row {line_number}: "
            f"expected {entity}, got {text[start:end]}"
        )
    return start, end


def suggest_label(
    *,
    weak_label: AssertionLabel,
    model_label: AssertionLabel,
    rule_label: AssertionLabel,
    rule_confidence: float,
    threshold: float,
) -> tuple[AssertionLabel, bool, str]:
    if rule_label != "affirmed" and rule_confidence >= threshold:
        needs_review = model_label != rule_label or weak_label != rule_label
        return rule_label, needs_review, "高置信规则命中，优先采用 rule_label；模型或弱标分歧时保留复核标记。"
    if model_label == weak_label:
        return weak_label, False, "model_label 与 weak_label 一致，保留弱标注作为银标建议。"
    if rule_label == weak_label:
        return weak_label, True, "rule_label 与 weak_label 一致，但 model_label 分歧，建议人工复核。"
    return weak_label, True, "weak/model/rule 三方不一致或规则置信不足，暂保留 weak_label 并要求人工复核。"


def build_silver_rows(
    rows: list[dict[str, Any]],
    predictor: AssertionPredictor,
    *,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    csv_rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    suggested_counts: Counter[str] = Counter()
    agreement_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        text = str(row["text"])
        entity = str(row["entity"])
        start, end = _validate_offsets(row, index)
        weak_label = str(row["label"])
        if weak_label not in ASSERTION_LABELS:
            raise ValueError(f"Invalid weak label at row {index}: {weak_label}")

        rule = classify_assertion_by_rules(text, entity)
        model = predictor.predict(text, entity)
        suggested, needs_review, notes = suggest_label(
            weak_label=weak_label,  # type: ignore[arg-type]
            model_label=model.label,
            rule_label=rule.label,
            rule_confidence=rule.confidence,
            threshold=threshold,
        )

        answer_span = text[start:end]
        csv_row = {
            "id": f"silver-{index:05d}",
            "weak_label": weak_label,
            "model_label": model.label,
            "model_confidence": round(model.confidence, 4),
            "model_source": model.model,
            "rule_label": rule.label,
            "rule_confidence": round(rule.confidence, 4),
            "suggested_gold_label": suggested,
            "needs_review": str(needs_review).lower(),
            "entity": entity,
            "answer_span": answer_span,
            "start": start,
            "end": end,
            "cue": rule.cue or row.get("cue", ""),
            "text": text,
            "source": row.get("source", ""),
            "notes": notes,
        }
        csv_rows.append(csv_row)
        jsonl_rows.append(
            {
                "text": text,
                "entity": entity,
                "label": suggested,
                "start": start,
                "end": end,
                "cue": csv_row["cue"],
                "source": f"silver_prelabeled:{csv_row['id']}:{row.get('source', '')}",
                "confidence": 0.8 if not needs_review else 0.6,
                "difficulty": "silver",
                "domain": "medical",
                "weak_label": weak_label,
                "model_label": model.label,
                "model_confidence": round(model.confidence, 4),
                "model_source": model.model,
                "rule_label": rule.label,
                "rule_confidence": round(rule.confidence, 4),
                "needs_review": needs_review,
                "notes": notes,
            }
        )
        suggested_counts[suggested] += 1
        agreement_counts["weak_model"] += int(weak_label == model.label)
        agreement_counts["weak_rule"] += int(weak_label == rule.label)
        agreement_counts["model_rule"] += int(model.label == rule.label)
        agreement_counts["needs_review"] += int(needs_review)

    total = len(rows)
    summary = {
        "total": total,
        "labels": {label: suggested_counts.get(label, 0) for label in ASSERTION_LABELS},
        "agreements": {
            key: round(value / total, 4) if total else 0.0
            for key, value in agreement_counts.items()
        },
        "counts": dict(agreement_counts),
        "label_source": "silver_prelabeled",
    }
    return csv_rows, jsonl_rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/assertion/dialogue_review_sample.jsonl")
    parser.add_argument("--model-dir", default="models/assertion_tinybert")
    parser.add_argument("--csv-output", default="data/assertion/silver_review_prelabeled.csv")
    parser.add_argument("--jsonl-output", default="data/assertion/silver_assertion.jsonl")
    parser.add_argument("--stats-output", default="data/assertion/silver_assertion_stats.json")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-length", type=int, default=160)
    args = parser.parse_args()

    predictor = TinyBertAssertionClassifier(
        AssertionClassifierConfig(
            model_path=args.model_dir,
            threshold=args.threshold,
            max_length=args.max_length,
        )
    )
    csv_rows, jsonl_rows, summary = build_silver_rows(
        load_jsonl(Path(args.input)),
        predictor,
        threshold=args.threshold,
    )
    write_csv(Path(args.csv_output), csv_rows)
    write_jsonl(Path(args.jsonl_output), jsonl_rows)
    Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
