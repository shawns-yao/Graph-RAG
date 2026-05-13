#!/usr/bin/env python3
"""Export assertion JSONL samples to a CSV file for manual gold-label review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = ("affirmed", "negated", "speculated", "conditional", "historical", "family_history")
CSV_FIELDS = (
    "id",
    "gold_label",
    "weak_label",
    "entity",
    "answer_span",
    "start",
    "end",
    "cue",
    "text",
    "source",
    "notes",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_rows(rows: list[dict[str, Any]], per_label: int, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    buckets: dict[str, int] = defaultdict(int)
    for row in rows:
        label = str(row.get("label", ""))
        if label not in LABELS:
            continue
        if per_label and buckets[label] >= per_label:
            continue
        selected.append(row)
        buckets[label] += 1
        if limit and len(selected) >= limit:
            break
    return selected


def export_review_csv(input_path: Path, output_path: Path, *, per_label: int, limit: int) -> dict[str, Any]:
    rows = select_rows(load_jsonl(input_path), per_label=per_label, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(row.get("label", "")) for row in rows)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            text = str(row["text"])
            start = int(row["start"])
            end = int(row["end"])
            writer.writerow(
                {
                    "id": f"gold-{index:05d}",
                    "gold_label": "",
                    "weak_label": row["label"],
                    "entity": row["entity"],
                    "answer_span": text[start:end],
                    "start": start,
                    "end": end,
                    "cue": row.get("cue", ""),
                    "text": text,
                    "source": row.get("source", ""),
                    "notes": "",
                }
            )
    return {"total": len(rows), "labels": {label: counts.get(label, 0) for label in LABELS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/assertion/dialogue_review_sample.jsonl")
    parser.add_argument("--output", default="data/assertion/gold_review_todo.csv")
    parser.add_argument("--per-label", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    summary = export_review_csv(
        Path(args.input),
        Path(args.output),
        per_label=args.per_label,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
