#!/usr/bin/env python3
"""Compile manually reviewed assertion CSV rows into gold JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ("affirmed", "negated", "speculated", "conditional", "historical", "family_history")


def _resolve_entity_offsets(row: dict[str, str], line_number: int) -> tuple[int, int]:
    text = str(row.get("text", ""))
    entity = str(row.get("entity", ""))
    raw_start = str(row.get("start", "")).strip()
    raw_end = str(row.get("end", "")).strip()

    if raw_start and raw_end:
        try:
            start = int(raw_start)
            end = int(raw_end)
        except ValueError as exc:
            raise ValueError(f"Invalid start/end at line {line_number}: {raw_start}/{raw_end}") from exc
    else:
        start = text.find(entity)
        end = start + len(entity)

    if start < 0 or end > len(text) or start >= end:
        raise ValueError(f"Invalid entity offsets at line {line_number}: {start}/{end}")
    if text[start:end] != entity:
        raise ValueError(
            f"Offset validation failed at line {line_number}: "
            f"expected {entity}, got {text[start:end]}"
        )
    return start, end


def compile_gold_csv(input_path: Path, output_path: Path, stats_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            gold_label = str(row.get("gold_label", "")).strip()
            if not gold_label:
                skipped["missing_gold_label"] += 1
                continue
            if gold_label not in LABELS:
                raise ValueError(f"Invalid gold_label at line {line_number}: {gold_label}")

            text = str(row.get("text", ""))
            entity = str(row.get("entity", ""))
            start, end = _resolve_entity_offsets(row, line_number)

            rows.append(
                {
                    "text": text,
                    "entity": entity,
                    "label": gold_label,
                    "start": start,
                    "end": end,
                    "cue": str(row.get("cue", "")).strip(),
                    "source": f"gold_review:{row.get('id', line_number)}:{row.get('source', '')}",
                    "confidence": 1.0,
                    "difficulty": "gold",
                    "domain": "medical",
                    "weak_label": str(row.get("weak_label", "")).strip(),
                    "notes": str(row.get("notes", "")).strip(),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    labels = Counter(row["label"] for row in rows)
    weak_changed = sum(1 for row in rows if row.get("weak_label") and row["weak_label"] != row["label"])
    summary = {
        "total": len(rows),
        "labels": {label: labels.get(label, 0) for label in LABELS},
        "weak_label_changed": weak_changed,
        "skipped": dict(skipped),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/assertion/gold_review_todo.csv")
    parser.add_argument("--output", default="data/assertion/gold_reviewed_assertion.jsonl")
    parser.add_argument("--stats-output", default="data/assertion/gold_reviewed_stats.json")
    args = parser.parse_args()

    summary = compile_gold_csv(Path(args.input), Path(args.output), Path(args.stats_output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
