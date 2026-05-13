#!/usr/bin/env python3
"""Build weak assertion-status JSONL data from Chinese medical text files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_graph_rag.indexing.assertion_rules import (  # noqa: E402
    ASSERTION_LABELS,
    AssertionExample,
    extract_assertion_candidates,
)


def iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for pattern in ("*.txt", "*.md", "*.jsonl"):
        yield from sorted(path.rglob(pattern))


def read_records(path: Path) -> Iterable[tuple[str, str]]:
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            text = str(payload.get("text") or payload.get("content") or "")
            if text.strip():
                yield f"{path.name}:{line_number}", text
        return
    yield path.name, path.read_text(encoding="utf-8")


def build_examples(input_path: Path, max_examples: int) -> list[AssertionExample]:
    examples: list[AssertionExample] = []
    for file_path in iter_input_files(input_path):
        for source, text in read_records(file_path):
            for example in extract_assertion_candidates(text):
                enriched = AssertionExample(
                    text=example.text,
                    entity=example.entity,
                    label=example.label,
                    start=example.start,
                    end=example.end,
                    cue=example.cue,
                    source=f"weak_rule:{source}",
                    confidence=example.confidence,
                    difficulty=example.difficulty,
                    domain=example.domain,
                )
                if enriched.text[enriched.start:enriched.end] != enriched.entity:
                    continue
                examples.append(enriched)
                if max_examples and len(examples) >= max_examples:
                    return examples
    return examples


def write_jsonl(path: Path, examples: list[AssertionExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(example.to_json(), ensure_ascii=False) for example in examples]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_review_sample(
    path: Path,
    examples: list[AssertionExample],
    per_label: int,
) -> None:
    if per_label <= 0:
        return
    buckets: dict[str, list[AssertionExample]] = defaultdict(list)
    for example in examples:
        if len(buckets[example.label]) < per_label:
            buckets[example.label].append(example)
    sampled: list[AssertionExample] = []
    for label in ASSERTION_LABELS:
        sampled.extend(buckets[label])
    write_jsonl(path, sampled)


def write_stats(path: Path, examples: list[AssertionExample]) -> None:
    labels = Counter(example.label for example in examples)
    difficulty = Counter(example.difficulty for example in examples)
    payload = {
        "total": len(examples),
        "labels": {label: labels.get(label, 0) for label in ASSERTION_LABELS},
        "difficulty": dict(sorted(difficulty.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input file or directory.")
    parser.add_argument(
        "--output",
        default="data/assertion/weak_train.jsonl",
        help="Weak-label JSONL output path.",
    )
    parser.add_argument(
        "--review-output",
        default="data/assertion/review_sample.jsonl",
        help="Balanced sample for manual review.",
    )
    parser.add_argument(
        "--stats-output",
        default="data/assertion/label_stats.json",
        help="Label statistics JSON path.",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--review-per-label", type=int, default=50)
    args = parser.parse_args()

    examples = build_examples(Path(args.input), args.max_examples)
    write_jsonl(Path(args.output), examples)
    write_review_sample(Path(args.review_output), examples, args.review_per_label)
    write_stats(Path(args.stats_output), examples)
    print(f"wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
