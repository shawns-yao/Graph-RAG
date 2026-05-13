#!/usr/bin/env python3
"""Convert CBLUE task files into weak assertion-status JSONL data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_graph_rag.indexing.assertion_rules import (  # noqa: E402
    ASSERTION_LABELS,
    AssertionDecision,
    AssertionExample,
    classify_assertion_by_rules,
    extract_assertion_candidates,
    iter_sentence_spans,
)

TASK_FILE_PATTERNS = (
    "CMeEE_*.json",
    "CHIP-CTC_*.json",
    "CHIP-CDN_*.json",
    "CMeIE_*.json",
)


def iter_cblue_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for pattern in TASK_FILE_PATTERNS:
        yield from sorted(root.rglob(pattern))


def load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "questions", "items"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [payload]
    return []


def convert_cblue_root(root: Path, *, max_examples: int = 0) -> list[AssertionExample]:
    examples: list[AssertionExample] = []
    seen: set[tuple[str, str, int, int, str]] = set()
    for path in iter_cblue_files(root):
        task = infer_task_name(path)
        records = load_json_records(path)
        for index, record in enumerate(records):
            for example in convert_record(record, task=task, source=f"{path.name}:{index}"):
                key = (example.text, example.entity, example.start, example.end, example.label)
                if key in seen:
                    continue
                seen.add(key)
                examples.append(example)
                if max_examples and len(examples) >= max_examples:
                    return examples
    return examples


def convert_record(record: dict[str, Any], *, task: str, source: str) -> list[AssertionExample]:
    if task == "CMeEE":
        return convert_cmeee_record(record, task=task, source=source)
    if task == "CMeIE":
        return convert_cmeie_record(record, task=task, source=source)
    if task == "CHIP-CDN":
        return convert_cdn_record(record, task=task, source=source)
    return convert_text_record(record, task=task, source=source)


def convert_cmeee_record(record: dict[str, Any], *, task: str, source: str) -> list[AssertionExample]:
    text = normalized_text(record.get("text"))
    if not text:
        return []
    examples: list[AssertionExample] = []
    for entity in record.get("entities", []):
        if not isinstance(entity, dict):
            continue
        surface = normalized_text(entity.get("entity"))
        start = int(entity.get("start_idx", -1))
        end_inclusive = int(entity.get("end_idx", -1))
        if not surface and 0 <= start <= end_inclusive < len(text):
            surface = text[start:end_inclusive + 1]
        if surface:
            examples.extend(
                build_span_examples(
                    text,
                    surface,
                    start=start,
                    end=end_inclusive + 1,
                    source=f"cblue_weak:{task}:{source}",
                )
            )
    return examples


def convert_cmeie_record(record: dict[str, Any], *, task: str, source: str) -> list[AssertionExample]:
    text = normalized_text(record.get("text"))
    if not text:
        return []
    surfaces: list[str] = []
    for spo in record.get("spo_list", []):
        if not isinstance(spo, dict):
            continue
        surfaces.append(normalized_text(spo.get("subject")))
        obj = spo.get("object")
        if isinstance(obj, dict):
            surfaces.extend(normalized_text(value) for value in obj.values())
        else:
            surfaces.append(normalized_text(obj))
    examples: list[AssertionExample] = []
    for surface in surfaces:
        if surface:
            examples.extend(
                build_span_examples(
                    text,
                    surface,
                    start=text.find(surface),
                    end=text.find(surface) + len(surface),
                    source=f"cblue_weak:{task}:{source}",
                )
            )
    return examples


def convert_cdn_record(record: dict[str, Any], *, task: str, source: str) -> list[AssertionExample]:
    mention = normalized_text(record.get("text"))
    if not mention:
        return []
    return build_span_examples(
        mention,
        mention,
        start=0,
        end=len(mention),
        source=f"cblue_weak:{task}:{source}",
    )


def convert_text_record(record: dict[str, Any], *, task: str, source: str) -> list[AssertionExample]:
    text = normalized_text(
        record.get("text")
        or record.get("sentence")
        or record.get("criteria")
        or record.get("description")
    )
    if not text:
        return []
    examples: list[AssertionExample] = []
    for example in extract_assertion_candidates(text):
        examples.append(
            AssertionExample(
                text=example.text,
                entity=example.entity,
                label=example.label,
                start=example.start,
                end=example.end,
                cue=example.cue,
                source=f"cblue_weak:{task}:{source}",
                confidence=example.confidence,
                difficulty=example.difficulty,
                domain=example.domain,
            )
        )
    return examples


def build_span_examples(
    text: str,
    entity: str,
    *,
    start: int,
    end: int,
    source: str,
) -> list[AssertionExample]:
    if not entity:
        return []
    if start < 0 or end <= start or text[start:end] != entity:
        start = text.find(entity)
        end = start + len(entity)
    if start < 0:
        return []

    for sentence, sentence_start, sentence_end in iter_sentence_spans(text):
        if sentence_start <= start < sentence_end:
            rel_start = start - sentence_start
            rel_end = rel_start + len(entity)
            if sentence[rel_start:rel_end] != entity:
                return []
            decision = classify_assertion_by_rules(sentence, entity)
            return [
                make_example(
                    sentence=sentence,
                    entity=entity,
                    start=rel_start,
                    end=rel_end,
                    decision=decision,
                    source=source,
                )
            ]
    return []


def make_example(
    *,
    sentence: str,
    entity: str,
    start: int,
    end: int,
    decision: AssertionDecision,
    source: str,
) -> AssertionExample:
    return AssertionExample(
        text=sentence,
        entity=entity,
        label=decision.label,
        start=start,
        end=end,
        cue=decision.cue,
        source=source,
        confidence=decision.confidence,
        difficulty="easy" if decision.label == "affirmed" else "medium",
        domain="medical",
    )


def write_jsonl(path: Path, examples: list[AssertionExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(example.to_json(), ensure_ascii=False) for example in examples]
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_review_sample(path: Path, examples: list[AssertionExample], per_label: int) -> None:
    buckets: dict[str, list[AssertionExample]] = defaultdict(list)
    for example in examples:
        if len(buckets[example.label]) < per_label:
            buckets[example.label].append(example)
    sampled: list[AssertionExample] = []
    for label in ASSERTION_LABELS:
        sampled.extend(buckets[label])
    write_jsonl(path, sampled)


def write_stats(path: Path, examples: list[AssertionExample]) -> None:
    label_counts = Counter(example.label for example in examples)
    source_counts = Counter(example.source.split(":")[1] for example in examples)
    payload = {
        "total": len(examples),
        "labels": {label: label_counts.get(label, 0) for label in ASSERTION_LABELS},
        "sources": dict(sorted(source_counts.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def infer_task_name(path: Path) -> str:
    name = path.name
    for task in ("CMeEE", "CHIP-CTC", "CHIP-CDN", "CMeIE"):
        if name.startswith(task) or task in path.parts:
            return task
    return "UNKNOWN"


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/CBLUE-main", help="CBLUE root or task file.")
    parser.add_argument(
        "--output",
        default="data/assertion/cblue_weak_assertion.jsonl",
        help="Weak assertion JSONL output.",
    )
    parser.add_argument(
        "--review-output",
        default="data/assertion/cblue_review_sample.jsonl",
        help="Balanced manual-review sample.",
    )
    parser.add_argument(
        "--stats-output",
        default="data/assertion/cblue_label_stats.json",
        help="Label/source statistics output.",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--review-per-label", type=int, default=100)
    args = parser.parse_args()

    examples = convert_cblue_root(Path(args.input), max_examples=args.max_examples)
    write_jsonl(Path(args.output), examples)
    write_review_sample(Path(args.review_output), examples, args.review_per_label)
    write_stats(Path(args.stats_output), examples)
    print(f"wrote {len(examples)} CBLUE-derived assertion examples to {args.output}")


if __name__ == "__main__":
    main()
