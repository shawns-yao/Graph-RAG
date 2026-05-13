#!/usr/bin/env python3
"""Clean Chinese medical dialogue CSV files into assertion-status JSONL data."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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

ENCODINGS = ("utf-8-sig", "gb18030", "gbk")
TEXT_FIELDS = ("answer", "ask")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_BOILERPLATE_RE = re.compile(
    r"(?:感谢(?:您的)?咨询|希望(?:我的)?(?:回答|解释)对(?:你|您)有所帮助|祝(?:您)?(?:早日)?康复)"
)


@dataclass
class DialogueStats:
    files: int = 0
    rows: int = 0
    used_rows: int = 0
    examples: int = 0
    skipped_duplicates: int = 0
    labels: Counter[str] | None = None
    departments: Counter[str] | None = None
    fields: Counter[str] | None = None
    encodings: Counter[str] | None = None

    def __post_init__(self) -> None:
        self.labels = self.labels or Counter()
        self.departments = self.departments or Counter()
        self.fields = self.fields or Counter()
        self.encodings = self.encodings or Counter()

    def to_json(self) -> dict[str, object]:
        return {
            "files": self.files,
            "rows": self.rows,
            "used_rows": self.used_rows,
            "examples": self.examples,
            "skipped_duplicates": self.skipped_duplicates,
            "labels": {label: self.labels.get(label, 0) for label in ASSERTION_LABELS},
            "departments": dict(self.departments.most_common()),
            "fields": dict(self.fields.most_common()),
            "encodings": dict(self.encodings.most_common()),
        }


def iter_csv_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("*.csv"))


def detect_encoding(path: Path) -> str:
    sample = path.read_bytes()[:65536]
    for encoding in ENCODINGS:
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "gb18030"


def iter_dialogue_rows(path: Path) -> Iterable[tuple[dict[str, str], str]]:
    encoding = detect_encoding(path)
    with path.open("r", encoding=encoding, newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = {str(key or "").strip().lower(): str(value or "") for key, value in row.items()}
            yield normalized, encoding


def clean_text(value: str, *, max_chars: int) -> str:
    text = repair_mojibake(html.unescape(value or ""))
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("\u3000", " ").replace("\r", " ").replace("\n", " ")
    text = _BOILERPLATE_RE.sub("。", text)
    text = _SPACE_RE.sub(" ", text).strip(" ,，。；;")
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    return text


def repair_mojibake(value: str) -> str:
    """Repair common UTF-8/GBK text decoded through the wrong single-byte path."""
    if not value:
        return ""
    candidates = [value]
    raw = _single_byte_mojibake_bytes(value)
    if raw:
        for encoding in ("utf-8", "gb18030", "gbk"):
            try:
                candidates.append(raw.decode(encoding))
            except UnicodeDecodeError:
                pass
    return min(candidates, key=_mojibake_score)


def _single_byte_mojibake_bytes(value: str) -> bytes:
    buffer = bytearray()
    for char in value:
        codepoint = ord(char)
        if codepoint <= 255:
            buffer.append(codepoint)
            continue
        try:
            encoded = char.encode("cp1252")
        except UnicodeEncodeError:
            return b""
        if len(encoded) != 1:
            return b""
        buffer.extend(encoded)
    return bytes(buffer)


def _mojibake_score(value: str) -> tuple[int, int]:
    bad_chars = sum(1 for char in value if char in "ÃÂ�þýðÉñ¾¿Æçèéåäöü")
    cjk_chars = sum(1 for char in value if "\u4e00" <= char <= "\u9fff")
    return bad_chars - cjk_chars, len(value)


def build_dialogue_examples(
    input_path: Path,
    *,
    fields: tuple[str, ...] = TEXT_FIELDS,
    max_records: int = 0,
    max_examples: int = 0,
    max_per_label: int = 0,
    max_text_chars: int = 600,
) -> tuple[list[AssertionExample], DialogueStats]:
    examples: list[AssertionExample] = []
    stats = DialogueStats()
    seen: set[tuple[str, str, int, int, str]] = set()
    per_label: Counter[str] = Counter()

    for csv_path in iter_csv_files(input_path):
        stats.files += 1
        for row_index, (row, encoding) in enumerate(iter_dialogue_rows(csv_path), start=1):
            if max_records and stats.rows >= max_records:
                return examples, stats
            stats.rows += 1
            stats.encodings[encoding] += 1
            department = clean_text(row.get("department", ""), max_chars=80) or csv_path.parent.name
            row_used = False
            for field in fields:
                text = clean_text(row.get(field, ""), max_chars=max_text_chars)
                if len(text) < 6:
                    continue
                for example in extract_assertion_candidates(text):
                    if max_per_label and per_label[example.label] >= max_per_label:
                        continue
                    enriched = AssertionExample(
                        text=example.text,
                        entity=example.entity,
                        label=example.label,
                        start=example.start,
                        end=example.end,
                        cue=example.cue,
                        source=f"dialogue_weak:{department}:{field}:{csv_path.name}:{row_index}",
                        confidence=example.confidence,
                        difficulty=example.difficulty,
                        domain=example.domain,
                    )
                    if enriched.text[enriched.start:enriched.end] != enriched.entity:
                        continue
                    key = (
                        enriched.text,
                        enriched.entity,
                        enriched.start,
                        enriched.end,
                        enriched.label,
                    )
                    if key in seen:
                        stats.skipped_duplicates += 1
                        continue
                    seen.add(key)
                    examples.append(enriched)
                    per_label[enriched.label] += 1
                    stats.labels[enriched.label] += 1
                    stats.departments[department] += 1
                    stats.fields[field] += 1
                    stats.examples += 1
                    row_used = True
                    if max_examples and stats.examples >= max_examples:
                        if row_used:
                            stats.used_rows += 1
                        return examples, stats
            if row_used:
                stats.used_rows += 1
    return examples, stats


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


def write_stats(path: Path, stats: DialogueStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_fields(value: str) -> tuple[str, ...]:
    fields = tuple(field.strip().lower() for field in value.split(",") if field.strip())
    return fields or TEXT_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/Chinese-medical-dialogue-data-master/Data_数据",
        help="Dialogue data root or CSV file.",
    )
    parser.add_argument(
        "--output",
        default="data/assertion/dialogue_weak_assertion.jsonl",
        help="Weak assertion JSONL output path.",
    )
    parser.add_argument(
        "--review-output",
        default="data/assertion/dialogue_review_sample.jsonl",
        help="Balanced sample for manual review.",
    )
    parser.add_argument(
        "--stats-output",
        default="data/assertion/dialogue_label_stats.json",
        help="Cleaning statistics JSON path.",
    )
    parser.add_argument("--fields", default="answer", help="Comma-separated fields, e.g. answer,ask.")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-per-label", type=int, default=0)
    parser.add_argument("--max-text-chars", type=int, default=600)
    parser.add_argument("--review-per-label", type=int, default=100)
    args = parser.parse_args()

    examples, stats = build_dialogue_examples(
        Path(args.input),
        fields=parse_fields(args.fields),
        max_records=args.max_records,
        max_examples=args.max_examples,
        max_per_label=args.max_per_label,
        max_text_chars=args.max_text_chars,
    )
    write_jsonl(Path(args.output), examples)
    write_review_sample(Path(args.review_output), examples, args.review_per_label)
    write_stats(Path(args.stats_output), stats)
    print(json.dumps(stats.to_json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
