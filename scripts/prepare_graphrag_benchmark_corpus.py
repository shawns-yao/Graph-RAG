#!/usr/bin/env python3
"""Expand GraphRAG-Benchmark JSON corpora into plain text files for ingest."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return normalized or "document"


def _load_corpus_records(corpus_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError(f"Unsupported corpus payload type: {type(payload)!r}")


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def _split_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return [text]
    segments: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        separator_len = 2 if current else 0
        if current and current_len + separator_len + paragraph_len > max_chars:
            segments.append("\n\n".join(current))
            current = [paragraph]
            current_len = paragraph_len
            continue
        if paragraph_len > max_chars:
            if current:
                segments.append("\n\n".join(current))
                current = []
                current_len = 0
            for start in range(0, paragraph_len, max_chars):
                piece = paragraph[start : start + max_chars].strip()
                if piece:
                    segments.append(piece)
            continue
        current.append(paragraph)
        current_len += separator_len + paragraph_len
    if current:
        segments.append("\n\n".join(current))
    return segments or [text]


def prepare_corpus(
    corpus_path: str,
    output_dir: str,
    *,
    limit: int | None = None,
    max_chars_per_file: int = 0,
    overwrite: bool = False,
) -> list[Path]:
    source = Path(corpus_path)
    target_dir = Path(output_dir)
    records = _load_corpus_records(source)
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, record in enumerate(records, start=1):
        if limit is not None and len(written) >= limit:
            break
        context = _normalize_text(record.get("context"))
        if not context:
            continue
        corpus_name = str(record.get("corpus_name") or source.stem)
        segments = _split_text(context, max_chars_per_file)
        for segment_index, segment in enumerate(segments, start=1):
            if limit is not None and len(written) >= limit:
                break
            suffix = f"_part{segment_index:03d}" if len(segments) > 1 else ""
            file_name = f"{index:04d}_{_slugify(corpus_name)}{suffix}.txt"
            destination = target_dir / file_name
            if destination.exists() and not overwrite:
                written.append(destination)
                continue
            destination.write_text(segment, encoding="utf-8")
            written.append(destination)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GraphRAG-Benchmark corpus for ingest")
    parser.add_argument("corpus_path", help="Path to benchmark corpus JSON file")
    parser.add_argument("output_dir", help="Directory where extracted .txt files will be written")
    parser.add_argument("--limit", type=int, default=None, help="Optional max documents to emit")
    parser.add_argument(
        "--max-chars-per-file",
        type=int,
        default=0,
        help="Split oversized context into multiple files capped by this character budget",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    files = prepare_corpus(
        args.corpus_path,
        args.output_dir,
        limit=args.limit,
        max_chars_per_file=args.max_chars_per_file,
        overwrite=args.overwrite,
    )
    for path in files:
        print(path.resolve())
    print(f"prepared_files={len(files)}")


if __name__ == "__main__":
    main()
