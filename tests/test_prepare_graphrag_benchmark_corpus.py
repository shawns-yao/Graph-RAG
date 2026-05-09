"""Tests for GraphRAG-Benchmark corpus preparation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_graphrag_benchmark_corpus import prepare_corpus


def test_prepare_corpus_supports_single_dict_payload(tmp_path: Path) -> None:
    corpus_path = tmp_path / "medical.json"
    corpus_path.write_text(
        json.dumps(
            {
                "corpus_name": "Medical",
                "context": "Section A\n\nSection B",
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "prepared"
    files = prepare_corpus(str(corpus_path), str(output_dir))

    assert len(files) == 1
    assert files[0].name == "0001_medical.txt"
    assert files[0].read_text(encoding="utf-8") == "Section A\n\nSection B"


def test_prepare_corpus_supports_list_payload_and_limit(tmp_path: Path) -> None:
    corpus_path = tmp_path / "novel.json"
    corpus_path.write_text(
        json.dumps(
            [
                {"corpus_name": "Novel-1", "context": "First context"},
                {"corpus_name": "Novel-2", "context": "Second context"},
                {"corpus_name": "Novel-3", "context": ""},
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "prepared"
    files = prepare_corpus(str(corpus_path), str(output_dir), limit=1)

    assert len(files) == 1
    assert files[0].name == "0001_novel-1.txt"
    assert files[0].read_text(encoding="utf-8") == "First context"


def test_prepare_corpus_splits_oversized_context_by_character_budget(tmp_path: Path) -> None:
    corpus_path = tmp_path / "medical.json"
    corpus_path.write_text(
        json.dumps(
            {
                "corpus_name": "Medical",
                "context": "A" * 12 + "\n\n" + "B" * 12 + "\n\n" + "C" * 12,
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "prepared"
    files = prepare_corpus(
        str(corpus_path),
        str(output_dir),
        max_chars_per_file=20,
    )

    assert len(files) == 3
    assert files[0].name == "0001_medical_part001.txt"
    assert files[1].name == "0001_medical_part002.txt"
    assert files[2].name == "0001_medical_part003.txt"
