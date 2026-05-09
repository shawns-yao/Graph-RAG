#!/usr/bin/env python3
"""Compare legacy flat chunking vs graph-oriented chunking for Graph RAG ingest."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agentic_graph_rag.indexing.skeleton import (  # noqa: E402
    build_knn_graph,
    compute_pagerank,
    select_skeletal_chunks,
)
from rag_core.chunker import chunk_document, chunk_document_for_graph  # noqa: E402
from rag_core.loader import load_document  # noqa: E402


def _entity_count(chunk) -> int:
    value = chunk.metadata.get("graph_entity_count")
    if isinstance(value, int):
        return value
    return 0


def _summarize_chunks(chunks: list) -> dict[str, float | int]:
    if not chunks:
        return {
            "chunks": 0,
            "avg_chars": 0.0,
            "avg_entity_count": 0.0,
            "skeleton_candidates": 0,
            "peripheral_candidates": 0,
        }
    lengths = [len(chunk.content) for chunk in chunks]
    entity_counts = [_entity_count(chunk) for chunk in chunks]
    return {
        "chunks": len(chunks),
        "avg_chars": round(statistics.mean(lengths), 2),
        "avg_entity_count": round(statistics.mean(entity_counts), 2),
        "skeleton_candidates": sum(
            1 for chunk in chunks if chunk.metadata.get("graph_chunk_type") == "skeleton_candidate"
        ),
        "peripheral_candidates": sum(
            1 for chunk in chunks if chunk.metadata.get("graph_chunk_type") == "peripheral_candidate"
        ),
    }


def _mock_embeddings(chunks: list) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for index, chunk in enumerate(chunks):
        entity_count = _entity_count(chunk)
        length = max(1.0, float(len(chunk.content)))
        embeddings.append(
            [
                float(entity_count + 1),
                float(length / 1000.0),
                float((index % 7) + 1),
                float((entity_count + len(chunk.content.split())) / 10.0),
            ]
        )
    return embeddings


def _skeleton_metrics(chunks: list) -> dict[str, float | int]:
    if not chunks:
        return {
            "skeletal_chunks": 0,
            "peripheral_chunks": 0,
            "skeletal_avg_entity_count": 0.0,
            "peripheral_avg_entity_count": 0.0,
        }
    embeddings = _mock_embeddings(chunks)
    graph = build_knn_graph(chunks, embeddings)
    pagerank_scores = compute_pagerank(graph)
    skeletal, peripheral = select_skeletal_chunks(chunks, pagerank_scores)
    skeletal_entities = [_entity_count(chunk) for chunk in skeletal]
    peripheral_entities = [_entity_count(chunk) for chunk in peripheral]
    return {
        "skeletal_chunks": len(skeletal),
        "peripheral_chunks": len(peripheral),
        "skeletal_avg_entity_count": round(statistics.mean(skeletal_entities), 2) if skeletal_entities else 0.0,
        "peripheral_avg_entity_count": round(statistics.mean(peripheral_entities), 2) if peripheral_entities else 0.0,
    }


def compare_file(file_path: str) -> dict[str, object]:
    document = load_document(file_path)
    legacy_chunks = chunk_document(
        document,
        hierarchical=False,
        child_chunk_size=1000,
        child_chunk_overlap=200,
    )
    graph_chunks = chunk_document_for_graph(document)
    return {
        "file": str(Path(file_path).resolve()),
        "legacy": {
            **_summarize_chunks(legacy_chunks),
            **_skeleton_metrics(legacy_chunks),
        },
        "graph": {
            **_summarize_chunks(graph_chunks),
            **_skeleton_metrics(graph_chunks),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare graph chunking strategies")
    parser.add_argument("path", help="File to analyze")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    report = compare_file(args.path)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print(f"File: {report['file']}")
    for mode_name in ("legacy", "graph"):
        metrics = report[mode_name]
        print(f"\n[{mode_name}]")
        for key, value in metrics.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
