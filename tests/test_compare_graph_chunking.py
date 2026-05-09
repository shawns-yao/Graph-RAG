"""Tests for graph chunking comparison script."""

from pathlib import Path

from scripts.compare_graph_chunking import compare_file


def test_compare_file_reports_legacy_and_graph_metrics(tmp_path: Path):
    file_path = tmp_path / "graph_doc.md"
    file_path.write_text(
        "# Intro\n\nNeo4j GraphRAG FastAPI PageRank.\n\n"
        "## Data\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
        "Graph traversal connects Entity nodes with Passage nodes.",
        encoding="utf-8",
    )

    report = compare_file(str(file_path))
    assert report["legacy"]["chunks"] >= 1
    assert report["graph"]["chunks"] >= 1
    assert "skeletal_chunks" in report["legacy"]
    assert "skeletal_chunks" in report["graph"]
