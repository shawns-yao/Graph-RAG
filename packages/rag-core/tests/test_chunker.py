"""Tests for rag_core.chunker."""

from rag_core.chunker import (
    MAX_EPISODE_CHARS,
    chunk_document,
    chunk_document_for_graph,
    chunk_text,
    sanitize_for_graphiti,
    split_large_content,
)
from rag_core.loader import DocumentResult
from rag_core.models import DocumentBlock


class TestSanitizeForGraphiti:
    def test_removes_lucene_chars(self):
        assert sanitize_for_graphiti("hello/world") == "hello world"
        assert sanitize_for_graphiti("a*b?c") == "a b c"

    def test_preserves_normal_text(self):
        assert sanitize_for_graphiti("hello world") == "hello world"

    def test_removes_brackets(self):
        result = sanitize_for_graphiti("array[0]{key}")
        assert "[" not in result
        assert "{" not in result


class TestSplitLargeContent:
    def test_short_text_no_split(self):
        parts = split_large_content("short", "src")
        assert parts == [("short", "src")]

    def test_splits_at_paragraphs(self):
        text = ("A" * 5000) + "\n\n" + ("B" * 5000)
        parts = split_large_content(text, "doc", max_chars=6000)
        assert len(parts) >= 2
        assert all(name.startswith("doc_part_") for _, name in parts)

    def test_max_episode_chars_default(self):
        assert MAX_EPISODE_CHARS == 8_000


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world", chunk_size=1000, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"
        assert chunks[0].id != ""

    def test_chunk_has_id(self):
        chunks = chunk_text("Some text here", chunk_size=1000, chunk_overlap=0)
        assert len(chunks[0].id) == 8  # md5[:8]

    def test_paragraph_splitting(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=0)
        assert len(chunks) >= 2

    def test_header_splitting(self):
        text = "## Section A\n\nContent A\n\n## Section B\n\nContent B"
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=0)
        # Should split by headers
        assert len(chunks) >= 2
        titles = [c.metadata.get("section_title", "") for c in chunks]
        assert "Section A" in titles
        assert "Section B" in titles

    def test_table_kept_atomic(self):
        table = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
        chunks = chunk_text(table, chunk_size=10, chunk_overlap=0)
        # Table should be one chunk regardless of size
        assert len(chunks) == 1

    def test_chunk_index_in_metadata(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=0)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap(self):
        text = ("A" * 100) + "\n\n" + ("B" * 100) + "\n\n" + ("C" * 100)
        chunks = chunk_text(text, chunk_size=120, chunk_overlap=20)
        assert len(chunks) >= 2


class TestStructuredChunking:
    def test_chunk_document_falls_back_without_blocks(self):
        document = DocumentResult(markdown="hello world")
        chunks = chunk_document(document, child_chunk_size=100, child_chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].content == "hello world"

    def test_chunk_document_builds_parent_child_context(self):
        document = DocumentResult(
            markdown="ignored",
            blocks=[
                DocumentBlock(
                    block_type="paragraph",
                    text="Alpha " * 20,
                    heading_path=["Intro", "Overview"],
                    order_index=0,
                ),
                DocumentBlock(
                    block_type="paragraph",
                    text="Beta " * 20,
                    heading_path=["Intro", "Overview"],
                    order_index=1,
                ),
            ],
        )
        chunks = chunk_document(
            document,
            parent_chunk_size=400,
            child_chunk_size=80,
            child_chunk_overlap=10,
            context_chars=100,
            hierarchical=True,
        )
        assert len(chunks) >= 2
        assert all(chunk.metadata["parent_chunk_id"] for chunk in chunks)
        assert all(chunk.metadata["heading_path"] == ["Intro", "Overview"] for chunk in chunks)
        assert "Section: Intro > Overview" in chunks[0].context

    def test_chunk_document_keeps_tables_atomic(self):
        document = DocumentResult(
            markdown="ignored",
            blocks=[
                DocumentBlock(
                    block_type="table",
                    text="| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |",
                    heading_path=["Data"],
                    order_index=0,
                )
            ],
        )
        chunks = chunk_document(
            document,
            parent_chunk_size=50,
            child_chunk_size=10,
            child_chunk_overlap=0,
        )
        assert len(chunks) == 1
        assert chunks[0].content.startswith("| A | B |")

    def test_chunk_document_for_graph_skips_table_and_code(self):
        document = DocumentResult(
            markdown="X" * 7000,
            blocks=[
                DocumentBlock(
                    block_type="table",
                    text="| A | B |",
                    heading_path=["Data"],
                    order_index=0,
                ),
                DocumentBlock(
                    block_type="code",
                    text="print('x')",
                    heading_path=["Impl"],
                    order_index=1,
                ),
                DocumentBlock(
                    block_type="paragraph",
                    text="Neo4j connects GraphRAG with PageRank and FastAPI.",
                    heading_path=["Intro"],
                    order_index=2,
                ),
            ],
        )
        chunks = chunk_document_for_graph(document)
        assert len(chunks) == 1
        assert chunks[0].metadata["graph_chunk_type"] == "skeleton_candidate"
        assert chunks[0].metadata["block_type"] == "paragraph"

    def test_chunk_document_for_graph_marks_peripheral_when_entities_sparse(self):
        document = DocumentResult(
            markdown="X" * 7000,
            blocks=[
                DocumentBlock(
                    block_type="paragraph",
                    text="simple explanation without many anchors.",
                    heading_path=["Notes"],
                    order_index=0,
                ),
            ],
        )
        chunks = chunk_document_for_graph(document)
        assert len(chunks) == 1
        assert chunks[0].metadata["graph_chunk_type"] == "peripheral_candidate"

    def test_chunk_document_for_graph_respects_sentence_boundaries(self):
        document = DocumentResult(
            markdown="ignored",
            blocks=[
                DocumentBlock(
                    block_type="paragraph",
                    text=(
                        "Neo4j supports GraphRAG with FastAPI. "
                        "PageRank links Entity nodes with Passage nodes. "
                        "DeepSeek powers extraction."
                    ),
                    heading_path=["Graph"],
                    order_index=0,
                ),
            ],
        )
        chunks = chunk_document_for_graph(document)
        assert chunks
        assert all(not chunk.content.endswith("Fast") for chunk in chunks)

    def test_chunk_document_for_graph_uses_markdown_fallback_for_short_documents(self):
        markdown = (
            "## COPD\n\n"
            "COPD diagnosis requires spirometry after bronchodilator use. "
            "GOLD grading depends on FEV1 percentage predicted.\n\n"
            "Treatment selection depends on exacerbation risk and eosinophils."
        )
        document = DocumentResult(
            markdown=markdown,
            blocks=[
                DocumentBlock(
                    block_type="paragraph",
                    text="COPD diagnosis requires spirometry after bronchodilator use.",
                    heading_path=["COPD"],
                    order_index=0,
                ),
                DocumentBlock(
                    block_type="paragraph",
                    text="GOLD grading depends on FEV1 percentage predicted.",
                    heading_path=["COPD"],
                    order_index=1,
                ),
                DocumentBlock(
                    block_type="paragraph",
                    text="Treatment selection depends on exacerbation risk and eosinophils.",
                    heading_path=["COPD"],
                    order_index=2,
                ),
            ],
        )
        chunks = chunk_document_for_graph(document)
        assert len(chunks) == 1
        assert "COPD diagnosis requires spirometry" in chunks[0].content
        assert "graph_chunk_type" not in chunks[0].metadata
