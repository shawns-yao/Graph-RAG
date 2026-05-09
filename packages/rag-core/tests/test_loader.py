"""Tests for rag_core.loader."""

from pathlib import Path

import pytest
import rag_core.loader as loader_module
from rag_core.loader import (
    SUPPORTED_EXTENSIONS,
    DoclingLoader,
    DocumentResult,
    load_document,
    load_file,
)


class TestDocumentResult:
    def test_defaults(self):
        r = DocumentResult(markdown="hello")
        assert r.markdown == "hello"
        assert r.tables == []
        assert r.images == []
        assert r.formulas == []
        assert r.metadata == {}


class TestDoclingLoader:
    def test_init_defaults(self):
        loader = DoclingLoader()
        assert loader._use_gpu is False
        assert loader._converter is None
        assert loader._enable_image_ocr is False
        assert loader._enable_formula_ocr is False
        assert loader._pdf_table_backend == "none"

    def test_init_with_gpu(self):
        loader = DoclingLoader(use_gpu=True)
        assert loader._use_gpu is True

    def test_init_rejects_unknown_pdf_table_backend(self):
        with pytest.raises(ValueError, match="pdf_table_backend"):
            DoclingLoader(pdf_table_backend="auto")

    def test_load_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        loader = DoclingLoader()
        result = loader.load(str(f))
        assert result.markdown == "Hello, world!"
        assert result.blocks[0].text == "Hello, world!"
        assert result.metadata["format"] == ".txt"
        assert result.metadata["pages"] == 1

    def test_load_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nParagraph", encoding="utf-8")
        loader = DoclingLoader()
        result = loader.load(str(f))
        assert "# Title" in result.markdown
        assert result.blocks[0].heading_path == ["Title"]
        assert result.blocks[0].text == "Paragraph"

    def test_load_md_extracts_formulas(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("Equation: $$E=mc^2$$", encoding="utf-8")
        loader = DoclingLoader()
        result = loader.load(str(f))
        assert result.formulas == [{"latex": "E=mc^2", "source": "markdown"}]
        assert "## Extracted Formulas" in result.markdown

    def test_load_file_not_found(self):
        loader = DoclingLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.txt")

    def test_load_unsupported_format(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("data")
        loader = DoclingLoader()
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load(str(f))

    def test_load_bytes_txt(self):
        loader = DoclingLoader()
        result = loader.load_bytes(b"raw text data", "doc.txt")
        assert result.markdown == "raw text data"

    def test_load_bytes_md_extracts_formulas(self):
        loader = DoclingLoader()
        result = loader.load_bytes(b"Inline formula: \\(a+b\\)", "doc.md")
        assert result.formulas == [{"latex": "a+b", "source": "markdown"}]

    def test_load_bytes_unsupported(self):
        loader = DoclingLoader()
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load_bytes(b"data", "file.abc")

    def test_load_augments_markdown_with_multimodal_sections(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4")
        loader = DoclingLoader()

        fake_doc = _FakeDoc("Base markdown")
        fake_result = type("FakeResult", (), {"document": fake_doc})()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(loader, "_get_converter", lambda: _FakeConverter(fake_result))
            mp.setattr(
                loader,
                "_extract_tables",
                lambda doc, path: [{
                    "caption": "Revenue",
                    "markdown": "|A|\n|1|",
                    "csv": "A\n1",
                    "page": 1,
                    "source": "camelot",
                }],
            )
            mp.setattr(
                loader,
                "_extract_images",
                lambda doc: [{
                    "caption": "Chart",
                    "page": 2,
                    "ocr_text": "trend up",
                    "formula_latex": "",
                }],
            )
            result = loader.load(str(f))

        assert "## Extracted Tables" in result.markdown
        assert "## Extracted Images" in result.markdown
        assert result.metadata["tables_count"] == 1
        assert result.metadata["images_count"] == 1
        assert result.metadata["multimodal_augmented"] is True

    def test_extract_formulas_uses_image_formulas(self):
        formulas = DoclingLoader._extract_formulas("text only", image_formulas=["x^2 + y^2"])
        assert formulas == [{"latex": "x^2 + y^2", "source": "image_ocr"}]

    def test_merge_tables_deduplicates(self):
        primary = [{"page": 1, "markdown": "|A|", "caption": "t1"}]
        secondary = [
            {"page": 1, "markdown": "|A|", "caption": "dup"},
            {"page": 2, "markdown": "|B|", "caption": "t2"},
        ]
        merged = DoclingLoader._merge_tables(primary, secondary)
        assert len(merged) == 2

    def test_extract_tables_skips_pdf_fallback_by_default(self):
        loader = DoclingLoader()
        fake_doc = _FakeDoc("Base markdown")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                loader,
                "_extract_tables_with_pdf_tools",
                lambda _path: pytest.fail("pdf fallback should be opt-in"),
            )
            tables = loader._extract_tables(fake_doc, Path("report.pdf"))

        assert tables == []

    def test_extract_tables_uses_selected_pdf_backend(self):
        loader = DoclingLoader(pdf_table_backend="camelot")
        fake_doc = _FakeDoc("Base markdown")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                loader,
                "_extract_tables_with_camelot",
                lambda _path: [{
                    "caption": "Revenue",
                    "markdown": "|A|",
                    "csv": "A",
                    "page": 1,
                    "source": "camelot",
                }],
            )
            tables = loader._extract_tables(fake_doc, Path("report.pdf"))

        assert len(tables) == 1
        assert tables[0]["source"] == "camelot"

    def test_extract_images_uses_thread_pool_for_multiple_payloads(self):
        loader = DoclingLoader(enable_image_ocr=True, image_workers=2)
        fake_doc = _FakeDoc("Base markdown", items=[_FakeImageItem("Chart 1"), _FakeImageItem("Chart 2")])

        calls: list[int] = []

        class _FakeExecutor:
            def __init__(self, *, max_workers):
                calls.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, fn, payloads):
                return [fn(payload) for payload in payloads]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(loader_module, "ThreadPoolExecutor", _FakeExecutor)
            mp.setattr(loader, "_ocr_image", lambda image: f"text:{image}")
            mp.setattr(loader, "_ocr_formula", lambda image: "")
            images = loader._extract_images(fake_doc)

        assert calls == [2]
        assert [image["ocr_text"] for image in images] == ["text:Chart 1", "text:Chart 2"]


class TestLoadFile:
    def test_load_txt(self, tmp_path):
        f = tmp_path / "simple.txt"
        f.write_text("simple content", encoding="utf-8")
        text = load_file(str(f))
        assert text == "simple content"

    def test_load_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_file("/no/such/file.txt")

    def test_load_document_preserves_blocks(self, tmp_path):
        f = tmp_path / "structured.md"
        f.write_text("# Intro\n\n- a\n- b\n- c\n- d", encoding="utf-8")
        document = load_document(str(f))
        assert document.blocks
        assert any(block.block_type == "list_item" for block in document.blocks)


class TestSupportedExtensions:
    def test_contains_pdf(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_contains_docx(self):
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_contains_txt(self):
        assert ".txt" in SUPPORTED_EXTENSIONS


class _FakeDoc:
    def __init__(self, markdown: str, items=None):
        self._markdown = markdown
        self.num_pages = 3
        self._items = items or []

    def export_to_markdown(self):
        return self._markdown

    def iterate_items(self):
        return iter((item, 0) for item in self._items)


class _FakeConverter:
    def __init__(self, result):
        self._result = result

    def convert(self, _path):
        return self._result


class _FakeImageItem:
    def __init__(self, caption: str):
        self.caption = caption
        self.prov = [type("Prov", (), {"page_no": 1})()]

    def get_image(self, _doc):
        return self.caption
