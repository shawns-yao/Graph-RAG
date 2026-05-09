"""Document loader with optional multimodal extraction helpers."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_core.models import DocumentBlock

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_RE = re.compile(r"^\s*\|.+\|\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+).+")

_FORMULA_PATTERNS = (
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
)


@dataclass
class DocumentResult:
    """Result of document processing via Docling."""

    markdown: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    formulas: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[DocumentBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DoclingLoader:
    """Document loader with lazy Docling initialization and optional multimodal enrichments."""

    def __init__(
        self,
        use_gpu: bool = False,
        *,
        enable_image_ocr: bool = False,
        enable_formula_ocr: bool = False,
        pdf_table_backend: str = "none",
        image_workers: int = 4,
    ) -> None:
        backend = pdf_table_backend.strip().lower()
        if backend not in {"none", "camelot", "tabula"}:
            raise ValueError(
                "pdf_table_backend must be one of: none, camelot, tabula"
            )

        self._converter: DocumentConverter | None = None
        self._use_gpu = use_gpu
        self._enable_image_ocr = enable_image_ocr
        self._enable_formula_ocr = enable_formula_ocr
        self._pdf_table_backend = backend
        self._image_workers = max(1, image_workers)
        self._ocr_reader: Any = None
        self._formula_ocr_reader: Any = None

    def _get_converter(self) -> DocumentConverter:
        """Lazy-initialize Docling converter."""
        if self._converter is None:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_picture_images = True

            if self._use_gpu:
                try:
                    from docling.datamodel.accelerator_options import (
                        AcceleratorDevice,
                        AcceleratorOptions,
                    )

                    pipeline_options.accelerator_options = AcceleratorOptions(
                        device=AcceleratorDevice.AUTO
                    )
                    logger.info("GPU acceleration enabled for PDF")
                except ImportError:
                    logger.warning("GPU acceleration imports failed, falling back to CPU")

            from docling.datamodel.base_models import InputFormat

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        return self._converter

    def load(self, file_path: str | Path) -> DocumentResult:
        """Load a document and extract content with tables, images, and formulas."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if suffix in {".txt", ".md"}:
            markdown = path.read_text(encoding="utf-8")
            formulas = self._extract_formulas(markdown)
            markdown = self._augment_markdown(markdown, tables=[], images=[], formulas=formulas)
            return DocumentResult(
                markdown=markdown,
                formulas=formulas,
                blocks=self._build_structured_blocks(markdown),
                metadata={
                    "format": path.suffix,
                    "pages": 1,
                    "tables_count": 0,
                    "images_count": 0,
                    "formulas_count": len(formulas),
                },
            )

        converter = self._get_converter()
        result = converter.convert(str(path))
        doc = result.document

        tables = self._extract_tables(doc, path)
        images = self._extract_images(doc)
        raw_markdown = doc.export_to_markdown()
        formulas = self._extract_formulas(
            raw_markdown,
            image_formulas=[img.get("formula_latex", "") for img in images],
        )
        markdown = self._augment_markdown(raw_markdown, tables=tables, images=images, formulas=formulas)
        blocks = self._build_structured_blocks(markdown)

        pages = getattr(doc, "num_pages", None)
        if callable(pages):
            pages = pages()

        metadata = {
            "format": path.suffix,
            "pages": pages,
            "tables_count": len(tables),
            "images_count": len(images),
            "formulas_count": len(formulas),
            "multimodal_augmented": bool(tables or images or formulas),
        }

        logger.info(
            "Loaded %d chars from %s (%d tables, %d images, %d formulas)",
            len(markdown),
            path.name,
            len(tables),
            len(images),
            len(formulas),
        )

        return DocumentResult(
            markdown=markdown,
            tables=tables,
            images=images,
            formulas=formulas,
            blocks=blocks,
            metadata=metadata,
        )

    def load_bytes(self, data: bytes, filename: str) -> DocumentResult:
        """Load a document from bytes (for file upload handlers)."""
        import os
        import tempfile

        suffix = Path(filename).suffix
        if suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {suffix}")

        if suffix.lower() in {".txt", ".md"}:
            markdown = data.decode("utf-8", errors="replace")
            formulas = self._extract_formulas(markdown)
            markdown = self._augment_markdown(markdown, tables=[], images=[], formulas=formulas)
            return DocumentResult(
                markdown=markdown,
                formulas=formulas,
                blocks=self._build_structured_blocks(markdown),
                metadata={
                    "format": suffix,
                    "pages": 1,
                    "tables_count": 0,
                    "images_count": 0,
                    "formulas_count": len(formulas),
                },
            )

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        path = Path(tmp_path)
        try:
            os.close(fd)
            path.write_bytes(data)
            return self.load(path)
        finally:
            path.unlink(missing_ok=True)

    def _get_ocr_reader(self) -> Any | None:
        """Lazily initialize the configured OCR backend."""
        if not self._enable_image_ocr:
            return None
        if self._ocr_reader is not None:
            return self._ocr_reader

        try:
            import easyocr

            self._ocr_reader = easyocr.Reader(["en"], gpu=self._use_gpu)
            return self._ocr_reader
        except Exception as exc:
            logger.debug("EasyOCR unavailable: %s", exc)
        return None

    def _get_formula_ocr_reader(self) -> Any | None:
        """Lazily initialize the configured formula OCR backend."""
        if not self._enable_formula_ocr:
            return None
        if self._formula_ocr_reader is not None:
            return self._formula_ocr_reader

        try:
            from pix2tex.cli import LatexOCR

            self._formula_ocr_reader = LatexOCR()
            return self._formula_ocr_reader
        except Exception as exc:
            logger.debug("Formula OCR unavailable: %s", exc)
        return None

    def _ocr_image(self, image: object) -> str:
        """Run optional OCR over an extracted image."""
        reader = self._get_ocr_reader()
        if reader is None:
            return ""

        try:
            results = reader.readtext(image, detail=0)
            return " ".join(part.strip() for part in results if str(part).strip())
        except Exception as exc:
            logger.debug("Image OCR skipped: %s", exc)
            return ""

    def _ocr_formula(self, image: object) -> str:
        """Run optional formula OCR when explicitly enabled."""
        predictor = self._get_formula_ocr_reader()
        if predictor is None:
            return ""
        try:
            return str(predictor(image)).strip()
        except Exception as exc:
            logger.debug("Formula OCR skipped: %s", exc)
            return ""

    def _extract_tables(self, doc: object, source_path: Path | None = None) -> list[dict[str, Any]]:
        """Extract tables using Docling first, then optional PDF-specific fallbacks."""
        tables: list[dict[str, Any]] = []
        for item, _level in doc.iterate_items():  # type: ignore[attr-defined]
            if hasattr(item, "export_to_dataframe"):
                try:
                    df = item.export_to_dataframe()
                    page_num = self._page_number(item)
                    tables.append({
                        "caption": getattr(item, "caption", "") or "",
                        "markdown": df.to_markdown(index=False),
                        "csv": df.to_csv(index=False),
                        "page": page_num,
                        "source": "docling",
                    })
                except Exception as exc:
                    logger.debug("Docling table extraction skipped: %s", exc)

        if (
            source_path
            and source_path.suffix.lower() == ".pdf"
            and self._pdf_table_backend != "none"
        ):
            tables = self._merge_tables(tables, self._extract_tables_with_pdf_tools(source_path))
        return tables

    def _extract_tables_with_pdf_tools(self, source_path: Path) -> list[dict[str, Any]]:
        """Run the explicitly selected PDF table backend."""
        if self._pdf_table_backend == "camelot":
            return self._extract_tables_with_camelot(source_path)
        if self._pdf_table_backend == "tabula":
            return self._extract_tables_with_tabula(source_path)
        return []

    @staticmethod
    def _extract_tables_with_camelot(source_path: Path) -> list[dict[str, Any]]:
        """Extract PDF tables with Camelot when the dependency is installed."""
        extracted: list[dict[str, Any]] = []
        try:
            import camelot  # type: ignore

            camelot_tables = camelot.read_pdf(str(source_path), pages="all")
            for idx, table in enumerate(camelot_tables, start=1):
                df = table.df
                extracted.append({
                    "caption": f"camelot_table_{idx}",
                    "markdown": df.to_markdown(index=False),
                    "csv": df.to_csv(index=False),
                    "page": getattr(table, "page", None),
                    "source": "camelot",
                })
        except Exception as exc:
            logger.debug("Camelot extraction skipped: %s", exc)
        return extracted

    @staticmethod
    def _extract_tables_with_tabula(source_path: Path) -> list[dict[str, Any]]:
        """Extract PDF tables with Tabula when the dependency is installed."""
        extracted: list[dict[str, Any]] = []

        try:
            import tabula  # type: ignore

            dfs = tabula.read_pdf(str(source_path), pages="all", multiple_tables=True)
            for idx, df in enumerate(dfs, start=1):
                extracted.append({
                    "caption": f"tabula_table_{idx}",
                    "markdown": df.to_markdown(index=False),
                    "csv": df.to_csv(index=False),
                    "page": None,
                    "source": "tabula",
                })
        except Exception as exc:
            logger.debug("Tabula extraction skipped: %s", exc)

        return extracted

    @staticmethod
    def _merge_tables(
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge table lists while deduplicating by page+markdown."""
        merged = list(primary)
        seen = {(tbl.get("page"), tbl.get("markdown", "")) for tbl in primary}
        for table in secondary:
            key = (table.get("page"), table.get("markdown", ""))
            if key not in seen:
                seen.add(key)
                merged.append(table)
        return merged

    def _extract_images(self, doc: object) -> list[dict[str, Any]]:
        """Extract image metadata plus optional OCR / formula OCR."""
        payloads: list[dict[str, Any]] = []
        for item, _level in doc.iterate_items():  # type: ignore[attr-defined]
            if hasattr(item, "get_image"):
                try:
                    image = item.get_image(doc)
                    if image:
                        payloads.append({
                            "caption": getattr(item, "caption", "") or "",
                            "page": self._page_number(item),
                            "image": image,
                        })
                except Exception as exc:
                    logger.debug("Image extraction skipped: %s", exc)

        if not payloads:
            return []

        if (
            len(payloads) == 1
            or self._image_workers == 1
            or not (self._enable_image_ocr or self._enable_formula_ocr)
        ):
            return [self._process_image_payload(payload) for payload in payloads]

        max_workers = min(self._image_workers, len(payloads))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._process_image_payload, payloads))

    def _process_image_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Process one extracted image with optional OCR backends."""
        image = payload["image"]
        return {
            "caption": payload["caption"],
            "page": payload["page"],
            "ocr_text": self._ocr_image(image),
            "formula_latex": self._ocr_formula(image),
        }

    @staticmethod
    def _extract_formulas(markdown: str, image_formulas: list[str] | None = None) -> list[dict[str, Any]]:
        """Extract formulas from markdown and optional OCR-derived LaTeX."""
        formulas: list[dict[str, Any]] = []
        seen: set[str] = set()

        for pattern in _FORMULA_PATTERNS:
            for match in pattern.finditer(markdown):
                latex = match.group(1).strip()
                if latex and latex not in seen:
                    seen.add(latex)
                    formulas.append({"latex": latex, "source": "markdown"})

        for latex in image_formulas or []:
            cleaned = str(latex).strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                formulas.append({"latex": cleaned, "source": "image_ocr"})

        return formulas

    @staticmethod
    def _augment_markdown(
        markdown: str,
        *,
        tables: list[dict[str, Any]],
        images: list[dict[str, Any]],
        formulas: list[dict[str, Any]],
    ) -> str:
        """Append multimodal extractions to markdown for downstream text-only pipelines."""
        sections: list[str] = [markdown.strip()]

        if tables:
            table_lines = ["## Extracted Tables"]
            for idx, table in enumerate(tables, start=1):
                page = table.get("page")
                caption = table.get("caption") or f"Table {idx}"
                source = table.get("source", "unknown")
                table_lines.append(f"### Table {idx} ({source}, page={page})")
                table_lines.append(str(caption))
                table_lines.append(str(table.get("markdown", "")).strip())
            sections.append("\n\n".join(line for line in table_lines if line.strip()))

        if images:
            image_lines = ["## Extracted Images"]
            for idx, image in enumerate(images, start=1):
                page = image.get("page")
                caption = image.get("caption") or f"Image {idx}"
                image_lines.append(f"### Image {idx} (page={page})")
                image_lines.append(str(caption))
                if image.get("ocr_text"):
                    image_lines.append(f"OCR: {image['ocr_text']}")
                if image.get("formula_latex"):
                    image_lines.append(f"Formula: {image['formula_latex']}")
            sections.append("\n\n".join(line for line in image_lines if line.strip()))

        if formulas:
            formula_lines = ["## Extracted Formulas"]
            for idx, formula in enumerate(formulas, start=1):
                formula_lines.append(f"### Formula {idx} ({formula.get('source', 'unknown')})")
                formula_lines.append(str(formula.get("latex", "")).strip())
            sections.append("\n\n".join(line for line in formula_lines if line.strip()))

        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _page_number(item: object) -> int | None:
        """Get source page number when available."""
        if hasattr(item, "prov") and item.prov:
            return getattr(item.prov[0], "page_no", None)
        return None

    @classmethod
    def _build_structured_blocks(cls, markdown: str) -> list[DocumentBlock]:
        """Parse markdown into structure-aware blocks for downstream chunking."""
        if not markdown.strip():
            return []

        lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        blocks: list[DocumentBlock] = []
        heading_path: list[str] = []
        paragraph_lines: list[str] = []
        order_index = 0
        cursor = 0

        def emit(block_type: str, text: str, metadata: dict[str, Any] | None = None) -> None:
            nonlocal order_index
            payload = text.strip()
            if not payload:
                return
            blocks.append(
                DocumentBlock(
                    block_type=block_type,
                    text=payload,
                    heading_path=list(heading_path),
                    order_index=order_index,
                    metadata=metadata or {},
                )
            )
            order_index += 1

        def flush_paragraph() -> None:
            if paragraph_lines:
                emit("paragraph", "\n".join(paragraph_lines))
                paragraph_lines.clear()

        while cursor < len(lines):
            line = lines[cursor]
            stripped = line.strip()

            if not stripped:
                flush_paragraph()
                cursor += 1
                continue

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                flush_paragraph()
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                heading_path = heading_path[: level - 1] + [title]
                cursor += 1
                continue

            if stripped.startswith("```"):
                flush_paragraph()
                code_lines = [line]
                cursor += 1
                while cursor < len(lines):
                    code_lines.append(lines[cursor])
                    if lines[cursor].strip().startswith("```"):
                        cursor += 1
                        break
                    cursor += 1
                emit("code", "\n".join(code_lines))
                continue

            if _TABLE_RE.match(line):
                flush_paragraph()
                table_lines = [line]
                cursor += 1
                while cursor < len(lines) and _TABLE_RE.match(lines[cursor]):
                    table_lines.append(lines[cursor])
                    cursor += 1
                emit("table", "\n".join(table_lines))
                continue

            if _LIST_RE.match(line):
                flush_paragraph()
                list_lines = [line]
                cursor += 1
                while cursor < len(lines) and _LIST_RE.match(lines[cursor]):
                    list_lines.append(lines[cursor])
                    cursor += 1
                if len(list_lines) <= 3:
                    emit("list", "\n".join(list_lines), {"items_count": len(list_lines)})
                else:
                    for item_index, item_text in enumerate(list_lines):
                        emit(
                            "list_item",
                            item_text,
                            {"items_count": len(list_lines), "item_index": item_index},
                        )
                continue

            paragraph_lines.append(line)
            cursor += 1

        flush_paragraph()
        return blocks


def load_file(file_path: str, use_gpu: bool = False) -> str:
    """Load document and return markdown text augmented with multimodal extracts."""
    loader = DoclingLoader(use_gpu=use_gpu)
    result = loader.load(file_path)
    return result.markdown


def load_document(file_path: str, use_gpu: bool = False) -> DocumentResult:
    """Load document and preserve both markdown and structure-aware blocks."""
    loader = DoclingLoader(use_gpu=use_gpu)
    return loader.load(file_path)
