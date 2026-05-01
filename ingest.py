"""
Step 1: Document Ingestion Module.

Parses PDF files locally using IBM Docling, preserving semantic layout
(headers, tables, code blocks, formulas-as-LaTeX), and converts the parsed
content into LangChain Document objects suitable for downstream embedding.

No network calls. No external APIs. 100% offline.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
from langchain_core.documents import Document

# HybridChunker is the production-recommended chunker: it builds on the
# hierarchical (layout-aware) split and then merges small adjacent chunks
# up to a token budget, so each chunk carries enough context for retrieval.
try:
    from docling_core.transforms.chunker import HybridChunker
    _HAS_HYBRID = True
except ImportError:  # pragma: no cover
    HybridChunker = None  # type: ignore
    _HAS_HYBRID = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("./data")
# Docling auto-detects format from extension. PDFs go through the layout
# pipeline; image suffixes go through the image pipeline which OCRs the
# page via RapidOCR (already cached locally as a Docling dep). The chunker
# downstream is format-agnostic - it just sees text + structure.
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES


@dataclass
class IngestionResult:
    """Container for the output of an ingestion run."""

    source_path: Path
    markdown: str
    documents: List[Document]

    @property
    def chunk_count(self) -> int:
        return len(self.documents)


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------
class PDFIngestor:
    """
    Wraps Docling's DocumentConverter and HierarchicalChunker to produce
    layout-aware LangChain Documents.

    The HierarchicalChunker splits on the document's semantic structure
    (sections, tables, code blocks, list groups) rather than fixed character
    counts, which preserves the meaning required for high-precision RAG.
    """

    def __init__(self, max_tokens: int = 512) -> None:
        """
        Parameters
        ----------
        max_tokens : int
            Token budget per chunk. The HybridChunker will merge small
            adjacent layout chunks up to this limit so each chunk carries
            enough context for high-precision retrieval. Falls back to the
            plain HierarchicalChunker if HybridChunker is unavailable.
        """
        try:
            self._converter = DocumentConverter()
            if _HAS_HYBRID:
                # tokenizer=None makes HybridChunker use a simple character
                # heuristic (~4 chars per token) instead of pulling a real
                # HF tokenizer. Avoids an extra model download for ingestion.
                self._chunker = HybridChunker(
                    tokenizer=None,
                    max_tokens=max_tokens,
                    merge_peers=True,
                )
                logger.info(
                    "Docling DocumentConverter + HybridChunker (max_tokens=%d) ready.",
                    max_tokens,
                )
            else:
                self._chunker = HierarchicalChunker()
                logger.info(
                    "Docling DocumentConverter + HierarchicalChunker (fallback) ready."
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to initialize Docling converter.")
            raise RuntimeError(
                "Could not initialize Docling. Verify the docling install."
            ) from exc

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _validate_path(src_path: Path) -> Path:
        src_path = Path(src_path).expanduser().resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"File not found: {src_path}")
        if src_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported file type '{src_path.suffix}'. "
                f"Expected one of {sorted(SUPPORTED_SUFFIXES)}."
            )
        return src_path

    @staticmethod
    def _is_image(path: Path) -> bool:
        return path.suffix.lower() in IMAGE_SUFFIXES

    @staticmethod
    def _clean_text(text: str) -> str:
        # Collapse excessive blank lines while keeping paragraph breaks.
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    # --------------------------------------------------------------- main API
    def ingest(self, src_path: str | Path) -> IngestionResult:
        """
        Convert a single PDF or document image (JPG/PNG/TIFF/BMP) into
        Markdown + a list of LangChain Documents. Images go through
        Docling's image pipeline which runs OCR via RapidOCR.

        Returns
        -------
        IngestionResult
            Populated with markdown and chunked Documents.
        """
        src_path = self._validate_path(Path(src_path))
        kind = "image (OCR)" if self._is_image(src_path) else "PDF"
        logger.info("Parsing %s with Docling: %s", kind, src_path.name)

        try:
            conversion = self._converter.convert(str(src_path))
        except Exception as exc:
            logger.exception("Docling failed to parse %s", src_path)
            raise RuntimeError(f"Docling parse error for {src_path}") from exc

        docling_doc = conversion.document

        # Markdown preserves tables as pipe-tables and formulas as LaTeX,
        # which is ideal for the LLM to reason over downstream. For images
        # the markdown is the OCR text, structured by Docling's layout
        # detector (so single-column scans become paragraphs, multi-column
        # become tables, etc.).
        try:
            markdown = docling_doc.export_to_markdown()
        except Exception:
            logger.warning("Markdown export failed; falling back to text.")
            markdown = docling_doc.export_to_text()

        markdown = self._clean_text(markdown)

        # Layout-aware chunking using Docling's hierarchical chunker.
        documents = self._chunk_to_documents(docling_doc, source=src_path)

        logger.info(
            "Extracted %d chunks from %s (markdown length: %d chars)",
            len(documents),
            src_path.name,
            len(markdown),
        )

        return IngestionResult(
            source_path=src_path,
            markdown=markdown,
            documents=documents,
        )

    def ingest_many(self, paths: Iterable[str | Path]) -> List[IngestionResult]:
        results: List[IngestionResult] = []
        for p in paths:
            try:
                results.append(self.ingest(p))
            except Exception as exc:
                logger.error("Skipping %s due to error: %s", p, exc)
        return results

    def ingest_directory(
        self,
        directory: str | Path = DEFAULT_DATA_DIR,
        recursive: bool = True,
    ) -> List[IngestionResult]:
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Glob every supported extension (case-insensitive on Windows).
        files: List[Path] = []
        for suffix in SUPPORTED_SUFFIXES:
            pattern = f"**/*{suffix}" if recursive else f"*{suffix}"
            files.extend(directory.glob(pattern))
        files = sorted(set(files))
        logger.info("Found %d ingestible file(s) under %s", len(files), directory)
        return self.ingest_many(files)

    # ----------------------------------------------------------------- chunks
    def _chunk_to_documents(
        self, docling_doc, source: Path
    ) -> List[Document]:
        """
        Use Docling's HierarchicalChunker to split the document along its
        natural structure, then wrap each chunk as a LangChain Document with
        rich metadata (page numbers, headings, content type).
        """
        documents: List[Document] = []

        try:
            raw_chunks = list(self._chunker.chunk(docling_doc))
        except Exception as exc:
            logger.exception("Hierarchical chunking failed.")
            raise RuntimeError("Failed to chunk Docling document.") from exc

        for idx, chunk in enumerate(raw_chunks):
            text = getattr(chunk, "text", None) or str(chunk)
            text = self._clean_text(text)
            if not text:
                continue

            metadata = {
                "source": str(source),
                "filename": source.name,
                "chunk_index": idx,
            }

            # Pull headings and page references when available.
            meta = getattr(chunk, "meta", None)
            if meta is not None:
                headings = getattr(meta, "headings", None)
                if headings:
                    metadata["headings"] = " > ".join(map(str, headings))

                doc_items = getattr(meta, "doc_items", None) or []
                pages: list[int] = []
                labels: list[str] = []
                for item in doc_items:
                    label = getattr(item, "label", None)
                    if label and str(label) not in labels:
                        labels.append(str(label))
                    for prov in getattr(item, "prov", []) or []:
                        page = getattr(prov, "page_no", None)
                        if page is not None and page not in pages:
                            pages.append(page)
                if pages:
                    metadata["pages"] = sorted(pages)
                    metadata["page"] = pages[0]
                if labels:
                    metadata["content_types"] = ",".join(labels)

            documents.append(Document(page_content=text, metadata=metadata))

        return documents


# ---------------------------------------------------------------------------
# CLI entry point — handy for ad-hoc testing
# ---------------------------------------------------------------------------
def _print_summary(results: List[IngestionResult]) -> None:
    total_chunks = sum(r.chunk_count for r in results)
    print(f"\nIngested {len(results)} document(s), {total_chunks} chunks total.")
    for r in results:
        print(f"  - {r.source_path.name}: {r.chunk_count} chunks")


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse PDFs with Docling into LangChain Documents."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="One or more PDF files. If omitted, ingests ./data/*.pdf.",
    )
    args = parser.parse_args(argv)

    ingestor = PDFIngestor()

    if args.paths:
        results = ingestor.ingest_many(args.paths)
    else:
        results = ingestor.ingest_directory(DEFAULT_DATA_DIR)

    _print_summary(results)


if __name__ == "__main__":
    main()
