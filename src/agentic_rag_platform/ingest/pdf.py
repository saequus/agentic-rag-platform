"""Load PDFs with pypdf, keeping per-page text and 1-based page numbers."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

from agentic_rag_platform.ingest.types import ParsedDocument, PdfPage


class PdfLoadError(ValueError):
    """The file could not be parsed as a readable, unencrypted PDF."""


def load_pdf(path: Path | str) -> ParsedDocument:
    source = Path(path)
    if not source.is_file():
        raise PdfLoadError(f"PDF not found: {source}")
    try:
        reader = PdfReader(str(source))
    except PdfReadError as exc:
        raise PdfLoadError(f"Invalid PDF: {source}: {exc}") from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except (FileNotDecryptedError, NotImplementedError, PdfReadError, ValueError) as exc:
            raise PdfLoadError(f"Encrypted PDF is not supported: {source}") from exc
        if not unlocked:
            raise PdfLoadError(f"Encrypted PDF is not supported: {source}")

    pages: list[PdfPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except (PdfReadError, ValueError) as exc:
            raise PdfLoadError(f"Failed to extract text from {source} page {index}: {exc}") from exc
        pages.append(PdfPage(page=index, text=text))

    return ParsedDocument(
        source_path=str(source.resolve()),
        title=_title(reader, source),
        pages=tuple(pages),
    )


def _title(reader: PdfReader, source: Path) -> str | None:
    meta = reader.metadata
    if meta is not None:
        raw = meta.title
        if raw and str(raw).strip():
            return str(raw).strip()
    return source.stem
