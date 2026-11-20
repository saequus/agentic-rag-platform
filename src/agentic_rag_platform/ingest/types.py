"""Shared dataclasses for the Day 2 ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PdfPage:
    """One page of a parsed PDF. ``page`` is 1-based."""

    page: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    source_path: str
    title: str | None
    pages: tuple[PdfPage, ...]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_no: int
    page: int | None
    section: str | None
    text: str
    content_hash: str


@dataclass(frozen=True)
class StoredChunkRef:
    """The columns we need to decide whether a chunk must be re-embedded."""

    tenant_id: str
    doc_id: str
    chunk_no: int
    content_hash: str
    embedding_model: str


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    chunk_no: int
    page: int | None
    section: str | None
    text: str
    content_hash: str
    score: float


def needs_embedding(
    existing: StoredChunkRef | None,
    *,
    content_hash: str,
    embedding_model: str,
) -> bool:
    """Return True if the chunk is new or its text or embedding model changed."""
    if existing is None:
        return True
    return existing.content_hash != content_hash or existing.embedding_model != embedding_model
