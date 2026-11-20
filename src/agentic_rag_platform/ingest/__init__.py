"""Day 2 ingestion pipeline: PDFs → chunks → embeddings → pgvector."""

from __future__ import annotations

from agentic_rag_platform.ingest.hashing import content_hash
from agentic_rag_platform.ingest.types import Chunk, ParsedDocument, SearchHit, needs_embedding

__all__ = [
    "Chunk",
    "ParsedDocument",
    "SearchHit",
    "content_hash",
    "needs_embedding",
]
