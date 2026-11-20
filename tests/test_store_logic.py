"""Unit tests for upsert skip / re-embed decisions (no database)."""

from __future__ import annotations

from agentic_rag_platform.ingest.hashing import content_hash
from agentic_rag_platform.ingest.types import StoredChunkRef, needs_embedding


def _ref(content: str, model: str = "intfloat/multilingual-e5-base") -> StoredChunkRef:
    return StoredChunkRef(
        tenant_id="default",
        doc_id="eb202501",
        chunk_no=0,
        content_hash=content_hash(content),
        embedding_model=model,
    )


def test_new_chunk_needs_embedding() -> None:
    assert needs_embedding(None, content_hash=content_hash("hello"), embedding_model="e5")


def test_unchanged_text_and_model_skips_embedding() -> None:
    text = "Euro area inflation is close to the 2% medium-term target."
    existing = _ref(text)
    assert not needs_embedding(
        existing,
        content_hash=content_hash(text),
        embedding_model=existing.embedding_model,
    )


def test_whitespace_only_change_is_treated_as_same_content() -> None:
    existing = _ref("The rate is 2%")
    assert not needs_embedding(
        existing,
        content_hash=content_hash("The   rate" + chr(9) + "is 2%"),
        embedding_model=existing.embedding_model,
    )


def test_text_change_requires_reembed() -> None:
    existing = _ref("The rate is 2%")
    assert needs_embedding(
        existing,
        content_hash=content_hash("The rate is 3%"),
        embedding_model=existing.embedding_model,
    )


def test_model_change_requires_reembed_even_if_text_is_the_same() -> None:
    text = "The rate is 2%"
    existing = _ref(text, model="intfloat/multilingual-e5-base")
    assert needs_embedding(
        existing,
        content_hash=content_hash(text),
        embedding_model="BAAI/bge-m3",
    )
