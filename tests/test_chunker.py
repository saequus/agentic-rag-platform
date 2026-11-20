"""Unit tests for the structure-aware chunker and content hash."""

from __future__ import annotations

from agentic_rag_platform.ingest.chunker import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_OVERLAP_RATIO,
    chunk_document,
    count_tokens,
    is_heading,
)
from agentic_rag_platform.ingest.hashing import content_hash, normalize_text
from agentic_rag_platform.ingest.types import ParsedDocument, PdfPage


def _doc(pages: list[tuple[int, str]], title: str = "Test") -> ParsedDocument:
    return ParsedDocument(
        source_path="memory://test.pdf",
        title=title,
        pages=tuple(PdfPage(page=page, text=text) for page, text in pages),
    )


def _filler(n_tokens: int, word: str = "inflation") -> str:
    """Build a string that tokenises to roughly ``n_tokens`` (cl100k_base)."""
    # A short English word plus a space is typically 2 tokens, but we measure.
    parts: list[str] = []
    while count_tokens(" ".join(parts)) < n_tokens:
        parts.append(word)
    # Trim if we overshot.
    while parts and count_tokens(" ".join(parts)) > n_tokens:
        parts.pop()
    return " ".join(parts)


def test_content_hash_is_stable_under_whitespace() -> None:
    left = content_hash("  The  rate" + chr(10) + chr(10) + "is  2% ")
    right = content_hash("The rate is 2%")
    assert left == right
    assert len(left) == 64
    assert left != content_hash("The rate is 3%")


def test_normalize_text_collapses_runs() -> None:
    messy = " a" + chr(9) + "  b " + chr(10) + " c "
    assert normalize_text(messy) == "a b c"


def test_short_document_is_a_single_chunk() -> None:
    text = "Narodowy Bank Polski published the inflation report."
    chunks = chunk_document(_doc([(1, text)]), doc_id="nbp-short")
    assert len(chunks) == 1
    assert chunks[0].doc_id == "nbp-short"
    assert chunks[0].chunk_no == 0
    assert chunks[0].page == 1
    assert "inflation report" in chunks[0].text
    assert chunks[0].content_hash == content_hash(chunks[0].text)


def test_headings_are_kept_as_section_metadata() -> None:
    blank = chr(10) + chr(10)
    pages = [
        (
            1,
            "1 External environment"
            + blank
            + _filler(200, "trade")
            + blank
            + "2 Economic activity"
            + blank
            + _filler(200, "growth"),
        )
    ]
    chunks = chunk_document(_doc(pages), doc_id="eb")
    assert chunks  # at least one
    sections = {chunk.section for chunk in chunks}
    assert "1 External environment" in sections
    assert "2 Economic activity" in sections
    for chunk in chunks:
        if chunk.section == "1 External environment":
            assert "trade" in chunk.text
        if chunk.section == "2 Economic activity":
            assert "growth" in chunk.text


def test_long_section_splits_within_token_bounds_with_overlap() -> None:
    body = _filler(2400, "projection")
    pages = [(1, "3 Prices and costs" + chr(10) + chr(10) + body)]
    chunks = chunk_document(
        _doc(pages),
        doc_id="long",
        max_tokens=DEFAULT_MAX_TOKENS,
        min_tokens=DEFAULT_MIN_TOKENS,
        overlap_ratio=DEFAULT_OVERLAP_RATIO,
    )
    assert len(chunks) >= 3
    # Every non-final chunk should sit in [min, max]; the final one may be
    # shorter, but the merger keeps it from being a tiny leftover when possible.
    for chunk in chunks[:-1]:
        tokens = count_tokens(chunk.text)
        assert tokens <= DEFAULT_MAX_TOKENS
        assert tokens >= DEFAULT_MIN_TOKENS
    assert count_tokens(chunks[-1].text) <= DEFAULT_MAX_TOKENS
    assert all(chunk.section == "3 Prices and costs" for chunk in chunks)
    assert [chunk.chunk_no for chunk in chunks] == list(range(len(chunks)))

    overlap = int(round(DEFAULT_MAX_TOKENS * DEFAULT_OVERLAP_RATIO))
    # The start of chunk i should share an overlap window with the end of i-1.
    for prev, current in zip(chunks, chunks[1:], strict=False):
        prev_tokens = count_tokens(prev.text)
        if prev_tokens < overlap:
            continue
        # A few tokens of the previous tail must appear in the next chunk.
        # We check a mid-length suffix of words to avoid token-boundary noise.
        tail_words = prev.text.split()[-12:]
        assert any(word in current.text for word in tail_words)


def test_is_heading_detects_numbered_and_all_caps() -> None:
    assert is_heading("1 External environment")
    assert is_heading("1.2 Financing conditions")
    assert is_heading("Rozdział 2 Projekcja inflacji")
    assert is_heading("LIST EMISYJNY NR 107/2026")
    assert not is_heading(
        "At its meeting on 30 January 2025, the Governing Council decided to lower rates."
    )


def test_empty_pages_are_ignored() -> None:
    body = "Summary" + chr(10) + chr(10) + "The disinflation process is well on track."
    chunks = chunk_document(
        _doc([(1, "   "), (2, body)]),
        doc_id="empty-page",
    )
    assert len(chunks) == 1
    assert chunks[0].page == 2
    assert chunks[0].section == "Summary"
