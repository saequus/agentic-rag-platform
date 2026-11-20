"""Recursive, structure-aware chunker.

Chunks target 512–800 tokens with 10–15% overlap. Section headings are
detected and stored as metadata on every chunk they cover. Token counts use
tiktoken ``cl100k_base`` as a stable approximation; the embedding model's
own tokenizer may differ by a few tokens.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import tiktoken

from agentic_rag_platform.ingest.hashing import content_hash
from agentic_rag_platform.ingest.types import Chunk, ParsedDocument, PdfPage

DEFAULT_MAX_TOKENS = 800
DEFAULT_MIN_TOKENS = 512
DEFAULT_OVERLAP_RATIO = 0.12  # middle of the 10–15% range
ENCODING_NAME = "cl100k_base"

_NUMBERED_HEADING = re.compile(
    r"^\d{1,2}(?:\.\d{1,3}){0,3}\.?\s+\S.{1,120}$",
)
_KEYWORD_HEADING = re.compile(
    r"^(?:"
    r"section|chapter|part|article|annex|appendix|box|"
    r"rozdział|rozdzial|część|czesc|artykuł|artykul|aneks|załącznik|zalacznik"
    r")\s+[\wIVXivx]+(?:[.:)]\s+|\s+).+",
    re.IGNORECASE,
)
_ALL_CAPS_HEADING = re.compile(
    r"^[A-Z0-9][A-Z0-9ÁÉÍÓÚÄÖÜĄĆĘŁŃÓŚŹŻ ,/&'()\-]{6,80}$",
)
_BARE_TITLE = re.compile(
    r"^[A-ZĄĆĘŁŃÓŚŹŻ][\wĄĆĘŁŃÓŚŹŻąćęłńóśźż ,/&'()\-]{2,80}$",
)

_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


@dataclass(frozen=True)
class _Unit:
    text: str
    page: int
    section: str


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text))


def token_tail(text: str, n_tokens: int) -> str:
    """Return the last ``n_tokens`` tokens of ``text`` (or all of it)."""
    if n_tokens <= 0 or not text:
        return ""
    tokens = _encoding().encode(text)
    if len(tokens) <= n_tokens:
        return text
    return _encoding().decode(tokens[-n_tokens:])


def token_head(text: str, n_tokens: int) -> str:
    """Return the first ``n_tokens`` tokens of ``text`` (or all of it)."""
    if n_tokens <= 0 or not text:
        return ""
    tokens = _encoding().encode(text)
    if len(tokens) <= n_tokens:
        return text
    return _encoding().decode(tokens[:n_tokens])


def is_heading(line: str) -> bool:
    """Return True if ``line`` looks like a section heading, not body text."""
    candidate = " ".join(line.strip().split())
    if not candidate or len(candidate) > 140:
        return False
    if _NUMBERED_HEADING.match(candidate):
        return True
    if _KEYWORD_HEADING.match(candidate):
        return True
    if _ALL_CAPS_HEADING.match(candidate) and not candidate.endswith("."):
        return True
    # Short title-case line with no sentence punctuation — common in ECB/NBP
    # contents pages and chapter openings.
    return bool(
        _BARE_TITLE.match(candidate) and "." not in candidate and count_tokens(candidate) <= 16
    )


def chunk_document(
    document: ParsedDocument,
    *,
    doc_id: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list[Chunk]:
    """Split ``document`` into overlapping, section-aware chunks."""
    if max_tokens < 32:
        raise ValueError("max_tokens must be at least 32")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be in [0, 1)")
    if min_tokens > max_tokens:
        raise ValueError("min_tokens must be <= max_tokens")

    overlap_tokens = max(0, int(round(max_tokens * overlap_ratio)))
    units = _units_from_pages(document.pages)
    packed = _pack_units(units, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    packed = _merge_short_tail(packed, min_tokens=min_tokens, max_tokens=max_tokens)

    chunks: list[Chunk] = []
    for index, unit in enumerate(packed):
        text = unit.text.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_no=index,
                page=unit.page,
                section=unit.section or None,
                text=text,
                content_hash=content_hash(text),
            )
        )
    return chunks


def _units_from_pages(pages: Sequence[PdfPage]) -> list[_Unit]:
    """Turn pages into heading-bounded paragraph units, carrying the section."""
    units: list[_Unit] = []
    section = ""
    for page in pages:
        for raw_block in _blocks(page.text):
            block = raw_block.strip()
            if not block:
                continue
            if _block_is_heading(block):
                section = " ".join(block.split())
                continue
            units.append(_Unit(text=block, page=page.page, section=section))
    return units


def _blocks(text: str) -> list[str]:
    """Split page text into paragraphs, keeping single-line headings intact."""
    if not text.strip():
        return []
    paragraphs = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    blocks: list[str] = []
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        if not lines:
            continue
        buffer: list[str] = []
        for line in lines:
            if is_heading(line) and buffer:
                blocks.append(" ".join(buffer))
                buffer = []
                blocks.append(line)
            elif is_heading(line):
                blocks.append(line)
            else:
                buffer.append(line)
        if buffer:
            blocks.append(" ".join(buffer))
    return blocks


def _block_is_heading(block: str) -> bool:
    # A heading is a short block that itself looks like a heading, not a
    # paragraph that happens to start with a number.
    if "\n" in block:
        return False
    return is_heading(block) and count_tokens(block) <= 24


def _pack_units(
    units: Sequence[_Unit],
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[_Unit]:
    if not units:
        return []

    chunks: list[_Unit] = []
    current_parts: list[str] = []
    current_tokens = 0
    current_page = units[0].page
    current_section = units[0].section

    def flush() -> None:
        nonlocal current_parts, current_tokens
        text = " ".join(part for part in current_parts if part).strip()
        if text:
            chunks.append(_Unit(text=text, page=current_page, section=current_section))
        current_parts = []
        current_tokens = 0

    def start_new_chunk(*, carry_overlap: bool, page: int, section: str) -> None:
        nonlocal current_parts, current_tokens, current_page, current_section
        flush()
        current_page = page
        current_section = section
        if carry_overlap and chunks and overlap_tokens:
            overlap = token_tail(chunks[-1].text, overlap_tokens)
            if overlap:
                current_parts = [overlap]
                current_tokens = count_tokens(overlap)

    for unit in units:
        queue = list(_split_to_max(unit.text, max_tokens))
        while queue:
            piece = queue.pop(0)
            piece_tokens = count_tokens(piece)
            if not piece_tokens:
                continue

            if current_parts and unit.section != current_section:
                start_new_chunk(carry_overlap=False, page=unit.page, section=unit.section)

            remaining = max_tokens - current_tokens
            if current_parts and piece_tokens > remaining:
                # Fill the current window with the head of this piece, then
                # open a new one. This avoids emitting a short leftover chunk
                # in the middle of a section.
                if remaining >= 8:
                    head, tail = _split_at(piece, remaining)
                    if head:
                        current_parts.append(head)
                        current_tokens += count_tokens(head)
                    piece = tail
                    piece_tokens = count_tokens(piece)
                start_new_chunk(carry_overlap=True, page=unit.page, section=unit.section)
                remaining = max_tokens - current_tokens
                if piece and piece_tokens > remaining:
                    if remaining < 8:
                        start_new_chunk(
                            carry_overlap=False,
                            page=unit.page,
                            section=unit.section,
                        )
                        remaining = max_tokens - current_tokens
                    head, tail = _split_at(piece, remaining)
                    if tail:
                        queue.insert(0, tail)
                    piece = head
                    piece_tokens = count_tokens(piece)

            if not piece:
                continue
            if not current_parts:
                current_page = unit.page
                current_section = unit.section
            current_parts.append(piece)
            current_tokens += piece_tokens
            if current_tokens > max_tokens and len(current_parts) == 1:
                flush()

    flush()
    return chunks


def _split_at(text: str, n_tokens: int) -> tuple[str, str]:
    """Split ``text`` so the first part has at most ``n_tokens`` tokens."""
    if n_tokens <= 0:
        return "", text
    tokens = _encoding().encode(text)
    if len(tokens) <= n_tokens:
        return text, ""
    head = _encoding().decode(tokens[:n_tokens]).strip()
    tail = _encoding().decode(tokens[n_tokens:]).strip()
    return head, tail


def _split_to_max(text: str, max_tokens: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]
    return _recursive_split(text, max_tokens, _SEPARATORS)


def _recursive_split(text: str, max_tokens: int, separators: tuple[str, ...]) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text] if text.strip() else []
    if not separators:
        return _hard_split(text, max_tokens)

    separator = separators[0]
    rest = separators[1:]
    if separator not in text:
        return _recursive_split(text, max_tokens, rest)

    parts = [part for part in text.split(separator) if part.strip()]
    assembled: list[str] = []
    buffer = ""
    for part in parts:
        candidate = part if not buffer else buffer + separator + part
        if buffer and count_tokens(candidate) > max_tokens:
            assembled.extend(_recursive_split(buffer.strip(), max_tokens, rest))
            buffer = part
        else:
            buffer = candidate
    if buffer.strip():
        assembled.extend(_recursive_split(buffer.strip(), max_tokens, rest))
    return [item for item in assembled if item]


def _hard_split(text: str, max_tokens: int) -> list[str]:
    tokens = _encoding().encode(text)
    pieces: list[str] = []
    for start in range(0, len(tokens), max_tokens):
        piece = _encoding().decode(tokens[start : start + max_tokens]).strip()
        if piece:
            pieces.append(piece)
    return pieces


def _merge_short_tail(
    chunks: list[_Unit],
    *,
    min_tokens: int,
    max_tokens: int,
) -> list[_Unit]:
    """Merge a short final chunk into the previous one when they fit and share a section."""
    if len(chunks) < 2:
        return chunks
    last = chunks[-1]
    prev = chunks[-2]
    last_tokens = count_tokens(last.text)
    if last_tokens >= min_tokens:
        return chunks
    if last.section != prev.section:
        return chunks
    combined = prev.text + " " + last.text
    if count_tokens(combined) > max_tokens:
        return chunks
    merged = replace(prev, text=combined)
    return [*chunks[:-2], merged]
