"""Stable content hashes used to skip re-embedding unchanged chunks."""

from __future__ import annotations

import hashlib


def normalize_text(text: str) -> str:
    """Collapse any whitespace run to a single space and trim the ends."""
    return " ".join(text.split())


def content_hash(text: str) -> str:
    """SHA-256 of whitespace-normalised UTF-8 text, returned as 64 hex chars."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
