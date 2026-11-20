"""Tests for the corpus URL list parser (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag_platform.ingest.corpus import parse_urls_file


def test_parse_committed_urls_file() -> None:
    entries = parse_urls_file(Path("data/urls.txt"))
    assert len(entries) == 23
    assert all(entry.url.startswith("https://") for entry in entries)
    names = [entry.filename for entry in entries]
    assert len(names) == len(set(names))
    assert any(name.startswith("ecb-eb") for name in names)
    assert any(name.startswith("pl-bond-") for name in names)


def test_parse_allows_hash_comments_and_tabs(tmp_path: Path) -> None:
    path = tmp_path / "urls.txt"
    path.write_bytes(
        b"# comment\nhttps://example.com/a.pdf\tec-a.pdf\n\nhttps://example.com/b.pdf b.pdf\n"
    )
    entries = parse_urls_file(path)
    assert [entry.filename for entry in entries] == ["ec-a.pdf", "b.pdf"]


def test_parse_rejects_http(tmp_path: Path) -> None:
    path = tmp_path / "urls.txt"
    path.write_bytes(b"http://example.com/a.pdf\ta.pdf\n")
    with pytest.raises(ValueError, match="https"):
        parse_urls_file(path)
