"""Parse and validate the committed corpus URL list."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusEntry:
    url: str
    filename: str
    source_line: int


def parse_urls_file(path: Path) -> list[CorpusEntry]:
    if not path.is_file():
        raise FileNotFoundError(f"URL list not found: {path}")
    entries: list[CorpusEntry] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            url, filename = line.split("\t", 1)
        else:
            parts = line.split()
            if len(parts) == 1:
                url, filename = parts[0], Path(parts[0]).name
            elif len(parts) == 2:
                url, filename = parts
            else:
                raise ValueError(f"{path}:{number}: expected '<url> <filename>', got {raw!r}")
        url = url.strip()
        filename = Path(filename.strip()).name
        if not url.startswith("https://"):
            raise ValueError(f"{path}:{number}: only https URLs are allowed: {url}")
        if not filename or filename in {".", ".."}:
            raise ValueError(f"{path}:{number}: invalid filename")
        entries.append(CorpusEntry(url=url, filename=filename, source_line=number))
    return entries


def is_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF-")
