#!/usr/bin/env python3
"""Download the public finance PDF corpus listed in data/urls.txt.

Usage:
    uv run python scripts/download_corpus.py
    uv run python scripts/download_corpus.py --check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

# Make the package importable when the script is run as a file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag_platform.ingest.corpus import (  # noqa: E402
    CorpusEntry,
    is_pdf,
    parse_urls_file,
)
from agentic_rag_platform.ingest.paths import default_raw_dir, default_urls_file  # noqa: E402

DEFAULT_USER_AGENT = (
    "agentic-rag-platform/0.1 (+https://github.com/saequus/agentic-rag-platform; "
    "corpus downloader) Mozilla/5.0"
)


def download_one(
    client: httpx.Client,
    entry: CorpusEntry,
    dest: Path,
    *,
    force: bool,
) -> tuple[str, str]:
    """Return (status, detail). Status is one of: skipped, downloaded, failed."""
    if dest.is_file() and not force:
        existing = dest.read_bytes()[:8]
        if is_pdf(existing):
            return "skipped", f"{dest.name} already present ({dest.stat().st_size} bytes)"
    try:
        response = client.get(entry.url)
    except httpx.HTTPError as exc:
        return "failed", f"{entry.filename}: request error: {exc}"

    content_type = response.headers.get("content-type", "")
    if response.status_code != 200:
        return "failed", (
            f"{entry.filename}: HTTP {response.status_code} ({content_type}) from {entry.url}"
        )
    if not is_pdf(response.content):
        preview = response.content[:80].decode("utf-8", errors="replace")
        preview = " ".join(preview.split())
        return "failed", (
            f"{entry.filename}: not a PDF (content-type={content_type!r}, starts {preview!r})"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    return "downloaded", f"{dest.name} ({len(response.content)} bytes)"


def check_one(client: httpx.Client, entry: CorpusEntry) -> tuple[str, str]:
    try:
        response = client.get(entry.url)
    except httpx.HTTPError as exc:
        return "failed", f"{entry.filename}: request error: {exc}"
    content_type = response.headers.get("content-type", "")
    if response.status_code != 200:
        return "failed", f"{entry.filename}: HTTP {response.status_code} from {entry.url}"
    if not is_pdf(response.content):
        return "failed", f"{entry.filename}: not a PDF (content-type={content_type!r})"
    return "ok", f"{entry.filename}: {len(response.content)} bytes, {content_type}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the Day 2 finance PDF corpus")
    parser.add_argument("--urls", type=Path, default=None, help="URL list (default: data/urls.txt)")
    parser.add_argument(
        "--out", type=Path, default=None, help="Output directory (default: data/raw)"
    )
    parser.add_argument("--check", action="store_true", help="Verify URLs without writing files")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    urls_file = args.urls or default_urls_file()
    out_dir = args.out or default_raw_dir()
    try:
        entries = parse_urls_file(urls_file)
    except (OSError, ValueError) as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    if not entries:
        print(f"No URLs in {urls_file}", file=sys.stderr)
        return 1

    print(f"{'Checking' if args.check else 'Downloading'} {len(entries)} PDFs from {urls_file}")
    ok = 0
    failed = 0
    skipped = 0
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/pdf,*/*"}
    with httpx.Client(
        headers=headers,
        timeout=args.timeout,
        follow_redirects=True,
        http2=False,
    ) as client:
        for index, entry in enumerate(entries, start=1):
            dest = out_dir / entry.filename
            if args.check:
                status, detail = check_one(client, entry)
            else:
                status, detail = download_one(client, entry, dest, force=args.force)
            print(f"  [{index}/{len(entries)}] {status}: {detail}")
            if status in {"ok", "downloaded"}:
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
            # Be polite to official sites when fetching the full set.
            if index < len(entries):
                time.sleep(0.2)

    print(f"Done. ok={ok} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
