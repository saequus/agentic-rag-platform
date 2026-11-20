"""CLI entry point: ingest every PDF under a path.

Usage:
    uv run python -m agentic_rag_platform.ingest.cli
    uv run python -m agentic_rag_platform.ingest.cli --path data/raw --tenant default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

from agentic_rag_platform.ingest.embedder import get_embedder
from agentic_rag_platform.ingest.paths import default_raw_dir
from agentic_rag_platform.ingest.pdf import PdfLoadError
from agentic_rag_platform.ingest.pipeline import ingest_paths, iter_pdf_paths
from agentic_rag_platform.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Postgres + pgvector")
    parser.add_argument(
        "--path",
        default=None,
        help="PDF file or directory (default: data/raw)",
    )
    parser.add_argument("--tenant", default=None, help="tenant_id (default: from settings)")
    parser.add_argument("--doc-id", default=None, help="Override doc_id for a single file")
    args = parser.parse_args(argv)

    settings = get_settings()
    source = default_raw_dir() if args.path is None else Path(args.path)
    tenant_id = args.tenant or settings.default_tenant_id
    try:
        paths = iter_pdf_paths(source)
    except FileNotFoundError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    if not paths:
        print(f"No PDFs found in {source}. Run: make download  (or scripts/download_corpus.py)")
        return 1

    print(f"Ingesting {len(paths)} PDF(s) from {source} as tenant={tenant_id}")
    try:
        embedder = get_embedder(settings)
        print(f"Embedder: {embedder.name} (dim={embedder.dimensions})")
        summary = ingest_paths(
            paths,
            tenant_id=tenant_id,
            database_url=settings.database_url,
            embedder=embedder,
            doc_id=args.doc_id,
        )
    except (PdfLoadError, RuntimeError, OSError, ValueError, psycopg.Error) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    for item in summary.documents:
        print(
            f"  {item.doc_id}: pages={item.pages} chunks={item.chunks} "
            f"embedded={item.embedded} skipped={item.skipped} deleted={item.deleted}"
        )
    print(f"Done. chunks={summary.chunks} embedded={summary.embedded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
