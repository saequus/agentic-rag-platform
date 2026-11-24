"""Apply the SQL migrations in ``migrations/`` in order (all idempotent).

Usage:
    uv run python -m agentic_rag_platform.ingest.migrate
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import psycopg

from agentic_rag_platform.ingest.embedder import dimension_for_model
from agentic_rag_platform.ingest.paths import migrations_dir
from agentic_rag_platform.settings import Settings, get_settings

PLACEHOLDER = "__EMBEDDING_DIM__"
_DIM_TYPE = re.compile(r"vector\((\d+)\)")


def init_schema(database_url: str, dimensions: int) -> None:
    """Create extension, tables and the HNSW index if they do not exist."""
    if dimensions < 1:
        raise ValueError("dimensions must be a positive integer")

    sql_paths = migration_files()
    if not sql_paths or sql_paths[0].name != "001_init.sql":
        raise FileNotFoundError(f"Migration file not found: {migrations_dir() / '001_init.sql'}")
    scripts = [
        path.read_text(encoding="utf-8").replace(PLACEHOLDER, str(dimensions)) for path in sql_paths
    ]

    with psycopg.connect(database_url, autocommit=True) as conn:
        existing = _existing_dimension(conn)
        if existing is not None and existing != dimensions:
            raise RuntimeError(
                f"chunks.embedding is vector({existing}) but the current model "
                f"needs vector({dimensions}). Create a new database, or drop "
                "the chunks table, then re-run make db-init and re-ingest."
            )
        if existing == dimensions:
            # Tables already match; still apply the script so new indexes are
            # created (CREATE IF NOT EXISTS is safe).
            pass
        for sql in scripts:
            _run_sql(conn, sql)


def migration_files() -> list[Path]:
    """All ``NNN_*.sql`` files in order (every script must be idempotent)."""
    return sorted(migrations_dir().glob("[0-9][0-9][0-9]_*.sql"))


def _existing_dimension(conn: psycopg.Connection) -> int | None:
    row = conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS typ
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname = 'chunks'
          AND a.attname = 'embedding'
          AND a.attnum > 0
          AND NOT a.attisdropped
        """
    ).fetchone()
    if row is None:
        return None
    typ = row[0] if not isinstance(row, dict) else row["typ"]
    match = _DIM_TYPE.search(str(typ))
    return int(match.group(1)) if match else None


def _run_sql(conn: psycopg.Connection, sql: str) -> None:
    statements = [part.strip() for part in _split_sql(sql) if part.strip()]
    for statement in statements:
        conn.execute(statement)


def _split_sql(sql: str) -> list[str]:
    """Split a simple SQL script on semicolons, keeping line comments out."""
    lines: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    return [chunk for chunk in cleaned.split(";") if chunk.strip()]


def _dimensions_from_settings(settings: Settings) -> int:
    if settings.embedding_provider == "openai":
        return dimension_for_model(settings.openai_embedding_model, default=1536)
    if settings.embedding_provider == "hash":
        return dimension_for_model(settings.embedding_model, default=768)
    return dimension_for_model(settings.embedding_model, default=768)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialise the pgvector schema")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override DATABASE_URL (defaults to the value in settings / .env)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Embedding dimension (default: inferred from EMBEDDING_MODEL)",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    url = args.database_url or settings.database_url
    dim = args.dimensions if args.dimensions is not None else _dimensions_from_settings(settings)
    print(f"Initialising schema on {url} with vector({dim}) ({settings.embedding_model})")
    try:
        init_schema(url, dim)
    except (OSError, psycopg.Error, RuntimeError, ValueError) as exc:
        print(f"db-init failed: {exc}", file=sys.stderr)
        return 1
    print("Schema is ready (documents, chunks, HNSW index, full-text GIN index).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
