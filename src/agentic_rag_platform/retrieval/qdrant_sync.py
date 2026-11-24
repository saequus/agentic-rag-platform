"""Copy chunks *and their stored embeddings* from Postgres into Qdrant.

Postgres stays the system of record. Reusing the pgvector embeddings means
both backends index exactly the same chunks and vectors, so any difference in
the eval comes from the engine and the sparse model, not from re-embedding.

Usage:
    uv run python -m agentic_rag_platform.retrieval.qdrant_sync [--tenant-id default] [--recreate]
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import psycopg

from agentic_rag_platform.ingest.store import connect
from agentic_rag_platform.retrieval import qdrant_index
from agentic_rag_platform.retrieval.sparse import average_length
from agentic_rag_platform.settings import get_settings


def load_chunks(
    database_url: str, tenant_id: str, embedding_model: str | None
) -> list[qdrant_index.IndexedChunk]:
    sql = """
        SELECT tenant_id, doc_id, chunk_no, page, section, content, content_hash,
               embedding_model, embedding
        FROM chunks WHERE tenant_id = %s
    """
    params: list[object] = [tenant_id]
    if embedding_model is not None:
        sql += " AND embedding_model = %s"
        params.append(embedding_model)
    sql += " ORDER BY doc_id, chunk_no"
    with connect(database_url) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        qdrant_index.IndexedChunk(
            tenant_id=row["tenant_id"],
            doc_id=row["doc_id"],
            chunk_no=int(row["chunk_no"]),
            page=row["page"],
            section=row["section"],
            content=row["content"],
            content_hash=row["content_hash"],
            embedding_model=row["embedding_model"],
            embedding=_floats(row["embedding"]),
        )
        for row in rows
    ]


def _floats(value: Any) -> list[float]:
    """pgvector returns ``Vector`` (psycopg adapter) or a numpy array."""
    if hasattr(value, "to_list"):
        value = value.to_list()
    return [float(component) for component in value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync Postgres chunks into Qdrant")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--embedding-model", default=None, help="Only sync one model's vectors")
    parser.add_argument("--recreate", action="store_true", help="Drop the collection first")
    args = parser.parse_args(argv)

    settings = get_settings()
    tenant_id = args.tenant_id or settings.default_tenant_id
    try:
        chunks = load_chunks(settings.database_url, tenant_id, args.embedding_model)
    except (OSError, psycopg.Error) as exc:
        print(f"qdrant-sync: cannot read Postgres: {exc}", file=sys.stderr)
        return 1
    if not chunks:
        print("qdrant-sync: no chunks in Postgres; run `make ingest` first", file=sys.stderr)
        return 1

    client = qdrant_index.make_client(settings)
    name = settings.qdrant_collection
    if args.recreate and client.collection_exists(name):
        client.delete_collection(name)
    qdrant_index.ensure_collection(client, name, len(chunks[0].embedding))
    avgdl = average_length(chunk.content for chunk in chunks)
    started = time.perf_counter()
    written = qdrant_index.upsert_chunks(client, name, chunks, avgdl=avgdl)
    elapsed = time.perf_counter() - started
    target = settings.qdrant_url or settings.qdrant_path
    print(
        f"Synced {written} chunks for tenant {tenant_id!r} into {target}/{name} "
        f"in {elapsed:.1f}s (BM25 avgdl={avgdl:.1f} tokens)"
    )
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
