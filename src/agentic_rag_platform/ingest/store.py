"""Postgres + pgvector storage: upserts, tenant-filtered search, exact scan."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Literal

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import DictRow, dict_row

from agentic_rag_platform.ingest.types import Chunk, SearchHit, StoredChunkRef

SearchBackend = Literal["hnsw", "exact"]
type DictConn = psycopg.Connection[DictRow]


@contextmanager
def connect(database_url: str) -> Iterator[DictConn]:
    """Open a psycopg 3 connection with the pgvector type registered."""
    conn = psycopg.connect(database_url, row_factory=dict_row)
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_document(
    conn: DictConn,
    *,
    tenant_id: str,
    doc_id: str,
    title: str | None,
    source_uri: str | None,
    page_count: int | None,
) -> None:
    conn.execute(
        """
        INSERT INTO documents (tenant_id, doc_id, title, source_uri, page_count)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (tenant_id, doc_id) DO UPDATE SET
            title = EXCLUDED.title,
            source_uri = EXCLUDED.source_uri,
            page_count = EXCLUDED.page_count,
            updated_at = now()
        """,
        (tenant_id, doc_id, title, source_uri, page_count),
    )


def existing_chunks(
    conn: DictConn,
    *,
    tenant_id: str,
    doc_id: str,
) -> dict[int, StoredChunkRef]:
    rows = conn.execute(
        """
        SELECT tenant_id, doc_id, chunk_no, content_hash, embedding_model
        FROM chunks
        WHERE tenant_id = %s AND doc_id = %s
        """,
        (tenant_id, doc_id),
    ).fetchall()
    result: dict[int, StoredChunkRef] = {}
    for row in rows:
        ref = StoredChunkRef(
            tenant_id=row["tenant_id"],
            doc_id=row["doc_id"],
            chunk_no=row["chunk_no"],
            content_hash=row["content_hash"],
            embedding_model=row["embedding_model"],
        )
        result[ref.chunk_no] = ref
    return result


def upsert_chunks(
    conn: DictConn,
    *,
    tenant_id: str,
    chunks: Sequence[Chunk],
    embeddings: Sequence[list[float] | None],
    embedding_model: str,
) -> None:
    """Upsert chunks by ``(tenant_id, doc_id, chunk_no)``.

    ``embeddings[i]`` may be ``None`` when the stored vector should be kept
    (text and model unchanged). In that case only metadata is updated.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        if embedding is None:
            conn.execute(
                """
                UPDATE chunks
                SET page = %s,
                    section = %s,
                    updated_at = now()
                WHERE tenant_id = %s AND doc_id = %s AND chunk_no = %s
                """,
                (chunk.page, chunk.section, tenant_id, chunk.doc_id, chunk.chunk_no),
            )
            continue
        conn.execute(
            """
            INSERT INTO chunks (
                tenant_id, doc_id, chunk_no, page, section,
                content, content_hash, embedding_model, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id, doc_id, chunk_no) DO UPDATE SET
                page = EXCLUDED.page,
                section = EXCLUDED.section,
                content = EXCLUDED.content,
                content_hash = EXCLUDED.content_hash,
                embedding_model = EXCLUDED.embedding_model,
                embedding = EXCLUDED.embedding,
                updated_at = now()
            """,
            (
                tenant_id,
                chunk.doc_id,
                chunk.chunk_no,
                chunk.page,
                chunk.section,
                chunk.text,
                chunk.content_hash,
                embedding_model,
                embedding,
            ),
        )


def delete_chunks_after(
    conn: DictConn,
    *,
    tenant_id: str,
    doc_id: str,
    last_chunk_no: int,
) -> int:
    """Drop leftover chunks when a document shrinks. Returns the row count."""
    result = conn.execute(
        """
        DELETE FROM chunks
        WHERE tenant_id = %s AND doc_id = %s AND chunk_no > %s
        """,
        (tenant_id, doc_id, last_chunk_no),
    )
    return int(result.rowcount or 0)


def fetch_chunk_texts(
    conn: DictConn,
    *,
    tenant_id: str,
    embedding_model: str | None = None,
    limit: int | None = None,
) -> list[SearchHit]:
    """Load stored chunks (no vectors) for benchmark query generation."""
    sql = """
        SELECT doc_id, chunk_no, page, section, content, content_hash,
               0.0 AS score
        FROM chunks
        WHERE tenant_id = %s
    """
    params: list[object] = [tenant_id]
    if embedding_model is not None:
        sql += " AND embedding_model = %s"
        params.append(embedding_model)
    sql += " ORDER BY doc_id, chunk_no"
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [_hit_from_row(row) for row in rows]


def search(
    conn: DictConn,
    *,
    tenant_id: str,
    query: Sequence[float],
    k: int = 10,
    embedding_model: str | None = None,
    ef_search: int | None = None,
    backend: SearchBackend = "hnsw",
) -> list[SearchHit]:
    """Return the top-``k`` neighbours for ``query`` inside one tenant.

    ``backend="exact"`` forces a sequential scan so we can measure HNSW
    recall against the true nearest neighbours. Cosine distance (``<=>``)
    is used; ``score`` is ``1 - distance``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    # SET LOCAL must run inside a transaction; the caller owns the connection
    # and we always wrap the query so the settings do not leak.
    with conn.transaction():
        if backend == "exact":
            conn.execute("SET LOCAL enable_indexscan = off")
            conn.execute("SET LOCAL enable_bitmapscan = off")
            conn.execute("SET LOCAL enable_indexonlyscan = off")
        elif ef_search is not None:
            # ef_search is a planner GUC; interpolate a validated integer.
            if ef_search < 1:
                raise ValueError("ef_search must be at least 1")
            conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")

        sql = """
            SELECT doc_id, chunk_no, page, section, content, content_hash,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE tenant_id = %s
        """
        params: list[object] = [list(query), tenant_id]
        if embedding_model is not None:
            sql += " AND embedding_model = %s"
            params.append(embedding_model)
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([list(query), k])
        rows = conn.execute(sql, params).fetchall()
    return [_hit_from_row(row) for row in rows]


def _hit_from_row(row: DictRow) -> SearchHit:
    page = row["page"]
    score = row["score"]
    return SearchHit(
        doc_id=str(row["doc_id"]),
        chunk_no=int(row["chunk_no"]),
        page=int(page) if page is not None else None,
        section=str(row["section"]) if row["section"] is not None else None,
        text=str(row["content"]),
        content_hash=str(row["content_hash"]),
        score=float(score) if score is not None else 0.0,
    )


def count_chunks(conn: DictConn, *, tenant_id: str) -> int:
    row = conn.execute(
        "SELECT count(*) AS n FROM chunks WHERE tenant_id = %s",
        (tenant_id,),
    ).fetchone()
    return int(row["n"]) if row else 0
