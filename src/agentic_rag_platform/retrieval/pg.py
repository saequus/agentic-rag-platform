"""Postgres retrieval: lexical (tsvector) and dense + lexical RRF in one SQL query.

Requires ``migrations/002_fulltext.sql`` (a generated ``content_tsv`` column
and a GIN index).

Two lexical scorers:

* ``ts_rank_cd`` (naive baseline): one OR tsquery ranked by cover density.
  Postgres ranking has **no IDF**, so common words ("rate", "euro") dominate.
* ``idf`` (default): BM25-flavoured. Per query term, the document frequency is
  counted through the GIN index, ``idf = ln(1 + (N - df + 0.5) / (df + 0.5))``,
  and a chunk scores ``sum(idf(t) * ts_rank(content_tsv, t))`` over matched
  terms. Not exact BM25 (``ts_rank`` uses log tf, no k1/b saturation), but it
  recovers most of the gap. For real BM25 inside Postgres: ParadeDB
  ``pg_search`` or Timescale ``pg_textsearch``.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag_platform.ingest.store import DictConn, _hit_from_row
from agentic_rag_platform.ingest.types import SearchHit
from agentic_rag_platform.retrieval.sparse import query_terms, websearch_or_query
from agentic_rag_platform.settings import LexicalScoring

TS_CONFIG = "simple"

# MATERIALIZED stops Postgres inlining the CTEs, which would otherwise
# re-run the per-term document-frequency count for every candidate row.
_IDF_TERMS_CTE = f"""
    q AS MATERIALIZED (
        SELECT term, phraseto_tsquery('{TS_CONFIG}', term) AS tq
        FROM unnest(%(terms)s::text[]) AS term
    ),
    total AS MATERIALIZED (
        SELECT count(*)::float8 AS n FROM chunks WHERE tenant_id = %(tenant)s
    ),
    w AS MATERIALIZED (
        SELECT q.tq,
               ln(1 + (total.n - df.n + 0.5) / (df.n + 0.5)) AS idf
        FROM q CROSS JOIN total
        CROSS JOIN LATERAL (
            SELECT count(*)::float8 AS n FROM chunks c
            WHERE c.tenant_id = %(tenant)s AND c.content_tsv @@ q.tq
        ) df
        WHERE numnode(q.tq) > 0 AND df.n > 0
    )
"""

_IDF_RANKED = """
    SELECT c.doc_id, c.chunk_no, sum(w.idf * ts_rank(c.content_tsv, w.tq)) AS lex_score
    FROM w JOIN chunks c ON c.tenant_id = %(tenant)s AND c.content_tsv @@ w.tq
    GROUP BY c.doc_id, c.chunk_no
"""

_TSRANK_RANKED = f"""
    SELECT c.doc_id, c.chunk_no, ts_rank_cd(c.content_tsv, q) AS lex_score
    FROM chunks c, websearch_to_tsquery('{TS_CONFIG}', %(tsq)s) AS q
    WHERE c.tenant_id = %(tenant)s AND c.content_tsv @@ q
"""


def _lexical_parts(scoring: LexicalScoring) -> tuple[str, str]:
    """(extra CTEs, ranked-subquery SQL) for the chosen scorer."""
    if scoring == "idf":
        return _IDF_TERMS_CTE + ",", _IDF_RANKED
    return "", _TSRANK_RANKED


def _lexical_params(query_text: str) -> dict[str, object]:
    return {"terms": query_terms(query_text), "tsq": websearch_or_query(query_text) or ""}


def sparse_search(
    conn: DictConn,
    *,
    tenant_id: str,
    query_text: str,
    k: int = 10,
    scoring: LexicalScoring = "idf",
) -> list[SearchHit]:
    params = _lexical_params(query_text)
    if not params["terms"]:
        return []
    ctes, ranked = _lexical_parts(scoring)
    sql = f"""
        WITH {ctes}
        lex AS ({ranked})
        SELECT c.doc_id, c.chunk_no, c.page, c.section, c.content, c.content_hash,
               lex.lex_score::float8 AS score
        FROM lex JOIN chunks c
          ON c.tenant_id = %(tenant)s AND c.doc_id = lex.doc_id AND c.chunk_no = lex.chunk_no
        ORDER BY lex.lex_score DESC, c.doc_id, c.chunk_no
        LIMIT %(k)s
    """
    params.update(tenant=tenant_id, k=k)
    rows = conn.execute(sql, params).fetchall()
    return [_hit_from_row(row) for row in rows]


def hybrid_search(
    conn: DictConn,
    *,
    tenant_id: str,
    query_text: str,
    query_vector: Sequence[float],
    k: int = 10,
    candidates: int = 50,
    rrf_k: int = 60,
    embedding_model: str | None = None,
    ef_search: int | None = None,
    scoring: LexicalScoring = "idf",
) -> list[SearchHit]:
    """Dense (HNSW) top-N and full-text top-N fused with RRF inside Postgres.

    Each leg is an ORDER BY ... LIMIT subquery so the planner can use its index
    (HNSW for the dense leg, GIN for the sparse leg); ranks are assigned after
    the limit with ``row_number() OVER ()`` on the already-ordered rows.
    """
    if k < 1 or candidates < 1:
        raise ValueError("k and candidates must be at least 1")
    ctes, ranked = _lexical_parts(scoring)
    model_filter = "AND embedding_model = %(model)s" if embedding_model is not None else ""
    sql = f"""
        WITH {ctes}
        dense AS (
            SELECT doc_id, chunk_no, row_number() OVER () AS rnk
            FROM (
                SELECT doc_id, chunk_no
                FROM chunks
                WHERE tenant_id = %(tenant)s {model_filter}
                ORDER BY embedding <=> %(vec)s::vector
                LIMIT %(n)s
            ) d
        ),
        sparse AS (
            SELECT doc_id, chunk_no, row_number() OVER () AS rnk
            FROM (
                SELECT doc_id, chunk_no FROM ({ranked}) lex
                ORDER BY lex_score DESC, doc_id, chunk_no
                LIMIT %(n)s
            ) s
        ),
        fused AS (
            SELECT COALESCE(dense.doc_id, sparse.doc_id) AS doc_id,
                   COALESCE(dense.chunk_no, sparse.chunk_no) AS chunk_no,
                   COALESCE(1.0 / (%(rrf_k)s + dense.rnk), 0)
                 + COALESCE(1.0 / (%(rrf_k)s + sparse.rnk), 0) AS score
            FROM dense FULL OUTER JOIN sparse
              ON dense.doc_id = sparse.doc_id AND dense.chunk_no = sparse.chunk_no
        )
        SELECT c.doc_id, c.chunk_no, c.page, c.section, c.content, c.content_hash,
               fused.score::float8 AS score
        FROM fused
        JOIN chunks c
          ON c.tenant_id = %(tenant)s AND c.doc_id = fused.doc_id
         AND c.chunk_no = fused.chunk_no
        ORDER BY fused.score DESC, c.doc_id, c.chunk_no
        LIMIT %(k)s
    """
    params: dict[str, object] = {
        "tenant": tenant_id,
        "vec": list(query_vector),
        "n": candidates,
        **_lexical_params(query_text),
        "rrf_k": rrf_k,
        "k": k,
    }
    if embedding_model is not None:
        params["model"] = embedding_model
    if ef_search is not None and ef_search < 1:
        raise ValueError("ef_search must be at least 1")
    # HNSW returns at most ef_search rows, so the dense leg needs ef >= candidates.
    effective_ef = max(ef_search or 40, candidates)
    with conn.transaction():
        conn.execute(f"SET LOCAL hnsw.ef_search = {int(effective_ef)}")
        rows = conn.execute(sql, params).fetchall()
    return [_hit_from_row(row) for row in rows]
