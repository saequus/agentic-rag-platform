"""Live Postgres tests for full-text and SQL RRF hybrid. Skipped unless DATABASE_URL is set."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from agentic_rag_platform.ingest.embedder import HashEmbedder
from agentic_rag_platform.ingest.hashing import content_hash
from agentic_rag_platform.ingest.migrate import init_schema
from agentic_rag_platform.ingest.store import connect, upsert_chunks, upsert_document
from agentic_rag_platform.ingest.store import search as dense_search
from agentic_rag_platform.ingest.types import Chunk
from agentic_rag_platform.retrieval.fusion import rrf_scores
from agentic_rag_platform.retrieval.pg import hybrid_search, sparse_search
from agentic_rag_platform.settings import LexicalScoring

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL is unset"),
]

TEXTS = [
    "W pierwszym okresie odsetkowym stopa procentowa wynosi 5,35% w skali roku (EDO1136).",
    "W pierwszym okresie odsetkowym stopa procentowa wynosi 4,75% w skali roku (COI1130).",
    "The Governing Council decided to raise the three key ECB interest rates by 25 basis points.",
    "Euro area unemployment stood at 6.4% in July, unchanged from June.",
]


@pytest.fixture(scope="module")
def database_url() -> str:
    url = os.environ["DATABASE_URL"]
    init_schema(url, dimensions=768)
    return url


@pytest.fixture
def tenant_id(database_url: str) -> Iterator[str]:
    tenant = f"test-{uuid.uuid4().hex[:12]}"
    embedder = HashEmbedder()
    chunks = [
        Chunk(doc_id="doc", chunk_no=i, page=1, section=None, text=t, content_hash=content_hash(t))
        for i, t in enumerate(TEXTS)
    ]
    with connect(database_url) as conn:
        upsert_document(
            conn, tenant_id=tenant, doc_id="doc", title=None, source_uri=None, page_count=1
        )
        upsert_chunks(
            conn,
            tenant_id=tenant,
            chunks=chunks,
            embeddings=embedder.embed_passages(TEXTS),
            embedding_model=embedder.name,
        )
    yield tenant
    with connect(database_url) as conn:
        conn.execute("DELETE FROM documents WHERE tenant_id = %s", (tenant,))


@pytest.mark.parametrize("scoring", ["idf", "ts_rank_cd"])
def test_sparse_matches_identifier_and_number(
    database_url: str, tenant_id: str, scoring: LexicalScoring
) -> None:
    with connect(database_url) as conn:
        hits = sparse_search(
            conn, tenant_id=tenant_id, query_text="EDO1136 oprocentowanie", k=3, scoring=scoring
        )
        assert [hit.chunk_no for hit in hits] == [0]
        hits = sparse_search(
            conn, tenant_id=tenant_id, query_text="unemployment July", k=3, scoring=scoring
        )
        assert hits[0].chunk_no == 3
        # "5,35" is two tokens in the 'simple' parser; phrase matching keeps them together
        hits = sparse_search(conn, tenant_id=tenant_id, query_text="5,35", k=3, scoring=scoring)
        assert [hit.chunk_no for hit in hits][:1] == [0]
        assert sparse_search(conn, tenant_id=tenant_id, query_text="the of", k=3) == []
        # tenant isolation
        assert sparse_search(conn, tenant_id="nobody", query_text="EDO1136", k=3) == []


def test_idf_prefers_rare_terms(database_url: str, tenant_id: str) -> None:
    # "stopa" is in two chunks, "coi1130" in one: the rare term must win.
    with connect(database_url) as conn:
        hits = sparse_search(conn, tenant_id=tenant_id, query_text="stopa COI1130", k=2)
    assert hits[0].chunk_no == 1
    assert hits[0].score > hits[1].score


def test_sql_rrf_equals_python_rrf(database_url: str, tenant_id: str) -> None:
    embedder = HashEmbedder()
    query = "ECB interest rates basis points"
    vector = embedder.embed_queries([query])[0]
    with connect(database_url) as conn:
        dense = dense_search(
            conn,
            tenant_id=tenant_id,
            query=vector,
            k=4,
            backend="exact",
            embedding_model=embedder.name,
        )
        sparse = sparse_search(conn, tenant_id=tenant_id, query_text=query, k=4)
        fused = hybrid_search(
            conn,
            tenant_id=tenant_id,
            query_text=query,
            query_vector=vector,
            k=4,
            candidates=4,
            embedding_model=embedder.name,
        )
    expected = rrf_scores([[h.chunk_no for h in dense], [h.chunk_no for h in sparse]], k=60)
    assert fused[0].chunk_no == 2  # in both legs
    for hit in fused:
        assert hit.score == pytest.approx(expected[hit.chunk_no])
