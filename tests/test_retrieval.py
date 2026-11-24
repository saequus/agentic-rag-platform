"""Day 3 retrieval units: tokenizer, BM25 vectors, RRF, reranking, Qdrant in-memory."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from agentic_rag_platform.ingest.embedder import HashEmbedder
from agentic_rag_platform.ingest.types import SearchHit
from agentic_rag_platform.retrieval import qdrant_index
from agentic_rag_platform.retrieval.fusion import rrf_fuse, rrf_scores
from agentic_rag_platform.retrieval.rerank import rerank
from agentic_rag_platform.retrieval.service import Retriever
from agentic_rag_platform.retrieval.sparse import (
    bm25_document_vector,
    bm25_query_vector,
    query_terms,
    term_id,
    tokenize,
    websearch_or_query,
)
from agentic_rag_platform.settings import Settings


def test_tokenize_keeps_identifiers_numbers_and_polish() -> None:
    tokens = tokenize("Stopa procentowa wynosi 5,35% dla EDO1136 i 0.5% w październiku.")
    assert "5,35" in tokens
    assert "edo1136" in tokens
    assert "0.5" in tokens
    assert "październiku" in tokens
    # stopwords and single letters are dropped
    assert "dla" not in tokens and "i" not in tokens and "w" not in tokens


def test_query_terms_are_unique_and_ordered() -> None:
    assert query_terms("ECB rates ECB the rates 2026") == ["ecb", "rates", "2026"]


def test_websearch_or_query() -> None:
    assert websearch_or_query("What is the rate of COI1130?") == "rate or coi1130"
    assert websearch_or_query("the of and") is None


def test_term_id_is_stable_and_31_bit() -> None:
    assert term_id("edo1136") == term_id("edo1136")
    assert 0 <= term_id("edo1136") < 2**31


def test_bm25_document_vector_saturates_and_normalises_length() -> None:
    once = bm25_document_vector("inflation", avgdl=10)
    many = bm25_document_vector("inflation " * 50, avgdl=10)
    assert once.indices == many.indices
    # term frequency saturates below k1 + 1 = 2.2 however often it repeats
    assert once.values[0] < many.values[0] < 2.2
    short = bm25_document_vector("inflation target", avgdl=10)
    long = bm25_document_vector("inflation target " + "word " * 40, avgdl=10)
    idx = short.indices.index(term_id("inflation"))
    long_idx = long.indices.index(term_id("inflation"))
    assert short.values[idx] > long.values[long_idx]


def test_bm25_query_vector_is_binary() -> None:
    vector = bm25_query_vector("EDO1136 EDO1136 rate")
    assert len(vector.indices) == 2
    assert vector.values == [1.0, 1.0]


def test_rrf_matches_formula() -> None:
    scores = rrf_scores([["a", "b", "c"], ["c", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert rrf_fuse([["a", "b", "c"], ["c", "a"]], k=60) == ["a", "c", "b"]
    assert rrf_fuse([["x"], ["y"]], k=60, limit=1) == ["x"]
    with pytest.raises(ValueError):
        rrf_scores([["a"]], k=0)


def _hit(doc: str, chunk: int, text: str, score: float = 0.0) -> SearchHit:
    return SearchHit(
        doc_id=doc, chunk_no=chunk, page=1, section=None, text=text, content_hash="h", score=score
    )


class KeywordReranker:
    """Scores a passage by how many query words it contains."""

    name = "keyword-reranker"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        words = set(query.lower().split())
        return [float(sum(word in passage.lower() for word in words)) for passage in passages]


def test_rerank_reorders_and_truncates() -> None:
    hits = [_hit("d", 0, "nothing here"), _hit("d", 1, "ecb rate decision"), _hit("d", 2, "ecb")]
    out = rerank(KeywordReranker(), "ecb rate decision", hits, top_k=2)
    assert [hit.chunk_no for hit in out] == [1, 2]
    assert out[0].score == 3.0


CORPUS = [
    ("pl-bond-edo-108-2026", 7, "W pierwszym okresie odsetkowym stopa procentowa wynosi 5,35%."),
    ("pl-bond-coi-107-2026", 6, "W pierwszym okresie odsetkowym stopa procentowa wynosi 4,75%."),
    ("ecb-eb202606.en", 12, "The Governing Council decided to raise the key ECB interest rates."),
    ("ecb-eb202607.en", 11, "The Governing Council decided to keep the key ECB interest rates."),
]


@pytest.fixture
def qdrant_retriever() -> Retriever:
    settings = Settings(qdrant_path=":memory:", retrieval_candidates=10)
    embedder = HashEmbedder(dimensions=32)
    client = qdrant_index.QdrantClient(location=":memory:")
    qdrant_index.ensure_collection(client, "chunks", 32)
    chunks = []
    for tenant in ("default", "other"):
        for doc_id, chunk_no, text in CORPUS:
            chunks.append(
                qdrant_index.IndexedChunk(
                    tenant_id=tenant,
                    doc_id=doc_id if tenant == "default" else f"{tenant}-{doc_id}",
                    chunk_no=chunk_no,
                    page=1,
                    section=None,
                    content=text,
                    content_hash="h",
                    embedding_model=embedder.name,
                    embedding=embedder.embed_passages([text])[0],
                )
            )
    qdrant_index.upsert_chunks(client, "chunks", chunks, avgdl=10)
    # ensure_collection is idempotent
    qdrant_index.ensure_collection(client, "chunks", 32)
    return Retriever(
        settings=settings,
        embedder=embedder,
        qdrant_client=client,
        reranker_factory=KeywordReranker,
    )


def test_qdrant_sparse_finds_exact_identifier(qdrant_retriever: Retriever) -> None:
    result = qdrant_retriever.retrieve(
        "5,35", tenant_id="default", k=2, strategy="sparse", backend="qdrant"
    )
    assert result.hits[0].doc_id == "pl-bond-edo-108-2026"
    assert "embed_ms" not in result.timings_ms


def test_qdrant_hybrid_is_tenant_isolated(qdrant_retriever: Retriever) -> None:
    for strategy in ("dense", "hybrid", "hybrid_rerank"):
        result = qdrant_retriever.retrieve(
            "raise ECB interest rates", tenant_id="other", k=4, strategy=strategy, backend="qdrant"
        )
        assert result.hits, strategy
        assert all(hit.doc_id.startswith("other-") for hit in result.hits), strategy


def test_qdrant_hybrid_rerank_puts_best_match_first(qdrant_retriever: Retriever) -> None:
    result = qdrant_retriever.retrieve(
        "decided to raise", tenant_id="default", k=2, strategy="hybrid_rerank", backend="qdrant"
    )
    assert result.hits[0].doc_id == "ecb-eb202606.en"
    assert len(result.hits) == 2
    assert set(result.timings_ms) == {"embed_ms", "search_ms", "rerank_ms"}


def test_qdrant_dbsf_fusion(qdrant_retriever: Retriever) -> None:
    qdrant_retriever.settings = qdrant_retriever.settings.model_copy(
        update={"qdrant_fusion": "dbsf"}
    )
    result = qdrant_retriever.retrieve(
        "5,35", tenant_id="default", k=1, strategy="hybrid", backend="qdrant"
    )
    assert result.hits[0].doc_id == "pl-bond-edo-108-2026"


def test_point_ids_are_deterministic() -> None:
    assert qdrant_index.point_id("t", "d", 1) == qdrant_index.point_id("t", "d", 1)
    assert qdrant_index.point_id("t", "d", 1) != qdrant_index.point_id("u", "d", 1)


def test_blank_query_rejected(qdrant_retriever: Retriever) -> None:
    with pytest.raises(ValueError):
        qdrant_retriever.retrieve("  ", tenant_id="default")


def test_qdrant_idf_statistics_are_scoped_to_the_tenant() -> None:
    """Another tenant's corpus must not change this tenant's BM25 weights."""
    client = qdrant_index.QdrantClient(location=":memory:")
    qdrant_index.ensure_collection(client, "chunks", 8)

    def chunk(tenant: str, no: int, text: str) -> qdrant_index.IndexedChunk:
        return qdrant_index.IndexedChunk(
            tenant_id=tenant,
            doc_id=f"{tenant}-doc",
            chunk_no=no,
            page=None,
            section=None,
            content=text,
            content_hash="h",
            embedding_model="m",
            embedding=[1.0] + [0.0] * 7,
        )

    # In "small", alpha and beta are equally rare. "big" is full of alpha.
    points = [chunk("small", 0, "alpha"), chunk("small", 1, "beta")]
    points += [chunk("big", n, "alpha filler") for n in range(30)]
    qdrant_index.upsert_chunks(client, "chunks", points, avgdl=1.0)
    hits = qdrant_index.search(
        client,
        "chunks",
        mode="sparse",
        tenant_id="small",
        query_text="alpha beta",
        query_vector=None,
        k=2,
    )
    assert {hit.chunk_no for hit in hits} == {0, 1}
    assert hits[0].score == pytest.approx(hits[1].score)
