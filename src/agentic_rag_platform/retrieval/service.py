"""One entry point for every retrieval strategy on either backend.

strategy       pgvector                         qdrant
dense          HNSW cosine                      named "dense" vector (HNSW)
sparse         tsvector + ts_rank_cd            BM25 sparse vector (IDF server-side)
hybrid         dense + sparse, RRF in SQL       Query API prefetch x2 + RRF/DBSF
hybrid_rerank  hybrid top-N -> cross-encoder -> top-k (either backend)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from agentic_rag_platform.ingest.embedder import Embedder
from agentic_rag_platform.ingest.store import connect
from agentic_rag_platform.ingest.store import search as pg_dense_search
from agentic_rag_platform.ingest.types import SearchHit
from agentic_rag_platform.retrieval import pg, qdrant_index
from agentic_rag_platform.retrieval.rerank import Reranker, rerank
from agentic_rag_platform.settings import RetrievalBackend, RetrievalStrategy, Settings
from agentic_rag_platform.telemetry.tracing import (
    OP_NAME,
    REQUEST_MODEL,
    RETRIEVAL_TOP_K,
    TENANT_ID,
    document_refs,
    start_span,
)


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    strategy: RetrievalStrategy
    backend: RetrievalBackend
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class Retriever:
    """Holds the long-lived pieces (settings, embedder, lazy reranker, Qdrant client)."""

    settings: Settings
    embedder: Embedder
    reranker_factory: Callable[[], Reranker] | None = None
    qdrant_client: object | None = None
    _reranker: Reranker | None = None

    def reranker(self) -> Reranker:
        if self._reranker is None:
            if self.reranker_factory is None:
                from agentic_rag_platform.retrieval.rerank import get_reranker

                self._reranker = get_reranker(
                    self.settings.rerank_model, self.settings.rerank_batch_size
                )
            else:
                self._reranker = self.reranker_factory()
        return self._reranker

    def qdrant(self) -> qdrant_index.QdrantClient:
        if self.qdrant_client is None:
            self.qdrant_client = qdrant_index.shared_client(self.settings)
        assert isinstance(self.qdrant_client, qdrant_index.QdrantClient)
        return self.qdrant_client

    def embed_query(self, query: str) -> list[float]:
        vectors = self.embedder.embed_queries([query])
        if not vectors:
            raise RuntimeError("embedder returned no query vector")
        return vectors[0]

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        k: int = 10,
        strategy: RetrievalStrategy | None = None,
        backend: RetrievalBackend | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> RetrievalResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        strategy = strategy or self.settings.retrieval_strategy
        backend = backend or self.settings.retrieval_backend
        timings: dict[str, float] = {}
        with start_span(
            "retrieval",
            attributes={
                OP_NAME: "retrieval",
                RETRIEVAL_TOP_K: k,
                TENANT_ID: tenant_id,
                "gen_ai.data_source.id": backend,
                "retrieval.strategy": strategy,
                REQUEST_MODEL: self.embedder.name,
            },
        ) as span:
            vector: Sequence[float] | None = query_vector
            if vector is None and strategy != "sparse":
                started = time.perf_counter()
                with start_span(
                    f"embeddings {self.embedder.name}",
                    attributes={OP_NAME: "embeddings", REQUEST_MODEL: self.embedder.name},
                ):
                    vector = self.embed_query(query)
                timings["embed_ms"] = _ms(started)

            first_stage = "hybrid" if strategy == "hybrid_rerank" else strategy
            first_k = (
                max(k, self.settings.retrieval_candidates) if strategy == "hybrid_rerank" else k
            )
            started = time.perf_counter()
            hits = self._first_stage(query, vector, first_stage, backend, tenant_id, first_k)
            timings["search_ms"] = _ms(started)

            if strategy == "hybrid_rerank":
                reranker = self.reranker()
                started = time.perf_counter()
                with start_span(
                    f"rerank {reranker.name}",
                    attributes={
                        OP_NAME: "rerank",
                        REQUEST_MODEL: reranker.name,
                        "rerank.candidates": len(hits),
                    },
                ):
                    hits = rerank(reranker, query, hits, k)
                timings["rerank_ms"] = _ms(started)

            span.set_attribute("gen_ai.retrieval.document.count", len(hits))
            refs = document_refs(hits)
            if refs:
                span.set_attribute("gen_ai.retrieval.document.ids", tuple(r["id"] for r in refs))
            return RetrievalResult(
                hits=hits, strategy=strategy, backend=backend, timings_ms=timings
            )

    def _first_stage(
        self,
        query: str,
        vector: Sequence[float] | None,
        strategy: str,
        backend: RetrievalBackend,
        tenant_id: str,
        k: int,
    ) -> list[SearchHit]:
        settings = self.settings
        if backend == "qdrant":
            mode: qdrant_index.QdrantMode = "hybrid"
            if strategy == "dense":
                mode = "dense"
            elif strategy == "sparse":
                mode = "sparse"
            return qdrant_index.search(
                self.qdrant(),
                settings.qdrant_collection,
                mode=mode,
                tenant_id=tenant_id,
                query_text=query,
                query_vector=vector,
                k=k,
                candidates=settings.retrieval_candidates,
                fusion=settings.qdrant_fusion,
                rrf_k=settings.rrf_k,
                embedding_model=self.embedder.name,
                hnsw_ef=settings.hnsw_ef_search,
            )
        with connect(settings.database_url) as conn:
            if strategy == "sparse":
                return pg.sparse_search(
                    conn,
                    tenant_id=tenant_id,
                    query_text=query,
                    k=k,
                    scoring=settings.pg_lexical_scoring,
                )
            assert vector is not None
            if strategy == "dense":
                return pg_dense_search(
                    conn,
                    tenant_id=tenant_id,
                    query=vector,
                    k=k,
                    embedding_model=self.embedder.name,
                    ef_search=max(settings.hnsw_ef_search, k),
                )
            return pg.hybrid_search(
                conn,
                tenant_id=tenant_id,
                query_text=query,
                query_vector=vector,
                k=k,
                candidates=settings.retrieval_candidates,
                rrf_k=settings.rrf_k,
                embedding_model=self.embedder.name,
                ef_search=settings.hnsw_ef_search,
                scoring=settings.pg_lexical_scoring,
            )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 2)
