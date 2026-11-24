"""Qdrant backend: one collection, a named dense vector and a BM25 sparse vector.

Layout::

    collection "chunks"
      vectors:        "dense"  (cosine, HNSW m=16, ef_construct=64, same as pgvector)
      sparse_vectors: "bm25"   (Modifier.IDF -> Qdrant applies IDF at query time)
      payload:        tenant_id (keyword index, is_tenant=true), doc_id, chunk_no,
                      page, section, content, content_hash, embedding_model

Hybrid queries use the Query API: two ``Prefetch`` legs (dense, sparse), each
filtered by tenant, fused server-side with RRF (or DBSF). The tenant filter is
always built here from the caller's ``tenant_id`` (which the API takes from
auth context in production), never from free-form user filters.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from qdrant_client import QdrantClient, models

__all__ = ["QdrantClient", "IndexedChunk", "ensure_collection", "search", "upsert_chunks"]

from agentic_rag_platform.ingest.types import SearchHit
from agentic_rag_platform.retrieval.sparse import (
    SparseVector,
    bm25_document_vector,
    bm25_query_vector,
)
from agentic_rag_platform.settings import Settings

DENSE = "dense"
SPARSE = "bm25"
_POINT_NAMESPACE = uuid.UUID("6f1d3d4e-5b7a-4c1e-9a55-2d3c0e7d9a10")

QdrantMode = Literal["dense", "sparse", "hybrid"]


def make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return QdrantClient(url=settings.qdrant_url, api_key=key, timeout=30)
    if settings.qdrant_path == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=settings.qdrant_path)


@lru_cache(maxsize=4)
def _cached_client(url: str | None, api_key: str | None, path: str) -> QdrantClient:
    if url:
        return QdrantClient(url=url, api_key=api_key, timeout=30)
    if path == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=path)


def shared_client(settings: Settings) -> QdrantClient:
    """Process-wide client. Embedded (path) mode holds a file lock, so only one
    client per path may exist; server mode reuses the HTTP connection pool."""
    key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return _cached_client(settings.qdrant_url, key, settings.qdrant_path)


def point_id(tenant_id: str, doc_id: str, chunk_no: int) -> str:
    """Deterministic id so re-syncing a chunk overwrites it (idempotent upserts)."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{tenant_id}/{doc_id}/{chunk_no}"))


@dataclass(frozen=True)
class IndexedChunk:
    tenant_id: str
    doc_id: str
    chunk_no: int
    page: int | None
    section: str | None
    content: str
    content_hash: str
    embedding_model: str
    embedding: Sequence[float]


def ensure_collection(client: QdrantClient, name: str, dimensions: int) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE: models.VectorParams(
                size=dimensions,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=64),
            )
        },
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    # is_tenant co-locates each tenant's points on disk and lets Qdrant build
    # per-tenant HNSW sub-graphs, so filtered search stays fast and exact.
    client.create_payload_index(
        collection_name=name,
        field_name="tenant_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD, is_tenant=True
        ),
    )


def upsert_chunks(
    client: QdrantClient,
    name: str,
    chunks: Sequence[IndexedChunk],
    *,
    avgdl: float,
    batch_size: int = 256,
) -> int:
    written = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        points = []
        for chunk in batch:
            sparse = bm25_document_vector(chunk.content, avgdl=avgdl)
            points.append(
                models.PointStruct(
                    id=point_id(chunk.tenant_id, chunk.doc_id, chunk.chunk_no),
                    vector={
                        DENSE: [float(value) for value in chunk.embedding],
                        SPARSE: _sparse(sparse),
                    },
                    payload={
                        "tenant_id": chunk.tenant_id,
                        "doc_id": chunk.doc_id,
                        "chunk_no": chunk.chunk_no,
                        "page": chunk.page,
                        "section": chunk.section,
                        "content": chunk.content,
                        "content_hash": chunk.content_hash,
                        "embedding_model": chunk.embedding_model,
                    },
                )
            )
        client.upsert(collection_name=name, points=points, wait=True)
        written += len(points)
    return written


def tenant_filter(tenant_id: str, embedding_model: str | None = None) -> models.Filter:
    must: list[models.Condition] = [
        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
    ]
    if embedding_model is not None:
        must.append(
            models.FieldCondition(
                key="embedding_model", match=models.MatchValue(value=embedding_model)
            )
        )
    return models.Filter(must=must)


def search(
    client: QdrantClient,
    name: str,
    *,
    mode: QdrantMode,
    tenant_id: str,
    query_text: str,
    query_vector: Sequence[float] | None,
    k: int = 10,
    candidates: int = 50,
    fusion: Literal["rrf", "dbsf"] = "rrf",
    rrf_k: int = 60,
    embedding_model: str | None = None,
    hnsw_ef: int | None = None,
) -> list[SearchHit]:
    flt = tenant_filter(tenant_id, embedding_model)
    # HNSW returns at most ef results; keep ef >= the number of results asked for.
    dense_params = models.SearchParams(hnsw_ef=max(hnsw_ef or 64, k))
    prefetch_params = models.SearchParams(hnsw_ef=max(hnsw_ef or 64, candidates))
    # IDF statistics are shard-wide by default, i.e. blended across tenants:
    # a term's rarity would leak other tenants' corpus and skew ranking.
    # Scope the statistics corpus to this tenant with the "idf" search param.
    sparse_params = models.SearchParams(idf=models.IdfCorpusParams(corpus=tenant_filter(tenant_id)))
    sparse_query = _sparse(bm25_query_vector(query_text))
    if mode == "sparse":
        if not sparse_query.indices:
            return []
        response = client.query_points(
            collection_name=name,
            query=sparse_query,
            using=SPARSE,
            query_filter=flt,
            search_params=sparse_params,
            limit=k,
            with_payload=True,
        )
        return [_hit(point) for point in response.points]

    if query_vector is None:
        raise ValueError("dense and hybrid modes need a query vector")
    dense_query = [float(value) for value in query_vector]
    if mode == "dense":
        response = client.query_points(
            collection_name=name,
            query=dense_query,
            using=DENSE,
            query_filter=flt,
            search_params=dense_params,
            limit=k,
            with_payload=True,
        )
        return [_hit(point) for point in response.points]

    prefetch = [
        models.Prefetch(
            query=dense_query, using=DENSE, filter=flt, limit=candidates, params=prefetch_params
        ),
    ]
    if sparse_query.indices:
        prefetch.append(
            models.Prefetch(
                query=sparse_query,
                using=SPARSE,
                filter=flt,
                limit=candidates,
                params=sparse_params,
            )
        )
    fusion_query: Any
    if fusion == "dbsf":
        fusion_query = models.FusionQuery(fusion=models.Fusion.DBSF)
    else:
        fusion_query = models.RrfQuery(rrf=models.Rrf(k=rrf_k))
    response = client.query_points(
        collection_name=name,
        prefetch=prefetch,
        query=fusion_query,
        query_filter=flt,
        limit=k,
        with_payload=True,
    )
    return [_hit(point) for point in response.points]


def count(client: QdrantClient, name: str, tenant_id: str) -> int:
    if not client.collection_exists(name):
        return 0
    return int(client.count(name, count_filter=tenant_filter(tenant_id), exact=True).count)


def _sparse(vector: SparseVector) -> models.SparseVector:
    return models.SparseVector(indices=vector.indices, values=vector.values)


def _hit(point: models.ScoredPoint) -> SearchHit:
    payload = point.payload or {}
    page = payload.get("page")
    section = payload.get("section")
    return SearchHit(
        doc_id=str(payload.get("doc_id")),
        chunk_no=int(payload.get("chunk_no", -1)),
        page=int(page) if page is not None else None,
        section=str(section) if section is not None else None,
        text=str(payload.get("content", "")),
        content_hash=str(payload.get("content_hash", "")),
        score=float(point.score),
    )
