"""End-to-end ingest and search helpers used by the CLI and the HTTP API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from opentelemetry.trace import SpanKind

from agentic_rag_platform.ingest.chunker import chunk_document
from agentic_rag_platform.ingest.embedder import Embedder
from agentic_rag_platform.ingest.pdf import PdfLoadError, load_pdf
from agentic_rag_platform.ingest.store import (
    connect,
    count_chunks,
    delete_chunks_after,
    existing_chunks,
    upsert_chunks,
    upsert_document,
)
from agentic_rag_platform.ingest.store import (
    search as store_search,
)
from agentic_rag_platform.ingest.types import Chunk, SearchHit, needs_embedding
from agentic_rag_platform.telemetry.tracing import (
    EMBEDDINGS_DIM,
    OP_NAME,
    REQUEST_MODEL,
    RETRIEVAL_TOP_K,
    TENANT_ID,
    document_refs,
    start_span,
)


@dataclass(frozen=True)
class IngestDocumentResult:
    doc_id: str
    pages: int
    chunks: int
    embedded: int
    skipped: int
    deleted: int


@dataclass(frozen=True)
class IngestSummary:
    tenant_id: str
    documents: tuple[IngestDocumentResult, ...]

    @property
    def chunks(self) -> int:
        return sum(item.chunks for item in self.documents)

    @property
    def embedded(self) -> int:
        return sum(item.embedded for item in self.documents)


def iter_pdf_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() == ".pdf")
    raise FileNotFoundError(f"Path not found: {source}")


def doc_id_for(path: Path, explicit: str | None = None) -> str:
    return explicit or path.stem


def ingest_paths(
    sources: list[Path],
    *,
    tenant_id: str,
    database_url: str,
    embedder: Embedder,
    doc_id: str | None = None,
) -> IngestSummary:
    if doc_id is not None and len(sources) != 1:
        raise ValueError("doc_id can only be set when ingesting a single file")

    with start_span(
        "ingest",
        attributes={
            OP_NAME: "ingest",
            TENANT_ID: tenant_id,
            REQUEST_MODEL: embedder.name,
        },
    ) as span:
        summary = _ingest_paths(
            sources,
            tenant_id=tenant_id,
            database_url=database_url,
            embedder=embedder,
            doc_id=doc_id,
        )
        span.set_attribute("ingest.document.count", len(summary.documents))
        span.set_attribute("ingest.chunk.count", summary.chunks)
        span.set_attribute("ingest.embedded.count", summary.embedded)
        return summary


def _ingest_paths(
    sources: list[Path],
    *,
    tenant_id: str,
    database_url: str,
    embedder: Embedder,
    doc_id: str | None = None,
) -> IngestSummary:
    results: list[IngestDocumentResult] = []
    with connect(database_url) as conn:
        for source in sources:
            parsed = load_pdf(source)
            this_id = doc_id_for(source, doc_id)
            chunks = chunk_document(parsed, doc_id=this_id)
            upsert_document(
                conn,
                tenant_id=tenant_id,
                doc_id=this_id,
                title=parsed.title,
                source_uri=parsed.source_path,
                page_count=len(parsed.pages),
            )
            stored = existing_chunks(conn, tenant_id=tenant_id, doc_id=this_id)
            to_embed: list[Chunk] = []
            embeddings: list[list[float] | None] = []
            for chunk in chunks:
                if needs_embedding(
                    stored.get(chunk.chunk_no),
                    content_hash=chunk.content_hash,
                    embedding_model=embedder.name,
                ):
                    to_embed.append(chunk)
                    embeddings.append(None)  # placeholder, filled below
                else:
                    embeddings.append(None)

            if to_embed:
                vectors = _embed_passages(embedder, [chunk.text for chunk in to_embed])
                if len(vectors) != len(to_embed):
                    raise RuntimeError("embedder returned the wrong number of vectors")
                vector_iter = iter(vectors)
                embeddings = [
                    next(vector_iter) if chunk in to_embed else None
                    for chunk, _existing in zip(chunks, embeddings, strict=True)
                ]

            if chunks:
                upsert_chunks(
                    conn,
                    tenant_id=tenant_id,
                    chunks=chunks,
                    embeddings=embeddings,
                    embedding_model=embedder.name,
                )
                deleted = delete_chunks_after(
                    conn,
                    tenant_id=tenant_id,
                    doc_id=this_id,
                    last_chunk_no=chunks[-1].chunk_no,
                )
            else:
                deleted = delete_chunks_after(
                    conn, tenant_id=tenant_id, doc_id=this_id, last_chunk_no=-1
                )

            skipped = len(chunks) - len(to_embed)
            results.append(
                IngestDocumentResult(
                    doc_id=this_id,
                    pages=len(parsed.pages),
                    chunks=len(chunks),
                    embedded=len(to_embed),
                    skipped=skipped,
                    deleted=deleted,
                )
            )
            conn.commit()
    return IngestSummary(tenant_id=tenant_id, documents=tuple(results))


def search_query(
    query: str,
    *,
    tenant_id: str,
    database_url: str,
    embedder: Embedder,
    k: int = 10,
    ef_search: int | None = None,
) -> list[SearchHit]:
    if not query.strip():
        raise ValueError("query must not be empty")
    with start_span(
        "retrieval",
        attributes={
            OP_NAME: "retrieval",
            RETRIEVAL_TOP_K: k,
            TENANT_ID: tenant_id,
            "gen_ai.data_source.id": "pgvector",
            REQUEST_MODEL: embedder.name,
        },
    ) as span:
        vectors = _embed_queries(embedder, [query])
        if not vectors:
            raise RuntimeError("embedder returned no query vector")
        with connect(database_url) as conn:
            hits = store_search(
                conn,
                tenant_id=tenant_id,
                query=vectors[0],
                k=k,
                embedding_model=embedder.name,
                ef_search=ef_search,
                backend="hnsw",
            )
        # id + score only; never attach chunk text (may contain sensitive content).
        from agentic_rag_platform.settings import get_settings

        if not get_settings().otel_redact_content:
            span.set_attribute("gen_ai.retrieval.query.text", query)

        span.set_attribute("gen_ai.retrieval.document.count", len(hits))
        refs = document_refs(hits)
        if refs:
            span.set_attribute("gen_ai.retrieval.document.ids", tuple(r["id"] for r in refs))
            span.set_attribute(
                "gen_ai.retrieval.document.scores",
                json.dumps([r["score"] for r in refs]),
            )
        return hits


def _embed_queries(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    return _embed(embedder, texts, "query")


def _embed_passages(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    return _embed(embedder, texts, "passage")


def _embed(embedder: Embedder, texts: list[str], kind: str) -> list[list[float]]:
    attrs: dict[str, str | int] = {
        OP_NAME: "embeddings",
        REQUEST_MODEL: embedder.name,
        EMBEDDINGS_DIM: embedder.dimensions,
        "gen_ai.embeddings.input.count": len(texts),
        "gen_ai.embeddings.kind": kind,
    }
    with start_span(f"embeddings {embedder.name}", kind=SpanKind.CLIENT, attributes=attrs):
        if kind == "query":
            return embedder.embed_queries(texts)
        return embedder.embed_passages(texts)


def stored_chunk_count(database_url: str, tenant_id: str) -> int:
    with connect(database_url) as conn:
        return count_chunks(conn, tenant_id=tenant_id)


# Re-export for callers that want a typed error.
__all__ = [
    "IngestDocumentResult",
    "IngestSummary",
    "PdfLoadError",
    "doc_id_for",
    "ingest_paths",
    "iter_pdf_paths",
    "search_query",
    "stored_chunk_count",
]
