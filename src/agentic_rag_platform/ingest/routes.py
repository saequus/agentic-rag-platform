"""HTTP routes for Day 2: ``POST /ingest`` and ``POST /search``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agentic_rag_platform.ingest.embedder import Embedder, get_embedder
from agentic_rag_platform.ingest.pdf import PdfLoadError
from agentic_rag_platform.ingest.pipeline import ingest_paths, iter_pdf_paths, search_query
from agentic_rag_platform.ingest.types import SearchHit
from agentic_rag_platform.settings import Settings, get_settings

router = APIRouter()


class IngestRequest(BaseModel):
    """Ingest a local PDF file or a directory of PDFs.

    ``source_path`` is resolved on the machine running the API (this is a lab
    endpoint, not a multi-tenant upload API).
    """

    source_path: str = Field(..., min_length=1)
    tenant_id: str | None = None
    doc_id: str | None = Field(
        default=None,
        description="Override doc_id; only valid when source_path is a single file.",
    )


class IngestDocumentBody(BaseModel):
    doc_id: str
    pages: int
    chunks: int
    embedded: int
    skipped: int
    deleted: int


class IngestResponse(BaseModel):
    tenant_id: str
    documents: list[IngestDocumentBody]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tenant_id: str | None = None
    k: int = Field(default=10, ge=1, le=100)
    ef_search: int | None = Field(default=None, ge=1, le=800)


class SearchHitBody(BaseModel):
    doc_id: str
    chunk_no: int
    page: int | None
    section: str | None
    text: str
    content_hash: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHitBody]
    model: str
    tenant_id: str


def settings_dep() -> Settings:
    return get_settings()


def embedder_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> Embedder:
    try:
        return get_embedder(settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    request: IngestRequest,
    settings: Annotated[Settings, Depends(settings_dep)],
    embedder: Annotated[Embedder, Depends(embedder_dep)],
) -> IngestResponse:
    tenant_id = request.tenant_id or settings.default_tenant_id
    try:
        sources = iter_pdf_paths(Path(request.source_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not sources:
        raise HTTPException(status_code=404, detail="No PDF files found at source_path")
    try:
        summary = ingest_paths(
            sources,
            tenant_id=tenant_id,
            database_url=settings.database_url,
            embedder=embedder,
            doc_id=request.doc_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PdfLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (OSError, psycopg.Error) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    return IngestResponse(
        tenant_id=summary.tenant_id,
        documents=[
            IngestDocumentBody(
                doc_id=item.doc_id,
                pages=item.pages,
                chunks=item.chunks,
                embedded=item.embedded,
                skipped=item.skipped,
                deleted=item.deleted,
            )
            for item in summary.documents
        ],
    )


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    settings: Annotated[Settings, Depends(settings_dep)],
    embedder: Annotated[Embedder, Depends(embedder_dep)],
) -> SearchResponse:
    tenant_id = request.tenant_id or settings.default_tenant_id
    try:
        hits = search_query(
            request.query,
            tenant_id=tenant_id,
            database_url=settings.database_url,
            embedder=embedder,
            k=request.k,
            ef_search=request.ef_search or settings.hnsw_ef_search,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (OSError, psycopg.Error) as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc
    return SearchResponse(
        hits=[_hit_body(hit) for hit in hits],
        model=embedder.name,
        tenant_id=tenant_id,
    )


def _hit_body(hit: SearchHit) -> SearchHitBody:
    return SearchHitBody(
        doc_id=hit.doc_id,
        chunk_no=hit.chunk_no,
        page=hit.page,
        section=hit.section,
        text=hit.text,
        content_hash=hit.content_hash,
        score=hit.score,
    )
