"""HTTP tests for /ingest and /search. The pipeline and embedder are mocked."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentic_rag_platform.ingest.pipeline import IngestDocumentResult, IngestSummary
from agentic_rag_platform.ingest.routes import embedder_dep
from agentic_rag_platform.ingest.types import SearchHit
from agentic_rag_platform.main import app
from agentic_rag_platform.retrieval.service import RetrievalResult, Retriever


class _FakeEmbedder:
    name = "hash-embedder-v1"
    dimensions = 768

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]


def test_ingest_validation_error(client: TestClient) -> None:
    response = client.post("/ingest", json={"source_path": "", "tenant_id": "t"})
    assert response.status_code == 422


def test_ingest_missing_path(client: TestClient) -> None:
    app.dependency_overrides[embedder_dep] = lambda: _FakeEmbedder()
    response = client.post("/ingest", json={"source_path": "/no/such/file.pdf"})
    assert response.status_code == 404
    detail = response.json()["detail"]
    lowered = detail.lower()
    assert "not found" in lowered or "no pdf" in lowered


def test_ingest_success_is_wired(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    app.dependency_overrides[embedder_dep] = lambda: _FakeEmbedder()

    def fake_ingest_paths(*_args: object, **_kwargs: object) -> IngestSummary:
        return IngestSummary(
            tenant_id="default",
            documents=(
                IngestDocumentResult(
                    doc_id="eb202501",
                    pages=12,
                    chunks=20,
                    embedded=20,
                    skipped=0,
                    deleted=0,
                ),
            ),
        )

    from pathlib import Path as _Path

    import agentic_rag_platform.ingest.routes as routes

    monkeypatch.setattr(routes, "iter_pdf_paths", lambda _path: [_Path("doc.pdf")])
    monkeypatch.setattr(routes, "ingest_paths", fake_ingest_paths)

    response = client.post("/ingest", json={"source_path": "/tmp/docs", "tenant_id": "default"})
    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "default"
    assert body["documents"][0]["doc_id"] == "eb202501"
    assert body["documents"][0]["chunks"] == 20
    assert body["documents"][0]["embedded"] == 20


def test_search_success_is_wired(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    app.dependency_overrides[embedder_dep] = lambda: _FakeEmbedder()
    seen: dict[str, object] = {}

    def fake_retrieve(self: Retriever, query: str, **kwargs: object) -> RetrievalResult:
        seen.update(kwargs, query=query)
        return RetrievalResult(
            hits=[
                SearchHit(
                    doc_id="eb202501",
                    chunk_no=0,
                    page=3,
                    section="Summary",
                    text="Inflation is close to the 2% target.",
                    content_hash="abc",
                    score=0.91,
                )
            ],
            strategy=kwargs.get("strategy") or "hybrid",  # type: ignore[arg-type]
            backend=kwargs.get("backend") or "pgvector",  # type: ignore[arg-type]
            timings_ms={"search_ms": 1.0},
        )

    monkeypatch.setattr(Retriever, "retrieve", fake_retrieve)
    response = client.post("/search", json={"query": "euro area inflation", "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "hash-embedder-v1"
    assert body["tenant_id"] == "default"
    assert body["hits"][0]["doc_id"] == "eb202501"
    assert body["hits"][0]["section"] == "Summary"
    assert body["hits"][0]["score"] == 0.91
    assert body["strategy"] == "hybrid"
    assert body["backend"] == "pgvector"
    assert seen["strategy"] is None and seen["k"] == 5

    response = client.post(
        "/search",
        json={"query": "EDO1136", "strategy": "hybrid_rerank", "backend": "qdrant"},
    )
    assert response.status_code == 200
    assert response.json()["strategy"] == "hybrid_rerank"
    assert response.json()["backend"] == "qdrant"
    assert seen["strategy"] == "hybrid_rerank" and seen["backend"] == "qdrant"


def test_search_rejects_unknown_strategy(client: TestClient) -> None:
    response = client.post("/search", json={"query": "x", "strategy": "magic"})
    assert response.status_code == 422


def test_search_rejects_blank_query(client: TestClient) -> None:
    response = client.post("/search", json={"query": ""})
    assert response.status_code == 422
