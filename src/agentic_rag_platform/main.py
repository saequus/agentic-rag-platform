"""FastAPI gateway.

Day 1: ``GET /health`` and ``POST /chat``.
Day 2: ``POST /ingest`` and ``POST /search``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag_platform import __version__
from agentic_rag_platform.agent.loop import run_agent
from agentic_rag_platform.ingest.routes import router as ingest_router
from agentic_rag_platform.providers.base import LLMProvider, ProviderError
from agentic_rag_platform.providers.factory import get_provider
from agentic_rag_platform.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Message,
)
from agentic_rag_platform.settings import Settings, get_settings
from agentic_rag_platform.telemetry import configure_telemetry, instrument_app, shutdown_telemetry


def _exclude_otel_path(scope: MutableMapping[str, Any]) -> bool:
    """Skip FastAPI-native spans for probes and docs."""
    return scope.get("path") in {"/health", "/docs", "/openapi.json", "/redoc"}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    if configure_telemetry(settings) is not None:
        instrument_app(_app)
    yield
    shutdown_telemetry()


app = FastAPI(
    title="agentic-rag-platform",
    version=__version__,
    description=(
        "On-prem-capable agentic RAG platform. Day 1 is the LLM gateway "
        "(tool-calling loop, structured outputs, default xAI Grok). "
        "Day 2 adds PDF ingestion, embeddings and pgvector search. "
        "OpenTelemetry traces (GenAI semantic conventions) are emitted when "
        "an OTLP exporter is configured."
    ),
    lifespan=lifespan,
    # FastAPI 0.165+ has native OpenTelemetry. We use it for HTTP server spans
    # and keep auto-export off so setup.py owns the TracerProvider / exporters.
    telemetry={
        "tracing": True,
        "metrics": False,
        "logs": False,
        "operation_spans": False,
        "auto_configure": False,
        "exclude": _exclude_otel_path,
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)


def settings_dep() -> Settings:
    return get_settings()


def provider_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> LLMProvider:
    try:
        return get_provider(settings)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(settings_dep)]) -> HealthResponse:
    return HealthResponse(status="ok", provider=settings.llm_provider, model=settings.model_name)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Annotated[Settings, Depends(settings_dep)],
    provider: Annotated[LLMProvider, Depends(provider_dep)],
) -> ChatResponse:
    history = [Message(role=turn.role, content=turn.content) for turn in request.history]
    try:
        result = await run_agent(
            provider,
            request.message,
            history=history,
            max_iterations=settings.max_tool_iterations,
            temperature=settings.llm_temperature,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.answer.answer,
        citations=result.answer.citations,
        confidence=result.answer.confidence,
        provider=provider.name,
        model=provider.model,
        tool_calls=result.tool_calls,
    )
