"""FastAPI gateway. Day 1 exposes ``GET /health`` and ``POST /chat``."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag_platform import __version__
from agentic_rag_platform.agent.loop import run_agent
from agentic_rag_platform.providers.base import LLMProvider, ProviderError
from agentic_rag_platform.providers.factory import get_provider
from agentic_rag_platform.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    Message,
)
from agentic_rag_platform.settings import Settings, get_settings

app = FastAPI(
    title="agentic-rag-platform",
    version=__version__,
    description=(
        "On-prem-capable agentic RAG platform. Day 1 is the LLM gateway: "
        "a provider-agnostic tool-calling loop with structured outputs, "
        "defaulting to xAI Grok."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
