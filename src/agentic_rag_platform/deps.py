"""Shared FastAPI dependencies (settings and the configured LLM provider)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException

from agentic_rag_platform.providers.base import LLMProvider, ProviderError
from agentic_rag_platform.providers.factory import get_provider
from agentic_rag_platform.settings import Settings, get_settings


def settings_dep() -> Settings:
    return get_settings()


def provider_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> LLMProvider:
    try:
        return get_provider(settings)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
