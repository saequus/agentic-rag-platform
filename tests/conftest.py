"""Shared fixtures. Providers and HTTP are mocked; CI never calls a real LLM."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from agentic_rag_platform.graph.checkpoint import reset_agent_runtime
from agentic_rag_platform.ingest.embedder import HashEmbedder
from agentic_rag_platform.ingest.routes import embedder_dep
from agentic_rag_platform.main import app, provider_dep
from agentic_rag_platform.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Tests assert the code defaults (xAI / grok-4.7) and must not call a real
    # LLM. A developer's .env is loaded by Settings, and env vars win over that
    # file, so pin the provider here. An empty key overrides a key in .env.
    monkeypatch.setenv("LLM_PROVIDER", "xai")
    monkeypatch.setenv("XAI_MODEL", "grok-4.7")
    monkeypatch.setenv("XAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    # Day 4: in-memory checkpoints in unit tests (the Postgres saver has its own
    # integration test). A fresh compiled graph per test = no shared threads.
    monkeypatch.setenv("AGENT_CHECKPOINTER", "memory")
    get_settings.cache_clear()
    reset_agent_runtime()
    app.dependency_overrides[embedder_dep] = lambda: HashEmbedder()
    yield
    get_settings.cache_clear()
    reset_agent_runtime()
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def override_provider() -> Callable[[object], None]:
    def _override(provider: object) -> None:
        app.dependency_overrides[provider_dep] = lambda: provider

    return _override
