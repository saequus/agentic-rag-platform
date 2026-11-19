"""Shared fixtures. Providers and HTTP are mocked; CI never calls a real LLM."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from agentic_rag_platform.main import app, provider_dep
from agentic_rag_platform.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def override_provider() -> Callable[[object], None]:
    def _override(provider: object) -> None:
        app.dependency_overrides[provider_dep] = lambda: provider

    return _override
