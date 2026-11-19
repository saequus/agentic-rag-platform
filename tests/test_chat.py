from collections.abc import Callable

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from agentic_rag_platform.main import app, provider_dep
from agentic_rag_platform.tools.fx_rate import NBP_RATE_URL
from tests.fakes import ScriptedProvider, answer, tool_request


def test_chat_returns_structured_answer(
    client: TestClient, override_provider: Callable[[object], None]
) -> None:
    override_provider(
        ScriptedProvider([answer("Warsaw is the capital of Poland.", confidence=0.85)])
    )
    response = client.post("/chat", json={"message": "What is the capital of Poland?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Warsaw is the capital of Poland."
    assert body["confidence"] == 0.85
    assert body["provider"] == "fake"
    assert body["model"] == "fake-model"
    assert body["tool_calls"] == []


@respx.mock
def test_chat_executes_fx_tool(
    client: TestClient, override_provider: Callable[[object], None]
) -> None:
    respx.get(NBP_RATE_URL.format(code="eur")).mock(
        return_value=httpx.Response(
            200,
            json={"rates": [{"effectiveDate": "2026-11-19", "mid": 4.3}]},
        )
    )
    override_provider(
        ScriptedProvider(
            [
                tool_request("t1", "get_fx_rate", base="EUR", quote="PLN"),
                answer("1 EUR is 4.3 PLN.", citations=["https://api.nbp.pl"], confidence=0.9),
            ]
        )
    )
    response = client.post("/chat", json={"message": "EUR to PLN?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "1 EUR is 4.3 PLN."
    assert body["tool_calls"][0]["name"] == "get_fx_rate"
    assert "4.3" in body["tool_calls"][0]["result"]


def test_chat_rejects_empty_message(
    client: TestClient, override_provider: Callable[[object], None]
) -> None:
    override_provider(ScriptedProvider([answer("should not be called")]))
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_without_provider_key_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    # No override: factory tries to build the real xAI client and must fail
    # closed when the key is missing, without making a network call.
    # An empty env var wins over `.env`, so a local key cannot leak into this test.
    monkeypatch.setenv("XAI_API_KEY", "")
    from agentic_rag_platform.settings import get_settings

    get_settings.cache_clear()
    # Drop any leftover override from previous tests.
    app.dependency_overrides.pop(provider_dep, None)
    local = TestClient(app)
    response = local.post("/chat", json={"message": "hello"})
    assert response.status_code == 503
    assert "XAI_API_KEY" in response.json()["detail"]
