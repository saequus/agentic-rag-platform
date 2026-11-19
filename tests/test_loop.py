import json

import httpx
import pytest
import respx

from agentic_rag_platform.agent.loop import run_agent
from agentic_rag_platform.providers.base import ProviderResponse
from agentic_rag_platform.schemas import Answer
from agentic_rag_platform.tools.fx_rate import NBP_RATE_URL
from tests.fakes import ScriptedProvider, answer, tool_request


@pytest.mark.asyncio
async def test_direct_structured_answer_skips_tools() -> None:
    provider = ScriptedProvider([answer("hello", confidence=0.8)])
    result = await run_agent(provider, "Say hello", tools=[])
    assert result.answer.answer == "hello"
    assert result.answer.confidence == 0.8
    assert result.tool_calls == []
    assert len(provider.calls) == 1


@respx.mock
@pytest.mark.asyncio
async def test_tool_loop_calls_get_fx_rate_then_structures() -> None:
    respx.get(NBP_RATE_URL.format(code="usd")).mock(
        return_value=httpx.Response(
            200,
            json={"rates": [{"effectiveDate": "2026-11-19", "mid": 4.25}]},
        )
    )
    provider = ScriptedProvider(
        [
            tool_request("call_1", "get_fx_rate", base="USD", quote="PLN"),
            answer(
                "1 USD is 4.25 PLN.",
                citations=["https://api.nbp.pl/api/exchangerates/rates/a/usd/"],
                confidence=0.93,
            ),
        ]
    )
    result = await run_agent(provider, "How many PLN is one US dollar?")
    assert result.answer.answer.startswith("1 USD")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_fx_rate"
    payload = json.loads(result.tool_calls[0].result)
    assert payload["rate"] == 4.25
    # First complete had tools; second produced the structured answer (possibly
    # without needing a third schema-only turn).
    assert provider.calls[0]["tools"] == ["get_fx_rate"]


@pytest.mark.asyncio
async def test_max_iterations_forces_final_schema_call() -> None:
    provider = ScriptedProvider(
        [
            tool_request("c1", "missing_tool"),
            tool_request("c2", "missing_tool"),
            answer("giving up", confidence=0.3),
        ]
    )
    result = await run_agent(provider, "loop please", max_iterations=2)
    assert result.answer.answer == "giving up"
    assert len(result.tool_calls) == 2
    # Two tool iterations, then one forced structured completion.
    assert len(provider.calls) == 3
    assert provider.calls[-1]["response_schema"] is Answer
    assert provider.calls[-1]["tools"] == []


@pytest.mark.asyncio
async def test_plain_text_final_is_wrapped_with_low_confidence() -> None:
    provider = ScriptedProvider(
        [
            ProviderResponse(content="Just some free text, not JSON."),
            ProviderResponse(content="Still not a schema."),
        ]
    )
    result = await run_agent(provider, "hi", tools=[])
    assert result.answer.answer == "Still not a schema."
    assert result.answer.confidence == 0.2
    assert result.answer.citations == []
