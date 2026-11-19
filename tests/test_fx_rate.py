import httpx
import pytest
import respx

from agentic_rag_platform.tools.fx_rate import FRANKFURTER_URL, NBP_RATE_URL, get_fx_rate
from agentic_rag_platform.tools.registry import execute_tool


def nbp_url(code: str) -> str:
    return NBP_RATE_URL.format(code=code.lower())


@respx.mock
@pytest.mark.asyncio
async def test_usd_pln_from_nbp() -> None:
    respx.get(nbp_url("usd")).mock(
        return_value=httpx.Response(
            200,
            json={
                "table": "A",
                "code": "USD",
                "rates": [{"no": "224/A/NBP/2026", "effectiveDate": "2026-11-19", "mid": 4.0}],
            },
        )
    )
    result = await get_fx_rate("usd", "pln")
    assert result["rate"] == 4.0
    assert result["source"] == "NBP"
    assert result["as_of"] == "2026-11-19"
    assert "nbp.pl" in result["source_url"]


@respx.mock
@pytest.mark.asyncio
async def test_pln_usd_inverts_nbp_mid() -> None:
    respx.get(nbp_url("usd")).mock(
        return_value=httpx.Response(
            200,
            json={"rates": [{"effectiveDate": "2026-11-19", "mid": 4.0}]},
        )
    )
    result = await get_fx_rate("PLN", "USD")
    assert result["rate"] == 0.25


@respx.mock
@pytest.mark.asyncio
async def test_cross_rate_via_pln() -> None:
    respx.get(nbp_url("usd")).mock(
        return_value=httpx.Response(
            200,
            json={"rates": [{"effectiveDate": "2026-11-18", "mid": 4.0}]},
        )
    )
    respx.get(nbp_url("eur")).mock(
        return_value=httpx.Response(
            200,
            json={"rates": [{"effectiveDate": "2026-11-19", "mid": 2.0}]},
        )
    )
    result = await get_fx_rate("USD", "EUR")
    assert result["rate"] == 2.0
    assert result["as_of"] == "2026-11-18"
    assert "cross" in result["source"]


@respx.mock
@pytest.mark.asyncio
async def test_frankfurter_fallback_when_nbp_fails() -> None:
    respx.get(nbp_url("usd")).mock(return_value=httpx.Response(503))
    respx.get(FRANKFURTER_URL).mock(
        return_value=httpx.Response(
            200,
            json={"amount": 1.0, "base": "USD", "date": "2026-11-18", "rates": {"PLN": 3.9}},
        )
    )
    result = await get_fx_rate("USD", "PLN")
    assert result["rate"] == 3.9
    assert result["source"].startswith("Frankfurter")
    assert "nbp_fallback_reason" in result


@respx.mock
@pytest.mark.asyncio
async def test_both_sources_fail_returns_error() -> None:
    respx.get(nbp_url("usd")).mock(return_value=httpx.Response(500))
    respx.get(FRANKFURTER_URL).mock(return_value=httpx.Response(500))
    result = await get_fx_rate("USD", "PLN")
    assert "error" in result
    assert "nbp_error" in result
    assert "frankfurter_error" in result


@pytest.mark.asyncio
async def test_identity_rate() -> None:
    result = await get_fx_rate("EUR", "EUR")
    assert result["rate"] == 1.0
    assert result["source"] == "identity"


@pytest.mark.asyncio
async def test_execute_tool_rejects_bad_code() -> None:
    payload = await execute_tool("get_fx_rate", {"base": "US", "quote": "PLN"})
    assert "error" in payload


@pytest.mark.asyncio
async def test_execute_unknown_tool() -> None:
    payload = await execute_tool("not_a_tool", {})
    assert "Unknown tool" in payload
