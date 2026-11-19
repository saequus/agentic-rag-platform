"""Look up a foreign-exchange mid rate with timeout and fallbacks.

Primary source is the free NBP table A API, which publishes official mid
rates against PLN (https://api.nbp.pl). Cross rates that do not involve PLN
are computed via PLN. If NBP is unavailable or does not list a currency,
we fall back to Frankfurter (https://api.frankfurter.app, ECB reference rates).
"""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

NBP_RATE_URL = "https://api.nbp.pl/api/exchangerates/rates/a/{code}/"
FRANKFURTER_URL = "https://api.frankfurter.app/latest"
DEFAULT_TIMEOUT_SECONDS = 8.0


class FxRateArgs(BaseModel):
    base: str = Field(..., description="ISO 4217 source currency, e.g. USD")
    quote: str = Field(..., description="ISO 4217 target currency, e.g. PLN")

    @field_validator("base", "quote")
    @classmethod
    def iso_code(cls, value: str) -> str:
        code = value.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError("currency codes must be 3-letter ISO 4217 symbols")
        return code


async def get_fx_rate(
    base: str,
    quote: str,
    *,
    http_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Return a mid rate for ``base``/``quote`` or a structured error."""
    args = FxRateArgs(base=base, quote=quote)
    if args.base == args.quote:
        return {
            "base": args.base,
            "quote": args.quote,
            "rate": 1.0,
            "as_of": None,
            "source": "identity",
            "source_url": None,
        }

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(http_timeout))
    try:
        try:
            return await _from_nbp(client, args.base, args.quote)
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as nbp_error:
            try:
                result = await _from_frankfurter(client, args.base, args.quote)
                result["nbp_fallback_reason"] = str(nbp_error)
                return result
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as frank_error:
                return {
                    "base": args.base,
                    "quote": args.quote,
                    "error": "Could not fetch an FX rate from NBP or Frankfurter.",
                    "nbp_error": str(nbp_error),
                    "frankfurter_error": str(frank_error),
                }
    finally:
        if owns_client:
            await client.aclose()


async def _from_nbp(client: httpx.AsyncClient, base: str, quote: str) -> dict[str, Any]:
    if base == "PLN":
        mid, as_of, url = await _nbp_mid(client, quote)
        return _ok(base, quote, 1.0 / mid, as_of, "NBP", url)
    if quote == "PLN":
        mid, as_of, url = await _nbp_mid(client, base)
        return _ok(base, quote, mid, as_of, "NBP", url)

    base_mid, base_date, base_url = await _nbp_mid(client, base)
    quote_mid, quote_date, quote_url = await _nbp_mid(client, quote)
    if quote_mid == 0:
        raise ValueError(f"NBP mid rate for {quote} was zero")
    as_of = min(base_date, quote_date)
    return _ok(
        base,
        quote,
        base_mid / quote_mid,
        as_of,
        "NBP (cross via PLN)",
        f"{base_url} | {quote_url}",
    )


async def _nbp_mid(client: httpx.AsyncClient, code: str) -> tuple[float, str, str]:
    url = NBP_RATE_URL.format(code=code.lower())
    response = await client.get(
        url, params={"format": "json"}, headers={"Accept": "application/json"}
    )
    response.raise_for_status()
    payload = response.json()
    rate = payload["rates"][0]
    mid = float(rate["mid"])
    if mid <= 0:
        raise ValueError(f"invalid NBP mid rate for {code}: {mid}")
    return mid, str(rate["effectiveDate"]), str(response.url)


async def _from_frankfurter(client: httpx.AsyncClient, base: str, quote: str) -> dict[str, Any]:
    response = await client.get(FRANKFURTER_URL, params={"from": base, "to": quote})
    response.raise_for_status()
    payload = response.json()
    rate = float(payload["rates"][quote])
    return _ok(base, quote, rate, str(payload.get("date")), "Frankfurter (ECB)", str(response.url))


def _ok(
    base: str, quote: str, rate: float, as_of: str | None, source: str, source_url: str
) -> dict[str, Any]:
    return {
        "base": base,
        "quote": quote,
        "rate": rate,
        "as_of": as_of,
        "source": source,
        "source_url": source_url,
    }
