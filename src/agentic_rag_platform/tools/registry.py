"""Tool specifications and a small dispatcher."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from agentic_rag_platform.telemetry.tracing import (
    AGENT_NAME,
    AGENT_NAME_ATTR,
    OP_NAME,
    TOOL_CALL_ID,
    TOOL_NAME,
    TOOL_TYPE,
    set_status_error,
    start_span,
)
from agentic_rag_platform.tools.fx_rate import FxRateArgs, get_fx_rate

ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSpec:
    """JSON-schema tool description plus the local async handler."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler


async def _get_fx_rate_handler(**kwargs: Any) -> dict[str, Any]:
    args = FxRateArgs.model_validate(kwargs)
    return await get_fx_rate(args.base, args.quote)


GET_FX_RATE = ToolSpec(
    name="get_fx_rate",
    description=(
        "Get the latest published mid foreign-exchange rate between two currencies. "
        "Use this for any currency conversion or FX question. "
        "Arguments are ISO 4217 codes (USD, EUR, PLN, GBP, ...). "
        "Rates against PLN come from the National Bank of Poland; other pairs may "
        "be computed as a cross rate or served from ECB reference rates."
    ),
    parameters={
        "type": "object",
        "properties": {
            "base": {
                "type": "string",
                "description": "ISO 4217 source currency code, e.g. USD",
            },
            "quote": {
                "type": "string",
                "description": "ISO 4217 target currency code, e.g. PLN",
            },
        },
        "required": ["base", "quote"],
        "additionalProperties": False,
    },
    handler=_get_fx_rate_handler,
)


def default_tools() -> list[ToolSpec]:
    return [GET_FX_RATE]


def get_tool(name: str, tools: list[ToolSpec] | None = None) -> ToolSpec | None:
    catalog = tools if tools is not None else default_tools()
    return next((spec for spec in catalog if spec.name == name), None)


async def execute_tool(
    name: str,
    arguments: Mapping[str, Any],
    tools: list[ToolSpec] | None = None,
    tool_call_id: str | None = None,
) -> str:
    """Run a named tool and always return a JSON string (including errors)."""
    with start_span(
        f"execute_tool {name}",
        attributes={
            OP_NAME: "execute_tool",
            TOOL_NAME: name,
            TOOL_TYPE: "function",
            AGENT_NAME_ATTR: AGENT_NAME,
            TOOL_CALL_ID: tool_call_id,
        },
    ) as span:
        return await _execute_tool_body(name, arguments, tools, span)


async def _execute_tool_body(
    name: str,
    arguments: Mapping[str, Any],
    tools: list[ToolSpec] | None,
    span: Any,
) -> str:
    spec = get_tool(name, tools)
    if spec is None:
        payload = json.dumps({"error": f"Unknown tool: {name}"})
        set_status_error(span, f"Unknown tool: {name}", "unknown_tool")
        return payload
    try:
        result = await spec.handler(**dict(arguments))
    except ValidationError as exc:
        set_status_error(span, "Invalid tool arguments", "ValidationError")
        return json.dumps(
            {"error": "Invalid tool arguments", "details": exc.errors(include_context=False)},
            default=str,
        )
    except TypeError as exc:
        set_status_error(span, str(exc), "TypeError")
        return json.dumps({"error": f"Invalid tool arguments: {exc}"})
    except Exception as exc:  # noqa: BLE001 — never break the agent loop
        set_status_error(span, str(exc), type(exc).__name__)
        return json.dumps({"error": f"Tool {name} failed: {exc}"})
    return json.dumps(result, default=str)
