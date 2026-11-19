"""OpenAI provider (official ``openai`` SDK).

This adapter exists so the OpenAI tool-calling and structured-output APIs can
be learned and compared with xAI. It is selected with ``LLM_PROVIDER=openai``.

Tool calling
------------
OpenAI wraps each tool as ``{"type": "function", "function": {name, description,
parameters, strict}}`` (https://platform.openai.com/docs/guides/function-calling).

The model returns ``choices[0].message.tool_calls``. You append that assistant
message, then a ``role="tool"`` message per call with ``tool_call_id``. The
model never executes the function; your process does.

``strict`` is **opt-in** on OpenAI (default false). We set ``strict=True`` so
tool arguments are grammar-constrained, closer to xAI (where strict is always
on) and to Anthropic's ``strict: true``.

Structured outputs
------------------
``client.chat.completions.parse(..., response_format=Answer)`` uses
``response_format.json_schema`` with ``strict: true`` and returns
``message.parsed`` as a Pydantic instance
(https://platform.openai.com/docs/guides/structured-outputs).

This is stronger than JSON mode (``{"type": "json_object"}``), which only
guarantees valid JSON, not a schema. Asking "reply as JSON" in the prompt
guarantees neither.

OpenAI also has a Responses API (``client.responses.parse``) that is the newer
surface; Chat Completions is used here because it is still the API most
interviewers walk through for tool calling.
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from agentic_rag_platform.providers.base import ProviderError, ProviderResponse
from agentic_rag_platform.schemas import Message, ToolCall
from agentic_rag_platform.tools.registry import ToolSpec


class OpenAIProvider:
    """OpenAI Chat Completions adapter with tools + ``parse()``."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self.model = model
        self.temperature = temperature
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        temp = self.temperature if temperature is None else temperature
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_message(m) for m in messages],
            "temperature": temp,
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(spec) for spec in tools]

        if response_schema is not None:
            completion = await self._client.chat.completions.parse(
                response_format=response_schema,
                **kwargs,
            )
            choice = completion.choices[0].message
            return ProviderResponse(
                content=choice.content,
                tool_calls=_from_openai_tool_calls(choice.tool_calls),
                parsed=choice.parsed,
            )

        completion = await self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0].message
        return ProviderResponse(
            content=choice.content,
            tool_calls=_from_openai_tool_calls(choice.tool_calls),
        )


def _to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
            "strict": True,
        },
    }


def _to_openai_message(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content or "",
        }
    if message.role == "assistant" and message.tool_calls:
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ],
        }
        return payload
    return {"role": message.role, "content": message.content or ""}


def _from_openai_tool_calls(calls: Any) -> list[ToolCall]:
    result: list[ToolCall] = []
    for call in calls or []:
        raw_args = call.function.arguments
        try:
            parsed_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            parsed_args = {"_raw": raw_args}
        if not isinstance(parsed_args, dict):
            parsed_args = {"_raw": raw_args}
        result.append(ToolCall(id=call.id, name=call.function.name, arguments=parsed_args))
    return result
