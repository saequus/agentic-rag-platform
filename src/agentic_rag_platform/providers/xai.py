"""xAI / Grok provider (default).

Official SDK
------------
xAI documents two Python paths in the quickstart
(https://docs.x.ai/developers/quickstart):

1. The first-party SDK: ``pip install xai-sdk`` (used here).
   ``AsyncClient(api_key=..., api_host="api.x.ai")`` speaks gRPC.
2. OpenAI-compatible REST: ``OpenAI(api_key=..., base_url="https://api.x.ai/v1")``
   using the Responses API (``client.responses.create``).

This adapter uses (1) so we get native ``tool()`` / ``tool_result()`` helpers
and ``chat.parse(PydanticModel)`` for structured outputs.

Tool calling
------------
* Tools are declared with ``xai_sdk.chat.tool(name, description, parameters)``
  where ``parameters`` is a JSON Schema object
  (https://docs.x.ai/developers/tools/function-calling).
* After ``await chat.sample()``, inspect ``response.tool_calls``. Each call has
  ``id`` and ``function.{name, arguments}`` (arguments are a JSON string).
* Append the raw response, then ``tool_result(result, tool_call_id=call.id)``.
  Tool-call arguments are always schema-constrained; the ``strict`` flag is
  implicitly always true.

Structured outputs
------------------
* Preferred: ``response, parsed = await chat.parse(Answer)`` — the SDK sets
  ``response_format`` to the Pydantic model's JSON Schema and returns a typed
  instance (https://docs.x.ai/developers/model-capabilities/text/structured-outputs).
* You can also pass ``response_format=Answer`` to ``chat.create`` and parse
  ``response.content`` yourself.
* Grok 4 family models can combine tools with structured outputs.

Default model: ``grok-4.7`` (current flagship for chat / agents, Nov 2026).
Override with ``XAI_MODEL``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from xai_sdk import AsyncClient
from xai_sdk.chat import assistant, system, tool, tool_result, user
from xai_sdk.proto.v6 import chat_pb2

from agentic_rag_platform.providers.base import ProviderError, ProviderResponse
from agentic_rag_platform.schemas import Message, ToolCall
from agentic_rag_platform.tools.registry import ToolSpec


class XAIProvider:
    """Default LLM backend: official ``xai-sdk`` against Grok."""

    name = "xai"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        api_host: str = "api.x.ai",
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise ProviderError("XAI_API_KEY is not set")
        self.model = model
        self.temperature = temperature
        self._api_key = api_key
        self._api_host = api_host
        self._timeout = timeout
        # Opened on first ``complete``. See ``_client_on_running_loop``.
        self._client: AsyncClient | None = None

    def _client_on_running_loop(self) -> AsyncClient:
        """Create the gRPC channel on the thread that is running the loop.

        ``grpc.aio.secure_channel`` calls ``asyncio.get_event_loop()``. Python
        3.12 raises when that thread has no running loop. FastAPI runs sync
        dependencies on an AnyIO worker thread, so constructing ``AsyncClient``
        in ``__init__`` (via ``provider_dep``) fails before any RPC.
        """
        if self._client is None:
            self._client = AsyncClient(
                api_key=self._api_key,
                api_host=self._api_host,
                timeout=self._timeout,
            )
        return self._client

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        temp = self.temperature if temperature is None else temperature
        kwargs: dict[str, Any] = {"model": self.model, "temperature": temp}
        if tools:
            kwargs["tools"] = [_to_xai_tool(spec) for spec in tools]
        chat = self._client_on_running_loop().chat.create(**kwargs)
        for message in messages:
            chat.append(_to_xai_message(message))

        if response_schema is not None:
            response, parsed = await chat.parse(response_schema)
            return ProviderResponse(
                content=response.content or None,
                tool_calls=_from_xai_tool_calls(response.tool_calls),
                parsed=parsed,
            )

        response = await chat.sample()
        return ProviderResponse(
            content=response.content or None,
            tool_calls=_from_xai_tool_calls(response.tool_calls),
        )

    async def stream_text(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream text deltas with ``chat.stream()`` (yields ``(response, chunk)``)."""
        temp = self.temperature if temperature is None else temperature
        chat = self._client_on_running_loop().chat.create(model=self.model, temperature=temp)
        for message in messages:
            chat.append(_to_xai_message(message))
        async for _response, chunk in chat.stream():
            if chunk.content:
                yield chunk.content


def _to_xai_tool(spec: ToolSpec) -> chat_pb2.Tool:
    return tool(name=spec.name, description=spec.description, parameters=spec.parameters)


def _to_xai_message(message: Message) -> chat_pb2.Message:
    if message.role == "system":
        return system(message.content or "")
    if message.role == "user":
        return user(message.content or "")
    if message.role == "tool":
        return tool_result(message.content or "", tool_call_id=message.tool_call_id)
    # assistant, possibly with tool calls that must be reconstructed so a
    # subsequent complete() can continue the same conversation.
    proto = assistant(message.content or "") if message.content else assistant()
    if not message.content:
        proto.ClearField("content")
    for call in message.tool_calls:
        proto.tool_calls.append(
            chat_pb2.ToolCall(
                id=call.id,
                type=chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
                function=chat_pb2.FunctionCall(
                    name=call.name,
                    arguments=json.dumps(call.arguments),
                ),
            )
        )
    return proto


def _from_xai_tool_calls(calls: Any) -> list[ToolCall]:
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
