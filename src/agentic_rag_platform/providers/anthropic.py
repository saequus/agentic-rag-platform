"""Anthropic / Claude provider (official ``anthropic`` SDK).

This adapter exists so Claude's tool-use and structured-output APIs can be
learned and compared with xAI. It is selected with ``LLM_PROVIDER=anthropic``.

Tool calling
------------
Claude's tool schema is flatter than OpenAI's: ``{"name", "description",
"input_schema"}`` — there is no ``type: function`` wrapper, and parameters are
called ``input_schema``
(https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview).

The model returns ``stop_reason="tool_use"`` and one or more ``tool_use``
content blocks (``id``, ``name``, ``input`` already parsed as a dict). You
echo the assistant ``content`` as the next assistant message, then send tool
results as a **user** message containing ``tool_result`` blocks with
``tool_use_id``. There is no ``role="tool"``.

``strict: true`` is opt-in and enables grammar-constrained tool inputs
(https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use).
The system prompt is a top-level ``system=`` argument, not a message.

Structured outputs
------------------
``client.messages.parse(..., output_format=Answer)`` is the recommended helper
(https://platform.claude.com/docs/en/build-with-claude/structured-outputs).
The SDK sends ``output_config.format`` as a JSON Schema and returns
``parsed_output``. JSON outputs and strict tool use can be combined in one
request: the schema constrains what Claude *says*, ``strict`` constrains how
it *calls* your functions.
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from agentic_rag_platform.providers.base import ProviderError, ProviderResponse
from agentic_rag_platform.schemas import Message, ToolCall
from agentic_rag_platform.tools.registry import ToolSpec


class AnthropicProvider:
    """Anthropic Messages adapter with tools + ``parse()``."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        # Anthropic Messages API 2.x no longer accepts `temperature` on
        # create/parse; the interface still takes it for signature parity.
        _ = temperature if temperature is not None else self.temperature
        system_prompt, anthropic_messages = _to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(spec) for spec in tools]

        if response_schema is not None:
            message = await self._client.messages.parse(output_format=response_schema, **kwargs)
            return ProviderResponse(
                content=_text_content(message.content),
                tool_calls=_from_anthropic_content(message.content),
                parsed=getattr(message, "parsed_output", None),
            )

        message = await self._client.messages.create(**kwargs)
        return ProviderResponse(
            content=_text_content(message.content),
            tool_calls=_from_anthropic_content(message.content),
        )


def _to_anthropic_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.parameters,
        "strict": True,
    }


def _to_anthropic_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Split system prompts out and collapse tool results into user turns."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results() -> None:
        if pending_tool_results:
            out.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
            continue
        if message.role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content or "",
                }
            )
            continue
        flush_tool_results()
        if message.role == "assistant" and message.tool_calls:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            out.append({"role": "assistant", "content": content})
            continue
        out.append({"role": message.role, "content": message.content or ""})

    flush_tool_results()
    return "\n\n".join(system_parts), out


def _text_content(blocks: Any) -> str | None:
    parts: list[str] = []
    for block in blocks or []:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            parts.append(block.text)
    text = "".join(parts).strip()
    return text or None


def _from_anthropic_content(blocks: Any) -> list[ToolCall]:
    result: list[ToolCall] = []
    for block in blocks or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        raw_input = getattr(block, "input", {}) or {}
        arguments = raw_input if isinstance(raw_input, dict) else {"_raw": raw_input}
        result.append(ToolCall(id=block.id, name=block.name, arguments=arguments))
    return result
