"""Provider-agnostic LLM interface used by the tool-calling loop."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from agentic_rag_platform.schemas import Message, ToolCall
from agentic_rag_platform.tools.registry import ToolSpec


class ProviderError(Exception):
    """Raised when a provider cannot be configured or a call fails."""


class ProviderResponse(BaseModel):
    """Normalised result of a single provider turn."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    parsed: BaseModel | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(Protocol):
    """Chat completion with optional tools and an optional response schema.

    Implementations must:
    - accept the generic `Message` / `ToolSpec` shapes
    - return tool calls in `ProviderResponse.tool_calls` when the model wants
      to invoke a function
    - when `response_schema` is set, constrain generation and return a
      Pydantic instance in `parsed` (and still validate again in the loop)
    """

    name: str
    model: str

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse: ...
