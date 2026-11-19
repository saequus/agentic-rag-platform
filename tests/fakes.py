"""In-memory LLM provider used by unit tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from agentic_rag_platform.providers.base import ProviderResponse
from agentic_rag_platform.schemas import Answer, Message, ToolCall
from agentic_rag_platform.tools.registry import ToolSpec


class ScriptedProvider:
    """Replay a list of canned ``ProviderResponse`` objects, in order."""

    name = "fake"
    model = "fake-model"

    def __init__(self, script: Sequence[ProviderResponse] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[dict[str, Any]] = []

    def enqueue(self, response: ProviderResponse) -> None:
        self.script.append(response)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": [t.name for t in tools] if tools else [],
                "response_schema": response_schema,
                "temperature": temperature,
            }
        )
        if not self.script:
            raise AssertionError("ScriptedProvider has no more canned responses")
        return self.script.pop(0)


def answer(
    text: str,
    citations: list[str] | None = None,
    confidence: float = 0.9,
) -> ProviderResponse:
    parsed = Answer(answer=text, citations=citations or [], confidence=confidence)
    return ProviderResponse(content=parsed.model_dump_json(), parsed=parsed)


def tool_request(tool_id: str, name: str, **arguments: Any) -> ProviderResponse:
    return ProviderResponse(
        content=None,
        tool_calls=[ToolCall(id=tool_id, name=name, arguments=arguments)],
    )
