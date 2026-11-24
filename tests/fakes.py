"""In-memory LLM provider used by unit tests."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from pydantic import BaseModel

from agentic_rag_platform.graph.state import (
    DocGrade,
    DocGrades,
    GroundingVerdict,
    RewrittenQuery,
    RouteDecision,
)
from agentic_rag_platform.ingest.types import SearchHit
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


# --- Day 4: a rule-based fake for the LangGraph agent -------------------------

_SOURCE = re.compile(r"\[(\d+)\] \(([^)]*)\)\n(.*?)(?=\n\n\[\d+\] \(|\Z)", re.DOTALL)


class FakeAgentLLM:
    """Answers by response schema, so tests do not depend on node call order.

    * ``RouteDecision``   -> the configured route / tool arguments
    * ``DocGrades``       -> ``relevant_if(passage_text)`` per numbered source
    * ``RewrittenQuery``  -> next item of ``rewrites`` (repeat the last one)
    * ``GroundingVerdict``-> ``grounded``
    * free text           -> ``answer`` (streamed word by word via ``stream_text``)
    """

    name = "fake"
    model = "fake-model"

    def __init__(
        self,
        *,
        route: str = "retrieve",
        tool_name: str | None = None,
        fx: tuple[str, str] = ("EUR", "PLN"),
        note: tuple[str, str] = ("Bond idea", "Check EDO1136 rate"),
        relevant_if: Callable[[str], bool] | None = None,
        rewrites: Sequence[str] = ("rewritten query",),
        answer: str = "The first-period rate is 5.35% [1].",
        grounded: bool = True,
    ) -> None:
        self.route = route
        self.tool_name = tool_name
        self.fx = fx
        self.note = note
        self.relevant_if = relevant_if or (lambda _text: True)
        self.rewrites = list(rewrites)
        self.answer = answer
        self.grounded = grounded
        self.calls: list[str] = []
        self.seen: list[list[Message]] = []

    def _structured(self, schema: type[BaseModel], messages: list[Message]) -> BaseModel:
        last = messages[-1].content or ""
        if schema is RouteDecision:
            return RouteDecision.model_validate(
                {
                    "route": self.route,
                    "tool_name": self.tool_name,
                    "fx_base": self.fx[0],
                    "fx_quote": self.fx[1],
                    "note_title": self.note[0],
                    "note_text": self.note[1],
                    "reason": "fake",
                }
            )
        if schema is DocGrades:
            grades = [
                DocGrade(n=int(n), relevant=self.relevant_if(text))
                for n, _ref, text in _SOURCE.findall(last)
            ]
            return DocGrades(grades=grades)
        if schema is RewrittenQuery:
            query = self.rewrites.pop(0) if len(self.rewrites) > 1 else self.rewrites[0]
            return RewrittenQuery(query=query)
        if schema is GroundingVerdict:
            return GroundingVerdict(grounded=self.grounded, confidence=0.8)
        raise AssertionError(f"unexpected schema {schema}")

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        self.seen.append(list(messages))
        if response_schema is not None:
            self.calls.append(response_schema.__name__)
            parsed = self._structured(response_schema, messages)
            return ProviderResponse(content=parsed.model_dump_json(), parsed=parsed)
        self.calls.append("text")
        return ProviderResponse(content=self.answer)

    async def stream_text(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> AsyncIterator[str]:
        self.seen.append(list(messages))
        self.calls.append("stream")
        for index, word in enumerate(self.answer.split(" ")):
            yield word if index == 0 else f" {word}"


class NonStreamingAgentLLM(FakeAgentLLM):
    """Same fake without ``stream_text``: the agent must fall back to ``complete``."""

    stream_text = None  # type: ignore[assignment]


def hit(doc_id: str, text: str, *, page: int | None = 1, chunk_no: int = 0) -> SearchHit:
    return SearchHit(
        doc_id=doc_id,
        chunk_no=chunk_no,
        page=page,
        section=None,
        text=text,
        content_hash=f"h-{doc_id}-{chunk_no}",
        score=0.5,
    )


class FakeRetrieve:
    """Async retrieve function: maps a query to hits (``default`` otherwise)."""

    def __init__(
        self,
        default: list[SearchHit] | None = None,
        by_query: dict[str, list[SearchHit]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.default = default or []
        self.by_query = by_query or {}
        self.error = error
        self.queries: list[str] = []

    async def __call__(
        self,
        query: str,
        *,
        tenant_id: str,
        k: int,
        strategy: str | None,
        backend: str | None,
    ) -> list[SearchHit]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.by_query.get(query, self.default)[:k]
