"""Pydantic AI version of retrieve -> answer, run against pydantic-ai's TestModel (no network)."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel

from agentic_rag_platform.agent.pydantic_ai_agent import CitedAnswer, RagDeps, build_agent
from tests.fakes import FakeRetrieve, hit


async def test_pydantic_ai_agent_calls_search_tool_and_returns_typed_output() -> None:
    retrieve = FakeRetrieve([hit("pl-bond-edo-108-2026", "EDO1136 5,35%", page=2)])
    agent = build_agent(TestModel())
    result = await agent.run("Jakie jest oprocentowanie EDO1136?", deps=RagDeps(retrieve=retrieve))
    assert isinstance(result.output, CitedAnswer)
    assert len(retrieve.queries) == 1  # TestModel calls every tool once
    assert 0.0 <= result.output.confidence <= 1.0
