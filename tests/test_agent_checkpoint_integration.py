"""Postgres checkpointer: an approval survives a process restart. Needs DATABASE_URL."""

from __future__ import annotations

import os
import uuid

import pytest
from langgraph.types import Command

from agentic_rag_platform.graph.builder import build_graph
from agentic_rag_platform.graph.checkpoint import open_postgres_saver
from agentic_rag_platform.graph.nodes import AgentContext
from agentic_rag_platform.graph.runner import run_turn
from agentic_rag_platform.settings import Settings
from agentic_rag_platform.tools.notes import NOTES
from tests.fakes import FakeAgentLLM, FakeRetrieve, hit

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL is unset"),
]


async def test_thread_state_and_interrupt_persist_across_restart() -> None:
    url = os.environ["DATABASE_URL"]
    thread_id = f"it-{uuid.uuid4().hex}"
    settings = Settings(agent_checkpointer="postgres")
    NOTES.clear()

    # Process 1: a normal turn, then a write that stops at the approval gate.
    saver, pool = await open_postgres_saver(url)
    try:
        graph = build_graph(saver)
        ctx = AgentContext(
            provider=FakeAgentLLM(),
            retrieve=FakeRetrieve([hit("pl-bond-edo-108-2026", "EDO1136 5,35%", page=2)]),
            settings=settings,
        )
        first = await run_turn(
            graph,
            ctx,
            thread_id=thread_id,
            payload={"question": "EDO1136?", "tenant_id": "default", "k": 5},
            tenant_id="default",
        )
        assert first.status == "completed"
        ctx.provider = FakeAgentLLM(route="tool", tool_name="save_note")
        second = await run_turn(
            graph,
            ctx,
            thread_id=thread_id,
            payload={"question": "Save a note", "tenant_id": "default", "k": 5},
            tenant_id="default",
        )
        assert second.status == "interrupted"
    finally:
        await pool.close()

    # Process 2: new pool + new compiled graph, same thread_id.
    saver, pool = await open_postgres_saver(url)
    try:
        graph = build_graph(saver)
        ctx = AgentContext(provider=FakeAgentLLM(), retrieve=FakeRetrieve(), settings=settings)
        resumed = await run_turn(
            graph,
            ctx,
            thread_id=thread_id,
            payload=Command(resume={"approved": True}),
            tenant_id="default",
        )
        assert resumed.status == "completed"
        assert resumed.values["answer"].startswith("Saved the note")
        assert len(resumed.values["messages"]) == 4  # both turns are in thread memory
        assert len(NOTES) == 1
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM checkpoints WHERE thread_id = %s", (thread_id,)
                )
            ).fetchone()
        assert row is not None and row["n"] > 0
        await saver.adelete_thread(thread_id)
    finally:
        await pool.close()
