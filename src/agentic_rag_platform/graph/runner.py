"""Run one agent turn (or a resume) and turn graph output into API-friendly events.

The same function backs ``POST /chat`` (collect everything) and
``/chat/stream`` (forward events to SSE as they happen). It owns the root
``invoke_agent`` span, so node, chat, retrieval and tool spans nest under it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.types import Command, StreamMode
from opentelemetry.trace import SpanKind

from agentic_rag_platform.graph.builder import GRAPH_NAME, AgentGraph
from agentic_rag_platform.graph.nodes import AgentContext
from agentic_rag_platform.graph.prompts import FALLBACKS
from agentic_rag_platform.telemetry.tracing import (
    AGENT_NAME,
    AGENT_NAME_ATTR,
    OP_NAME,
    PROVIDER_NAME,
    REQUEST_MODEL,
    TENANT_ID,
    start_span,
)

EventSink = Callable[[dict[str, Any]], Awaitable[None]]
TurnStatus = Literal["completed", "interrupted"]

# Keys a node update may expose to the client in a ``node`` event.
_PUBLIC_KEYS = ("route", "query", "rewrites", "grounded", "fallback_reason", "llm_calls")


@dataclass
class TurnOutcome:
    thread_id: str
    status: TurnStatus
    values: dict[str, Any]
    pending: dict[str, Any] | None = None
    interrupt_id: str | None = None


STREAM_MODES: list[StreamMode] = ["updates", "custom"]


def graph_config(thread_id: str, recursion_limit: int) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}


def node_summary(node: str, update: dict[str, Any] | None) -> dict[str, Any]:
    """Metadata for a node event; no passage text, no answer text."""
    update = update or {}
    summary: dict[str, Any] = {k: update[k] for k in _PUBLIC_KEYS if k in update}
    if "docs" in update:
        summary["docs"] = [
            {"n": d["n"], "doc_id": d["doc_id"], "page": d["page"], "score": round(d["score"], 4)}
            for d in update["docs"]
        ]
    if "relevant" in update:
        summary["relevant"] = [d["n"] for d in update["relevant"]]
    return {"type": "node", "node": node, **summary}


async def pending_interrupt(graph: AgentGraph, thread_id: str) -> tuple[dict[str, Any], str] | None:
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if snapshot.interrupts:
        first = snapshot.interrupts[0]
        value = first.value if isinstance(first.value, dict) else {"value": first.value}
        return value, first.id
    return None


async def run_turn(
    graph: AgentGraph,
    ctx: AgentContext,
    *,
    thread_id: str,
    payload: dict[str, Any] | Command[Any],
    tenant_id: str,
    on_event: EventSink | None = None,
) -> TurnOutcome:
    config = graph_config(thread_id, ctx.settings.agent_recursion_limit)

    async def emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            await on_event(event)

    with start_span(
        f"invoke_agent {AGENT_NAME}",
        kind=SpanKind.INTERNAL,
        attributes={
            OP_NAME: "invoke_agent",
            AGENT_NAME_ATTR: AGENT_NAME,
            PROVIDER_NAME: ctx.provider.name,
            REQUEST_MODEL: ctx.provider.model,
            TENANT_ID: tenant_id,
            "gen_ai.conversation.id": thread_id,
            "langgraph.graph": GRAPH_NAME,
            "langgraph.resume": isinstance(payload, Command),
        },
    ) as span:
        recursion_hit = False
        try:
            async for mode, chunk in graph.astream(
                cast(Any, payload),
                config,
                context=ctx,
                stream_mode=STREAM_MODES,
            ):
                if mode == "custom" and isinstance(chunk, dict):
                    await emit(chunk)
                elif mode == "updates" and isinstance(chunk, dict):
                    for node, update in chunk.items():
                        if node == "__interrupt__":
                            for item in update:
                                await emit(
                                    {"type": "interrupt", "id": item.id, "payload": item.value}
                                )
                        else:
                            await emit(node_summary(node, update))
        except GraphRecursionError:
            recursion_hit = True

        snapshot = await graph.aget_state(config)
        values = dict(snapshot.values)
        if recursion_hit:
            values.update(
                answer=FALLBACKS["recursion_limit"],
                citations=[],
                confidence=0.0,
                fallback_reason="recursion_limit",
            )
        span.set_attribute("agent.llm_calls", int(values.get("llm_calls", 0)))
        span.set_attribute("agent.steps", tuple(values.get("steps", [])))
        if snapshot.interrupts and not recursion_hit:
            first = snapshot.interrupts[0]
            span.set_attribute("langgraph.interrupted", True)
            pending = first.value if isinstance(first.value, dict) else {"value": first.value}
            return TurnOutcome(thread_id, "interrupted", values, pending, first.id)
        if values.get("fallback_reason"):
            span.set_attribute("agent.fallback_reason", str(values["fallback_reason"]))
        return TurnOutcome(thread_id, "completed", values)
