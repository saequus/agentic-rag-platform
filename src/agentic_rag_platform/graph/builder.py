"""Wire the nodes into a LangGraph ``StateGraph`` and compile it."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag_platform.graph import nodes
from agentic_rag_platform.graph.nodes import AgentContext
from agentic_rag_platform.graph.state import AgentState

GRAPH_NAME = "rag_agent"

AgentGraph = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None,
    *,
    max_rewrites: int = 2,
) -> AgentGraph:
    graph = StateGraph(AgentState, context_schema=AgentContext)
    graph.add_node("route", nodes.route)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_docs", nodes.grade_docs)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate", nodes.generate)
    graph.add_node("check_grounded", nodes.check_grounded)
    graph.add_node("tool_call", nodes.tool_call)
    graph.add_node("approval", nodes.approval)
    graph.add_node("chitchat", nodes.chitchat)
    graph.add_node("budget_exhausted", nodes.budget_exhausted)
    graph.add_node("fallback", nodes.fallback)
    graph.add_node("finalize", nodes.finalize)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        nodes.after_route,
        ["retrieve", "tool_call", "approval", "chitchat"],
    )
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        nodes.make_after_grade(max_rewrites),
        ["generate", "rewrite_query", "fallback", "budget_exhausted"],
    )
    graph.add_conditional_edges("rewrite_query", nodes.after_rewrite, ["retrieve", "fallback"])
    graph.add_edge("generate", "check_grounded")
    graph.add_conditional_edges("check_grounded", nodes.after_grounded, ["finalize", "fallback"])
    graph.add_edge("budget_exhausted", "fallback")
    for terminal in ("tool_call", "approval", "chitchat", "fallback"):
        graph.add_edge(terminal, "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, name=GRAPH_NAME)


def mermaid(graph: AgentGraph) -> str:
    """Mermaid source of the compiled graph (for README / docs)."""
    return graph.get_graph().draw_mermaid()
