"""LangGraph agent: routing, grading, rewrite loop, guards, interrupt, memory. No network."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentic_rag_platform.graph.builder import build_graph, mermaid
from agentic_rag_platform.graph.nodes import AgentContext, extract_citations
from agentic_rag_platform.graph.runner import TurnOutcome, run_turn
from agentic_rag_platform.settings import Settings
from agentic_rag_platform.tools.fx_rate import NBP_RATE_URL
from agentic_rag_platform.tools.notes import NOTES
from tests.fakes import FakeAgentLLM, FakeRetrieve, NonStreamingAgentLLM, hit

EDO = hit("pl-bond-edo-108-2026", "EDO1136: w pierwszym okresie odsetkowym 5,35%.", page=2)
ECB = hit("ecb-eb202607", "Euro area unemployment stood at 6.4% in July.", page=7, chunk_no=3)


def _settings(**overrides: Any) -> Settings:
    return Settings(agent_checkpointer="memory", **overrides)


async def _run(
    llm: FakeAgentLLM,
    retrieve: FakeRetrieve,
    question: str = "Jakie jest oprocentowanie EDO1136?",
    *,
    settings: Settings | None = None,
    thread_id: str = "t1",
    graph: Any = None,
    events: list[dict[str, Any]] | None = None,
) -> tuple[TurnOutcome, Any]:
    cfg = settings or _settings()
    graph = graph or build_graph(InMemorySaver(), max_rewrites=cfg.agent_max_rewrites)
    ctx = AgentContext(provider=llm, retrieve=retrieve, settings=cfg)

    async def sink(event: dict[str, Any]) -> None:
        if events is not None:
            events.append(event)

    outcome = await run_turn(
        graph,
        ctx,
        thread_id=thread_id,
        payload={"question": question, "tenant_id": "default", "k": 5},
        tenant_id="default",
        on_event=sink,
    )
    return outcome, graph


async def test_retrieve_grade_generate_grounded_with_citations() -> None:
    llm = FakeAgentLLM()
    events: list[dict[str, Any]] = []
    outcome, _ = await _run(llm, FakeRetrieve([EDO]), events=events)
    values = outcome.values
    assert outcome.status == "completed"
    assert values["steps"] == [
        "route",
        "retrieve",
        "grade_docs",
        "generate",
        "check_grounded",
        "finalize",
    ]
    assert values["answer"] == "The first-period rate is 5.35% [1]."
    assert values["grounded"] is True
    assert values["citations"][0]["doc_id"] == "pl-bond-edo-108-2026"
    assert values["citations"][0]["page"] == 2
    assert values["llm_calls"] == 4
    assert llm.calls == ["RouteDecision", "DocGrades", "stream", "GroundingVerdict"]
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == values["answer"]
    assert [e["node"] for e in events if e["type"] == "node"][:2] == ["route", "retrieve"]


async def test_non_streaming_provider_falls_back_to_complete() -> None:
    llm = NonStreamingAgentLLM()
    events: list[dict[str, Any]] = []
    outcome, _ = await _run(llm, FakeRetrieve([EDO]), events=events)
    assert outcome.values["grounded"] is True
    assert "text" in llm.calls
    assert [e["text"] for e in events if e["type"] == "token"] == [outcome.values["answer"]]


async def test_rewrite_loop_recovers_relevant_docs() -> None:
    llm = FakeAgentLLM(
        relevant_if=lambda text: "EDO1136" in text, rewrites=["EDO1136 oprocentowanie"]
    )
    retrieve = FakeRetrieve([ECB], by_query={"EDO1136 oprocentowanie": [EDO]})
    outcome, _ = await _run(llm, retrieve, "What does the 10-year bond pay first?")
    values = outcome.values
    assert values["rewrites"] == 1
    assert retrieve.queries == ["What does the 10-year bond pay first?", "EDO1136 oprocentowanie"]
    assert values["grounded"] is True
    assert values["citations"][0]["doc_id"] == "pl-bond-edo-108-2026"


async def test_max_rewrites_then_i_dont_know() -> None:
    llm = FakeAgentLLM(relevant_if=lambda _t: False, rewrites=["q2", "q3", "q4"])
    retrieve = FakeRetrieve([ECB])
    outcome, _ = await _run(llm, retrieve)
    values = outcome.values
    assert values["rewrites"] == 2
    assert len(retrieve.queries) == 3
    assert values["fallback_reason"] == "no_relevant_documents"
    assert values["answer"].startswith("I don't know")
    assert values["citations"] == []
    assert "stream" not in llm.calls  # never generated an answer


async def test_repeated_rewrite_is_detected_as_a_loop() -> None:
    question = "Jakie jest oprocentowanie EDO1136?"
    llm = FakeAgentLLM(relevant_if=lambda _t: False, rewrites=[question])
    outcome, _ = await _run(llm, FakeRetrieve([ECB]), question)
    assert outcome.values["fallback_reason"] == "repeated_query"
    assert outcome.values["rewrites"] == 1


async def test_ungrounded_answer_is_replaced_by_fallback() -> None:
    llm = FakeAgentLLM(grounded=False)
    outcome, _ = await _run(llm, FakeRetrieve([EDO]))
    values = outcome.values
    assert values["grounded"] is False
    assert values["fallback_reason"] == "not_grounded"
    assert values["answer"].startswith("I don't know")
    assert values["citations"] == []


async def test_answer_without_citations_fails_grounding_without_llm_call() -> None:
    llm = FakeAgentLLM(answer="The rate is 5.35%.")
    outcome, _ = await _run(llm, FakeRetrieve([EDO]))
    assert outcome.values["fallback_reason"] == "not_grounded"
    assert "GroundingVerdict" not in llm.calls


async def test_budget_in_state_stops_the_turn() -> None:
    llm = FakeAgentLLM(relevant_if=lambda _t: False)
    outcome, _ = await _run(llm, FakeRetrieve([ECB]), settings=_settings(agent_max_llm_calls=3))
    values = outcome.values
    assert values["fallback_reason"] == "budget_exhausted"
    assert values["llm_calls"] <= 3
    assert "budget_exhausted" in values["steps"]


async def test_recursion_limit_returns_fallback() -> None:
    llm = FakeAgentLLM(relevant_if=lambda _t: False, rewrites=["a", "b", "c", "d"])
    cfg = _settings(agent_recursion_limit=5, agent_max_rewrites=5)
    outcome, _ = await _run(llm, FakeRetrieve([ECB]), settings=cfg)
    assert outcome.values["fallback_reason"] == "recursion_limit"
    assert outcome.values["answer"].startswith("I don't know")


async def test_retrieval_error_degrades_to_fallback() -> None:
    outcome, _ = await _run(FakeAgentLLM(), FakeRetrieve(error=OSError("db down")))
    assert outcome.values["fallback_reason"] == "retrieval_error: OSError"
    assert outcome.values["answer"].startswith("I can't search")


async def test_chitchat_skips_retrieval() -> None:
    llm = FakeAgentLLM(route="chitchat", answer="Hi! I answer questions about bonds.")
    retrieve = FakeRetrieve([EDO])
    outcome, _ = await _run(llm, retrieve, "hello")
    assert outcome.values["steps"] == ["route", "chitchat", "finalize"]
    assert retrieve.queries == []
    assert outcome.values["answer"].startswith("Hi!")


@respx.mock
async def test_fx_tool_route_cites_tool_source() -> None:
    respx.get(NBP_RATE_URL.format(code="eur")).mock(
        return_value=httpx.Response(
            200, json={"rates": [{"effectiveDate": "2026-11-24", "mid": 4.25}]}
        )
    )
    llm = FakeAgentLLM(route="tool", tool_name="get_fx_rate", answer="1 EUR = 4.25 PLN.")
    outcome, _ = await _run(llm, FakeRetrieve([EDO]), "EUR to PLN?")
    values = outcome.values
    assert values["steps"] == ["route", "tool_call", "finalize"]
    assert values["tool_result"]["rate"] == 4.25
    assert values["citations"][0]["url"].startswith("https://")


async def test_write_tool_interrupts_then_resumes_on_approval() -> None:
    NOTES.clear()
    llm = FakeAgentLLM(route="tool", tool_name="save_note")
    events: list[dict[str, Any]] = []
    outcome, graph = await _run(llm, FakeRetrieve(), "Save a note: check EDO1136", events=events)
    assert outcome.status == "interrupted"
    assert outcome.pending is not None
    assert outcome.pending["tool"] == "save_note"
    assert outcome.pending["arguments"]["title"] == "Bond idea"
    assert NOTES == {}  # nothing written before approval
    assert any(e["type"] == "interrupt" for e in events)

    ctx = AgentContext(provider=llm, retrieve=FakeRetrieve(), settings=_settings())
    resumed = await run_turn(
        graph,
        ctx,
        thread_id="t1",
        payload=Command(resume={"approved": True}),
        tenant_id="default",
    )
    assert resumed.status == "completed"
    assert resumed.values["answer"].startswith('Saved the note "Bond idea"')
    assert len(NOTES) == 1
    assert resumed.values["messages"][-1]["role"] == "assistant"


async def test_write_tool_rejected_does_not_write() -> None:
    NOTES.clear()
    llm = FakeAgentLLM(route="tool", tool_name="save_note")
    outcome, graph = await _run(llm, FakeRetrieve(), "Save a note")
    assert outcome.status == "interrupted"
    ctx = AgentContext(provider=llm, retrieve=FakeRetrieve(), settings=_settings())
    resumed = await run_turn(
        graph,
        ctx,
        thread_id="t1",
        payload=Command(resume={"approved": False, "comment": "not now"}),
        tenant_id="default",
    )
    assert resumed.values["answer"] == "OK, I did not save the note. Reason: not now"
    assert NOTES == {}


async def test_thread_memory_feeds_history_into_next_turn() -> None:
    llm = FakeAgentLLM()
    retrieve = FakeRetrieve([EDO])
    first, graph = await _run(llm, retrieve, "Jakie jest oprocentowanie EDO1136?")
    llm.seen.clear()
    second, _ = await _run(llm, retrieve, "A w drugim okresie?", graph=graph)
    assert len(second.values["messages"]) == 4
    router_messages = llm.seen[0]
    contents = [m.content for m in router_messages]
    assert "Jakie jest oprocentowanie EDO1136?" in contents
    assert first.values["answer"] in contents
    # Per-turn scratch was reset: the new turn counts its own LLM calls.
    assert second.values["llm_calls"] == 4

    other, _ = await _run(llm, retrieve, "hello", graph=graph, thread_id="t2")
    assert len(other.values["messages"]) == 2


def test_extract_citations_ignores_unknown_numbers() -> None:
    docs = [
        {
            "n": 1,
            "doc_id": "d",
            "chunk_no": 0,
            "page": 3,
            "section": None,
            "score": 1.0,
            "text": "x",
        }
    ]
    cites = extract_citations("A [1] B [7] C [1]", docs)  # type: ignore[arg-type]
    assert [c["n"] for c in cites] == [1]
    assert cites[0]["source"] == "d p. 3"


def test_graph_shape_mermaid() -> None:
    source = mermaid(build_graph(None))
    for node in ("route", "retrieve", "grade_docs", "rewrite_query", "check_grounded", "approval"):
        assert node in source


@pytest.mark.parametrize("route", ["retrieve", "chitchat"])
async def test_router_failure_defaults_to_retrieve(route: str) -> None:
    class BrokenRouter(FakeAgentLLM):
        def _structured(self, schema: Any, messages: Any) -> Any:
            if schema.__name__ == "RouteDecision":
                raise TimeoutError
            return super()._structured(schema, messages)

    llm = BrokenRouter(route=route)
    outcome, _ = await _run(llm, FakeRetrieve([EDO]))
    assert outcome.values["route"] == "retrieve"
