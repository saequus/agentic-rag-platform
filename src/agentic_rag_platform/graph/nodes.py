"""Graph nodes and edge conditions.

Flow (see ``builder.py`` for the wiring)::

    route ─┬─ retrieve ─ grade_docs ─┬─ generate ─ check_grounded ─┬─ finalize
           │      ▲                  │                             └─ fallback ─ finalize
           │      └─ rewrite_query ◄─┤ (no relevant docs, max 2 rewrites)
           │                         └─ fallback ("I don't know")
           ├─ tool_call (read tool: get_fx_rate) ─ finalize
           ├─ approval (write tool: interrupt() for a human) ─ finalize
           └─ chitchat ─ finalize

Loop and cost guards: ``max_rewrites``, ``max_llm_calls`` budget in state,
repeated-query detection, a timeout on every LLM call, and the graph's
``recursion_limit`` (set by the runner).
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel, ValidationError

from agentic_rag_platform.agent.loop import traced_complete
from agentic_rag_platform.graph import prompts
from agentic_rag_platform.graph.state import (
    PER_TURN_RESET,
    AgentState,
    Citation,
    DocGrades,
    DocRef,
    GroundingVerdict,
    RewrittenQuery,
    RouteDecision,
    Turn,
)
from agentic_rag_platform.ingest.types import SearchHit
from agentic_rag_platform.providers.base import LLMProvider
from agentic_rag_platform.schemas import Message
from agentic_rag_platform.settings import Settings
from agentic_rag_platform.telemetry.tracing import (
    AGENT_NAME,
    AGENT_NAME_ATTR,
    ERROR_TYPE,
    OP_NAME,
    PROVIDER_NAME,
    REQUEST_MODEL,
    REQUEST_TEMPERATURE,
    record_exception,
    start_span,
    tracer,
)
from agentic_rag_platform.tools.fx_rate import FxRateArgs
from agentic_rag_platform.tools.notes import SAVE_NOTE
from agentic_rag_platform.tools.registry import GET_FX_RATE, execute_tool

NodeImpl = Callable[[AgentState, Runtime["AgentContext"]], Awaitable[dict[str, Any]]]


class NodeFn(Protocol):
    """LangGraph passes ``runtime`` by keyword (``_NodeWithRuntime``)."""

    def __call__(
        self, state: AgentState, *, runtime: Runtime[AgentContext]
    ) -> Awaitable[dict[str, Any]]: ...


PASSAGE_CHARS = 900
_CITE = re.compile(r"\[(\d{1,2})\]")


class RetrieveFn(Protocol):
    async def __call__(
        self,
        query: str,
        *,
        tenant_id: str,
        k: int,
        strategy: str | None,
        backend: str | None,
    ) -> list[SearchHit]: ...


@dataclass
class AgentContext:
    """Run-scoped dependencies (LangGraph ``context_schema``), not checkpointed."""

    provider: LLMProvider
    retrieve: RetrieveFn
    settings: Settings


# --- tracing ------------------------------------------------------------------


def traced_node(name: str) -> Callable[[NodeImpl], NodeFn]:
    """Wrap a node in a span and append its name to ``steps``.

    Span attributes are metadata only (route, counts, booleans). Questions,
    passages and answers never go on node spans; the redaction policy of the
    ``chat`` / ``retrieval`` child spans is unchanged.

    ``interrupt()`` works by raising ``GraphInterrupt`` (a ``GraphBubbleUp``).
    That is control flow, not an error, so the span is not marked failed.
    """

    def decorator(fn: NodeImpl) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: AgentState, *, runtime: Runtime[AgentContext]) -> dict[str, Any]:
            with tracer().start_as_current_span(
                f"node {name}",
                kind=SpanKind.INTERNAL,
                attributes={"langgraph.node": name, AGENT_NAME_ATTR: AGENT_NAME},
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                try:
                    update = await fn(state, runtime)
                except GraphBubbleUp:
                    span.set_attribute("langgraph.interrupted", True)
                    raise
                except Exception as exc:
                    record_exception(span, exc)
                    raise
                previous = [] if name == "route" else list(state.get("steps", []))
                update["steps"] = [*previous, name]
                for key in ("route", "rewrites", "llm_calls", "grounded", "fallback_reason"):
                    value = update.get(key)
                    if isinstance(value, bool | int | float | str):
                        span.set_attribute(f"agent.{key}", value)
                if "docs" in update:
                    span.set_attribute("agent.docs.count", len(update["docs"]))
                if "relevant" in update:
                    span.set_attribute("agent.relevant.count", len(update["relevant"]))
                return update

        return wrapper

    return decorator


# --- LLM helpers --------------------------------------------------------------


def _history(state: AgentState, settings: Settings) -> list[Message]:
    turns = state.get("messages", [])
    keep = settings.agent_history_turns * 2
    recent = turns[-keep:] if keep else []
    return [Message(role=turn["role"], content=turn["content"]) for turn in recent]


def _coerce[SchemaT: BaseModel](
    response_parsed: BaseModel | None, content: str | None, schema: type[SchemaT]
) -> SchemaT | None:
    if isinstance(response_parsed, schema):
        return response_parsed
    if response_parsed is not None:
        try:
            return schema.model_validate(response_parsed.model_dump())
        except ValidationError:
            pass
    if content:
        try:
            return schema.model_validate_json(content.strip())
        except ValidationError:
            return None
    return None


async def structured[SchemaT: BaseModel](
    ctx: AgentContext, messages: list[Message], schema: type[SchemaT]
) -> SchemaT | None:
    """One schema-constrained call (LLM-as-judge / router). ``None`` on timeout or bad JSON."""
    try:
        response = await asyncio.wait_for(
            traced_complete(ctx.provider, messages, response_schema=schema, temperature=0.0),
            timeout=ctx.settings.agent_node_timeout_seconds,
        )
    except TimeoutError:
        return None
    return _coerce(response.parsed, response.content, schema)


async def generate_text(
    ctx: AgentContext,
    messages: list[Message],
    *,
    node: str,
    writer: Callable[[Any], None],
) -> str:
    """Free-text generation, streamed as ``token`` events when the provider can stream."""
    provider = ctx.provider
    temperature = ctx.settings.llm_temperature
    stream = getattr(provider, "stream_text", None)
    if stream is None:
        response = await asyncio.wait_for(
            traced_complete(provider, messages, temperature=temperature),
            timeout=ctx.settings.agent_node_timeout_seconds,
        )
        text = (response.content or "").strip()
        if text:
            writer({"type": "token", "node": node, "text": text})
        return text

    parts: list[str] = []
    with start_span(
        f"chat {provider.model}",
        kind=SpanKind.CLIENT,
        attributes={
            OP_NAME: "chat",
            PROVIDER_NAME: provider.name,
            REQUEST_MODEL: provider.model,
            REQUEST_TEMPERATURE: temperature,
            AGENT_NAME_ATTR: AGENT_NAME,
            "gen_ai.request.stream": True,
        },
    ) as span:
        try:
            async with asyncio.timeout(ctx.settings.agent_node_timeout_seconds):
                async for delta in stream(messages, temperature=temperature):
                    parts.append(delta)
                    writer({"type": "token", "node": node, "text": delta})
        except TimeoutError:
            span.set_status(Status(StatusCode.ERROR, "stream timeout"))
            span.set_attribute(ERROR_TYPE, "timeout")
            raise
        span.set_attribute("gen_ai.response.chunk.count", len(parts))
    return "".join(parts).strip()


def budget_left(state: AgentState) -> int:
    return int(state.get("max_llm_calls", 0)) - int(state.get("llm_calls", 0))


# --- nodes --------------------------------------------------------------------


@traced_node("route")
async def route(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    question = state["question"]
    messages = [
        Message(role="system", content=prompts.ROUTER),
        *_history(state, ctx.settings),
        Message(role="user", content=question),
    ]
    decision = await structured(ctx, messages, RouteDecision)
    update: dict[str, Any] = {
        **PER_TURN_RESET,
        "query": question,
        "tried_queries": [question],
        "max_llm_calls": ctx.settings.agent_max_llm_calls,
        "llm_calls": 1,
    }
    if decision is None:
        update.update(route="retrieve", route_reason="router output invalid; default retrieve")
        return update

    chosen = decision.route
    tool_args: dict[str, Any] = {}
    if chosen == "tool" and decision.tool_name == "get_fx_rate":
        try:
            fx = FxRateArgs.model_validate({"base": decision.fx_base, "quote": decision.fx_quote})
            tool_args = fx.model_dump()
        except ValidationError:
            chosen = "retrieve"
    elif chosen == "tool" and decision.tool_name == "save_note":
        tool_args = {
            "title": (decision.note_title or "Note").strip()[:200] or "Note",
            "text": (decision.note_text or question).strip()[:4000],
            "tenant_id": state.get("tenant_id", ctx.settings.default_tenant_id),
        }
    elif chosen == "tool":
        chosen = "retrieve"
    update.update(
        route=chosen,
        route_reason=decision.reason,
        tool_name=decision.tool_name if chosen == "tool" else None,
        tool_args=tool_args,
    )
    return update


@traced_node("retrieve")
async def retrieve(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    settings = ctx.settings
    try:
        hits = await ctx.retrieve(
            state["query"],
            tenant_id=state.get("tenant_id", settings.default_tenant_id),
            k=state.get("k", settings.agent_top_k),
            strategy=state.get("strategy") or settings.agent_retrieval_strategy,
            backend=state.get("backend") or settings.agent_retrieval_backend,
        )
    except Exception as exc:  # DB / Qdrant / model load failures degrade to a fallback.
        return {"docs": [], "fallback_reason": f"retrieval_error: {type(exc).__name__}"}
    docs: list[DocRef] = [
        {
            "n": index,
            "doc_id": hit.doc_id,
            "chunk_no": hit.chunk_no,
            "page": hit.page,
            "section": hit.section,
            "score": float(hit.score),
            "text": hit.text,
        }
        for index, hit in enumerate(hits, start=1)
    ]
    return {"docs": docs}


def _sources_block(docs: list[DocRef]) -> str:
    lines = []
    for doc in docs:
        page = f", p. {doc['page']}" if doc["page"] is not None else ""
        lines.append(f"[{doc['n']}] ({doc['doc_id']}{page})\n{doc['text'][:PASSAGE_CHARS]}")
    return "\n\n".join(lines)


@traced_node("grade_docs")
async def grade_docs(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    docs = state.get("docs", [])
    if not docs:
        return {"relevant": []}
    messages = [
        Message(role="system", content=prompts.GRADER),
        Message(
            role="user",
            content=f"Question: {state['query']}\n\nSources:\n{_sources_block(docs)}",
        ),
    ]
    grades = await structured(ctx, messages, DocGrades)
    calls = state.get("llm_calls", 0) + 1
    if grades is None:
        # Fail open on relevance; check_grounded still guards the answer.
        return {"relevant": docs, "llm_calls": calls}
    keep = {grade.n for grade in grades.grades if grade.relevant}
    return {"relevant": [doc for doc in docs if doc["n"] in keep], "llm_calls": calls}


@traced_node("rewrite_query")
async def rewrite_query(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    tried = list(state.get("tried_queries", []))
    messages = [
        Message(role="system", content=prompts.REWRITER),
        *_history(state, ctx.settings),
        Message(
            role="user",
            content=(
                f"Question: {state['question']}\n"
                f"Already tried: {json.dumps(tried, ensure_ascii=False)}"
            ),
        ),
    ]
    rewritten = await structured(ctx, messages, RewrittenQuery)
    update: dict[str, Any] = {
        "rewrites": state.get("rewrites", 0) + 1,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
    new_query = rewritten.query.strip() if rewritten else ""
    normalised = {q.casefold().strip() for q in tried}
    if not new_query or new_query.casefold() in normalised:
        # Loop detection: the same query would return the same documents.
        update["fallback_reason"] = "repeated_query"
        return update
    update.update(query=new_query, tried_queries=[*tried, new_query])
    return update


def _doc_citation(doc: DocRef) -> Citation:
    page = f" p. {doc['page']}" if doc["page"] is not None else ""
    return {
        "n": doc["n"],
        "source": f"{doc['doc_id']}{page}",
        "doc_id": doc["doc_id"],
        "chunk_no": doc["chunk_no"],
        "page": doc["page"],
        "section": doc["section"],
        "score": doc["score"],
        "url": None,
    }


def extract_citations(answer: str, docs: list[DocRef]) -> list[Citation]:
    """Map ``[n]`` markers in the answer to the documents shown to the model."""
    by_n = {doc["n"]: doc for doc in docs}
    seen: list[int] = []
    for match in _CITE.finditer(answer):
        n = int(match.group(1))
        if n in by_n and n not in seen:
            seen.append(n)
    return [_doc_citation(by_n[n]) for n in seen]


@traced_node("generate")
async def generate(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    relevant = state.get("relevant", [])
    messages = [
        Message(
            role="system",
            content=f"{prompts.GENERATOR}\n\nSources:\n{_sources_block(relevant)}",
        ),
        *_history(state, ctx.settings),
        Message(role="user", content=state["question"]),
    ]
    text = await generate_text(ctx, messages, node="generate", writer=runtime.stream_writer)
    return {
        "answer": text,
        "citations": extract_citations(text, relevant),
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def _is_dont_know(text: str) -> bool:
    return text.strip().lower().rstrip(".!") in {"i don't know", "i do not know", "nie wiem"}


@traced_node("check_grounded")
async def check_grounded(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    if not answer or _is_dont_know(answer):
        return {"grounded": False, "fallback_reason": "no_relevant_documents"}
    if not citations:
        # Deterministic check first: an answer with no valid [n] is not grounded.
        return {"grounded": False, "confidence": 0.0}
    cited = {c["n"] for c in citations}
    sources = [doc for doc in state.get("relevant", []) if doc["n"] in cited]
    messages = [
        Message(role="system", content=prompts.GROUNDING),
        Message(
            role="user",
            content=f"Sources:\n{_sources_block(sources)}\n\nAnswer:\n{answer}",
        ),
    ]
    verdict = await structured(ctx, messages, GroundingVerdict)
    calls = state.get("llm_calls", 0) + 1
    if verdict is None:
        return {"grounded": False, "confidence": 0.0, "llm_calls": calls}
    return {
        "grounded": verdict.grounded,
        "confidence": verdict.confidence if verdict.grounded else 0.0,
        "llm_calls": calls,
    }


@traced_node("tool_call")
async def tool_call(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Read-only tool (no approval): run it, then let the model phrase the answer."""
    ctx = runtime.context
    args = state.get("tool_args", {})
    raw = await execute_tool(GET_FX_RATE.name, args, [GET_FX_RATE])
    try:
        result = cast(dict[str, Any], json.loads(raw))
    except json.JSONDecodeError:
        result = {"error": "tool returned invalid JSON"}
    messages = [
        Message(role="system", content=f"{prompts.TOOL_ANSWER}\n\nTool result:\n{raw}"),
        Message(role="user", content=state["question"]),
    ]
    text = await generate_text(ctx, messages, node="tool_call", writer=runtime.stream_writer)
    citations: list[Citation] = []
    url = result.get("source_url")
    if isinstance(url, str) and url:
        citations.append(
            {
                "n": 1,
                "source": url,
                "doc_id": None,
                "chunk_no": None,
                "page": None,
                "section": None,
                "score": None,
                "url": url,
            }
        )
    return {
        "tool_result": result,
        "answer": text,
        "citations": citations,
        "confidence": 0.3 if "error" in result else 0.9,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


@traced_node("approval")
async def approval(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Human-in-the-loop gate before a write tool.

    ``interrupt()`` saves the checkpoint and stops the run. ``/chat/resume``
    re-invokes the graph with ``Command(resume=...)``; this node then runs again
    **from the top**, and ``interrupt()`` returns the resume value. So nothing
    with side effects may happen before the ``interrupt()`` call.
    """
    args = dict(state.get("tool_args", {}))
    decision = interrupt(
        {
            "type": "approval_required",
            "tool": SAVE_NOTE.name,
            "arguments": args,
            "question": "Approve this write action? Resume with approved=true or false.",
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    comment = decision.get("comment") if isinstance(decision, dict) else None
    record = {"approved": approved, "comment": comment}
    if not approved:
        reason = f" Reason: {comment}" if comment else ""
        return {
            "approval": record,
            "answer": f"OK, I did not save the note.{reason}",
            "confidence": 1.0,
        }
    raw = await execute_tool(SAVE_NOTE.name, args, [SAVE_NOTE])
    result = cast(dict[str, Any], json.loads(raw))
    if result.get("status") == "saved":
        answer = f'Saved the note "{result["title"]}" (id {result["note_id"]}).'
    else:
        answer = f"I could not save the note: {result.get('error', 'unknown error')}."
    return {"approval": record, "tool_result": result, "answer": answer, "confidence": 1.0}


@traced_node("chitchat")
async def chitchat(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    messages = [
        Message(role="system", content=prompts.CHITCHAT),
        *_history(state, ctx.settings),
        Message(role="user", content=state["question"]),
    ]
    text = await generate_text(ctx, messages, node="chitchat", writer=runtime.stream_writer)
    return {
        "answer": text or "Hi!",
        "confidence": 0.9,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


@traced_node("fallback")
async def fallback(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    reason = state.get("fallback_reason")
    if not reason:
        reason = "not_grounded" if state.get("grounded") is False else "no_relevant_documents"
    key = reason.split(":", 1)[0]
    return {
        "fallback_reason": reason,
        "answer": prompts.FALLBACKS.get(key, prompts.FALLBACKS["no_relevant_documents"]),
        "citations": [],
        "confidence": 0.0,
    }


@traced_node("finalize")
async def finalize(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """Write the turn into thread memory (the ``messages`` reducer appends)."""
    turns: list[Turn] = [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state.get("answer", "")},
    ]
    return {"messages": turns}


# --- edge conditions ----------------------------------------------------------


def after_route(state: AgentState) -> str:
    chosen = state.get("route", "retrieve")
    if chosen == "tool":
        return "approval" if state.get("tool_name") == SAVE_NOTE.name else "tool_call"
    if chosen == "chitchat":
        return "chitchat"
    return "retrieve"


def make_after_grade(max_rewrites: int) -> Callable[[AgentState], str]:
    def after_grade(state: AgentState) -> str:
        if state.get("fallback_reason"):
            return "fallback"
        if state.get("relevant"):
            # generate + check_grounded need two calls.
            return "generate" if budget_left(state) >= 2 else "budget_exhausted"
        if state.get("rewrites", 0) >= max_rewrites:
            return "fallback"
        # rewrite + grade (+ generate + check_grounded later): need at least 2 now.
        return "rewrite_query" if budget_left(state) >= 2 else "budget_exhausted"

    return after_grade


def after_rewrite(state: AgentState) -> str:
    return "fallback" if state.get("fallback_reason") else "retrieve"


def after_grounded(state: AgentState) -> str:
    return "finalize" if state.get("grounded") else "fallback"


@traced_node("budget_exhausted")
async def budget_exhausted(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    return {"fallback_reason": "budget_exhausted"}
