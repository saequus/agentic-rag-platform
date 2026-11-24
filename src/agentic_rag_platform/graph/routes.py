"""Day 4 HTTP API for the LangGraph agent.

* ``POST /chat``         run one turn, return the answer (or a pending approval)
* ``POST /chat/stream``  same turn as Server-Sent Events: node events + tokens
* ``POST /chat/resume``  approve / reject a pending write action (``interrupt()``)
* ``GET  /chat/{thread_id}``  thread memory and any pending approval

Every request carries a ``thread_id`` (generated when omitted). The
checkpointer stores the graph state per thread, which is the agent's
short-term memory.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from agentic_rag_platform.deps import provider_dep, settings_dep
from agentic_rag_platform.graph.checkpoint import AgentRuntime, get_agent_runtime
from agentic_rag_platform.graph.nodes import AgentContext, RetrieveFn
from agentic_rag_platform.graph.runner import TurnOutcome, pending_interrupt, run_turn
from agentic_rag_platform.ingest.embedder import get_embedder
from agentic_rag_platform.ingest.types import SearchHit
from agentic_rag_platform.providers.base import LLMProvider, ProviderError
from agentic_rag_platform.retrieval.service import Retriever
from agentic_rag_platform.settings import (
    RetrievalBackend,
    RetrievalStrategy,
    Settings,
)

router = APIRouter(prefix="/chat", tags=["agent"])

ThreadId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,128}$")]


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: ThreadId | None = Field(
        default=None, description="Reuse to continue a conversation; omitted = new thread."
    )
    tenant_id: str | None = None
    strategy: RetrievalStrategy | None = None
    backend: RetrievalBackend | None = None
    k: int | None = Field(default=None, ge=1, le=20)


class ResumeRequest(BaseModel):
    thread_id: ThreadId
    approved: bool
    comment: str | None = Field(default=None, max_length=500)


class CitationBody(BaseModel):
    n: int
    source: str
    doc_id: str | None = None
    chunk_no: int | None = None
    page: int | None = None
    section: str | None = None
    score: float | None = None
    url: str | None = None


class PendingApproval(BaseModel):
    interrupt_id: str
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    question: str | None = None


class AgentChatResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "interrupted"]
    answer: str | None
    citations: list[CitationBody] = Field(default_factory=list)
    confidence: float | None = None
    grounded: bool | None = None
    route: str | None = None
    rewrites: int = 0
    llm_calls: int = 0
    steps: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    pending_approval: PendingApproval | None = None
    provider: str
    model: str
    checkpointer: str


class ThreadResponse(BaseModel):
    thread_id: str
    messages: list[dict[str, str]]
    pending_approval: PendingApproval | None = None


# --- dependencies -------------------------------------------------------------

_retriever: Retriever | None = None
_retriever_lock = threading.Lock()


def _shared_retriever(settings: Settings) -> Retriever:
    """One Retriever (and embedding model) per process, built on first use.

    Chit-chat and tool turns never load torch; the first retrieval does.
    """
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = Retriever(settings=settings, embedder=get_embedder(settings))
        return _retriever


def retrieve_fn_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> RetrieveFn:
    async def _retrieve(
        query: str,
        *,
        tenant_id: str,
        k: int,
        strategy: str | None,
        backend: str | None,
    ) -> list[SearchHit]:
        def run() -> list[SearchHit]:
            result = _shared_retriever(settings).retrieve(
                query,
                tenant_id=tenant_id,
                k=k,
                strategy=cast(RetrievalStrategy | None, strategy),
                backend=cast(RetrievalBackend | None, backend),
            )
            return result.hits

        # Sync psycopg + CPU embedding: keep them off the event loop.
        return await asyncio.to_thread(run)

    return _retrieve


async def agent_runtime_dep(
    settings: Annotated[Settings, Depends(settings_dep)],
) -> AgentRuntime:
    return await get_agent_runtime(settings)


SettingsDep = Annotated[Settings, Depends(settings_dep)]
ProviderDep = Annotated[LLMProvider, Depends(provider_dep)]
RetrieveDep = Annotated[RetrieveFn, Depends(retrieve_fn_dep)]
RuntimeDep = Annotated[AgentRuntime, Depends(agent_runtime_dep)]


# --- helpers ------------------------------------------------------------------


def _pending_body(
    payload: dict[str, Any] | None, interrupt_id: str | None
) -> PendingApproval | None:
    if payload is None or interrupt_id is None:
        return None
    return PendingApproval(
        interrupt_id=interrupt_id,
        tool=payload.get("tool"),
        arguments=payload.get("arguments", {}),
        question=payload.get("question"),
    )


def _response(
    outcome: TurnOutcome, provider: LLMProvider, runtime: AgentRuntime
) -> AgentChatResponse:
    values = outcome.values
    interrupted = outcome.status == "interrupted"
    return AgentChatResponse(
        thread_id=outcome.thread_id,
        status=outcome.status,
        answer=None if interrupted else values.get("answer"),
        citations=[] if interrupted else [CitationBody(**c) for c in values.get("citations", [])],
        confidence=None if interrupted else values.get("confidence"),
        grounded=values.get("grounded"),
        route=values.get("route"),
        rewrites=int(values.get("rewrites", 0)),
        llm_calls=int(values.get("llm_calls", 0)),
        steps=list(values.get("steps", [])),
        fallback_reason=values.get("fallback_reason"),
        pending_approval=_pending_body(outcome.pending, outcome.interrupt_id),
        provider=provider.name,
        model=provider.model,
        checkpointer=runtime.checkpointer,
    )


async def _prepare_turn(
    request: AgentChatRequest, settings: Settings, runtime: AgentRuntime
) -> tuple[str, str, dict[str, Any]]:
    thread_id = request.thread_id or uuid.uuid4().hex
    tenant_id = request.tenant_id or settings.default_tenant_id
    snapshot = await runtime.graph.aget_state({"configurable": {"thread_id": thread_id}})
    owner = snapshot.values.get("tenant_id") if snapshot.values else None
    if owner is not None and owner != tenant_id:
        # Thread memory is tenant-scoped; never continue another tenant's thread.
        raise HTTPException(status_code=403, detail="thread belongs to another tenant")
    if snapshot.interrupts:
        raise HTTPException(
            status_code=409,
            detail="thread is waiting for approval; call POST /chat/resume first",
        )
    payload: dict[str, Any] = {
        "question": request.message,
        "tenant_id": tenant_id,
        "strategy": request.strategy,
        "backend": request.backend,
        "k": request.k or settings.agent_top_k,
    }
    return thread_id, tenant_id, payload


def _sse(event: dict[str, Any]) -> str:
    kind = str(event.get("type", "message"))
    return f"event: {kind}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


# --- routes -------------------------------------------------------------------


@router.post("", response_model=AgentChatResponse)
async def chat(
    request: AgentChatRequest,
    settings: SettingsDep,
    provider: ProviderDep,
    retrieve: RetrieveDep,
    runtime: RuntimeDep,
) -> AgentChatResponse:
    thread_id, tenant_id, payload = await _prepare_turn(request, settings, runtime)
    ctx = AgentContext(provider=provider, retrieve=retrieve, settings=settings)
    try:
        outcome = await run_turn(
            runtime.graph, ctx, thread_id=thread_id, payload=payload, tenant_id=tenant_id
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _response(outcome, provider, runtime)


@router.post("/stream")
async def chat_stream(
    request: AgentChatRequest,
    settings: SettingsDep,
    provider: ProviderDep,
    retrieve: RetrieveDep,
    runtime: RuntimeDep,
) -> StreamingResponse:
    """SSE: ``start``, ``node`` (per node), ``token`` (text deltas), ``interrupt``, ``done``.

    Tokens from ``generate`` are provisional: ``check_grounded`` runs after the
    text is streamed. The ``done`` event carries the final answer, which is the
    "I don't know" fallback if grounding failed. Clients render ``done``.
    """
    thread_id, tenant_id, payload = await _prepare_turn(request, settings, runtime)
    ctx = AgentContext(provider=provider, retrieve=retrieve, settings=settings)
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def producer() -> None:
        try:
            outcome = await run_turn(
                runtime.graph,
                ctx,
                thread_id=thread_id,
                payload=payload,
                tenant_id=tenant_id,
                on_event=queue.put,
            )
            body = _response(outcome, provider, runtime).model_dump()
            await queue.put({"type": "done", **body})
        except Exception as exc:
            await queue.put({"type": "error", "detail": f"{type(exc).__name__}: {exc}"[:300]})
        finally:
            await queue.put(None)

    async def events() -> AsyncIterator[str]:
        yield _sse({"type": "start", "thread_id": thread_id})
        task = asyncio.create_task(producer())
        try:
            while (event := await queue.get()) is not None:
                yield _sse(event)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/resume", response_model=AgentChatResponse)
async def chat_resume(
    request: ResumeRequest,
    settings: SettingsDep,
    provider: ProviderDep,
    retrieve: RetrieveDep,
    runtime: RuntimeDep,
) -> AgentChatResponse:
    pending = await pending_interrupt(runtime.graph, request.thread_id)
    if pending is None:
        raise HTTPException(status_code=409, detail="no pending approval on this thread")
    snapshot = await runtime.graph.aget_state({"configurable": {"thread_id": request.thread_id}})
    tenant_id = str(snapshot.values.get("tenant_id", settings.default_tenant_id))
    ctx = AgentContext(provider=provider, retrieve=retrieve, settings=settings)
    command: Command[Any] = Command(
        resume={"approved": request.approved, "comment": request.comment}
    )
    try:
        outcome = await run_turn(
            runtime.graph,
            ctx,
            thread_id=request.thread_id,
            payload=command,
            tenant_id=tenant_id,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return _response(outcome, provider, runtime)


@router.get("/{thread_id}", response_model=ThreadResponse)
async def get_thread(
    thread_id: str,
    settings: SettingsDep,
    runtime: RuntimeDep,
    tenant_id: str | None = None,
) -> ThreadResponse:
    snapshot = await runtime.graph.aget_state({"configurable": {"thread_id": thread_id}})
    tenant = tenant_id or settings.default_tenant_id
    if not snapshot.values or snapshot.values.get("tenant_id", tenant) != tenant:
        # Same 404 for "missing" and "other tenant": do not leak thread ids.
        raise HTTPException(status_code=404, detail="unknown thread")
    pending = None
    if snapshot.interrupts:
        first = snapshot.interrupts[0]
        value = first.value if isinstance(first.value, dict) else {"value": first.value}
        pending = _pending_body(value, first.id)
    return ThreadResponse(
        thread_id=thread_id,
        messages=[dict(m) for m in snapshot.values.get("messages", [])],
        pending_approval=pending,
    )
