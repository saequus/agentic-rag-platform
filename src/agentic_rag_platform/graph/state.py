"""Graph state and the structured outputs the LLM-as-judge nodes ask for.

State is a ``TypedDict`` because LangGraph checkpoints it as plain data. Every
value is JSON-friendly (dicts, lists, str, int, float, bool) so the Postgres
checkpointer can serialise it without custom types.

Two kinds of keys:

* **Thread memory** (``messages``) uses an ``operator.add`` reducer and
  survives across turns of the same ``thread_id``.
* **Per-turn scratch** (everything else) is reset by the ``route`` node at the
  start of each turn.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

Route = Literal["retrieve", "tool", "chitchat"]
ToolName = Literal["get_fx_rate", "save_note"]


class Turn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class DocRef(TypedDict):
    """A retrieved chunk as the graph carries it (text kept for prompts only)."""

    n: int
    doc_id: str
    chunk_no: int
    page: int | None
    section: str | None
    score: float
    text: str


class Citation(TypedDict):
    """A source the answer cites: a document chunk, or a tool's source URL."""

    n: int
    source: str
    doc_id: str | None
    chunk_no: int | None
    page: int | None
    section: str | None
    score: float | None
    url: str | None


class AgentState(TypedDict, total=False):
    # Input for the current turn.
    question: str
    tenant_id: str
    strategy: str | None
    backend: str | None
    k: int

    # Thread memory (short-term): persisted by the checkpointer per thread_id.
    messages: Annotated[list[Turn], operator.add]

    # Per-turn scratch, reset by ``route``.
    route: Route
    route_reason: str
    tool_name: ToolName | None
    tool_args: dict[str, Any]
    tool_result: dict[str, Any] | None
    query: str
    tried_queries: list[str]
    rewrites: int
    docs: list[DocRef]
    relevant: list[DocRef]
    answer: str
    citations: list[Citation]
    grounded: bool | None
    confidence: float
    fallback_reason: str | None
    approval: dict[str, Any] | None

    # Cost guards kept in state so they are checkpointed with the turn.
    llm_calls: int
    max_llm_calls: int
    steps: list[str]


# --- Structured outputs (LLM-as-judge / router) -------------------------------


class RouteDecision(BaseModel):
    """Router output. Tool arguments are explicit fields (portable JSON Schema)."""

    route: Route = Field(
        description=(
            "retrieve: the question needs facts from the indexed finance documents "
            "(ECB bulletins, Polish treasury bond letters). tool: the user wants a "
            "live FX rate or asks to save a note. chitchat: greetings, thanks, or "
            "questions about the assistant itself."
        )
    )
    tool_name: ToolName | None = Field(default=None, description="Only when route is 'tool'.")
    fx_base: str | None = Field(default=None, description="ISO 4217 code for get_fx_rate")
    fx_quote: str | None = Field(default=None, description="ISO 4217 code for get_fx_rate")
    note_title: str | None = Field(default=None, description="Title for save_note")
    note_text: str | None = Field(default=None, description="Body for save_note")
    reason: str = Field(default="", description="One short sentence.")


class DocGrade(BaseModel):
    n: int = Field(description="The source number shown in brackets, e.g. 1 for [1].")
    relevant: bool
    reason: str = ""


class DocGrades(BaseModel):
    """One structured call grades every candidate (cheaper than one call per doc)."""

    grades: list[DocGrade] = Field(default_factory=list)


class RewrittenQuery(BaseModel):
    query: str = Field(min_length=1)


class GroundingVerdict(BaseModel):
    grounded: bool = Field(description="True only if every claim is supported by the sources.")
    unsupported_claims: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


PER_TURN_RESET: dict[str, Any] = {
    "route_reason": "",
    "tool_name": None,
    "tool_args": {},
    "tool_result": None,
    "tried_queries": [],
    "rewrites": 0,
    "docs": [],
    "relevant": [],
    "answer": "",
    "citations": [],
    "grounded": None,
    "confidence": 0.0,
    "fallback_reason": None,
    "approval": None,
    "llm_calls": 0,
}
