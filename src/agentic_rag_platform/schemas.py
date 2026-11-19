"""Shared request, response and conversation schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A provider-agnostic request from the model to run a local tool."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """One turn in the provider-agnostic conversation history."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class Answer(BaseModel):
    """Structured final answer returned by `/chat` and validated with Pydantic.

    Confidence is a self-reported score in `[0, 1]`. It is not a calibrated
    probability; treat it as a hint, not a guarantee.
    """

    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("answer")
    @classmethod
    def answer_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must not be empty")
        return value


class ChatMessage(BaseModel):
    """A user-supplied history turn (no tool protocol details)."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class ExecutedToolCall(BaseModel):
    """What the agent actually ran while answering, included for demos."""

    id: str
    name: str
    arguments: dict[str, Any]
    result: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
    provider: str
    model: str
    tool_calls: list[ExecutedToolCall] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider: str
    model: str
