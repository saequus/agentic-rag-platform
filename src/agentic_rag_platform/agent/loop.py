"""Provider-agnostic tool-calling loop with a structured final answer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from opentelemetry.trace import SpanKind
from pydantic import BaseModel, ValidationError

from agentic_rag_platform.providers.base import LLMProvider, ProviderResponse
from agentic_rag_platform.schemas import Answer, ExecutedToolCall, Message
from agentic_rag_platform.telemetry.tracing import (
    AGENT_NAME,
    AGENT_NAME_ATTR,
    OP_NAME,
    OUTPUT_TYPE,
    PROVIDER_NAME,
    REQUEST_MODEL,
    REQUEST_TEMPERATURE,
    RESPONSE_FINISH,
    USAGE_INPUT,
    USAGE_OUTPUT,
    maybe_set,
    start_span,
)
from agentic_rag_platform.tools.registry import ToolSpec, default_tools, execute_tool

DEFAULT_SYSTEM_PROMPT = """You are a careful assistant for finance-related questions.

When you need a foreign-exchange rate, call the get_fx_rate tool instead of guessing.
Use the tool result, including its source_url, when you cite sources.

Your final reply must be structured as:
- answer: a concise, factual reply in the user's language
- citations: source URLs or short source labels (empty list if none)
- confidence: a number between 0 and 1 reflecting how certain you are
  (lower it when sources are missing, stale, or the question is ambiguous)

Do not invent citations or rates.
"""


@dataclass
class AgentResult:
    answer: Answer
    tool_calls: list[ExecutedToolCall] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)


async def run_agent(
    provider: LLMProvider,
    user_message: str,
    *,
    history: list[Message] | None = None,
    tools: list[ToolSpec] | None = None,
    max_iterations: int = 5,
    temperature: float = 0.2,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> AgentResult:
    """Run tools until the model answers, then request a validated ``Answer``.

    The tool loop and the structured-output turn are separate so every provider
    can implement them with its own native API. ``max_iterations`` caps only
    the tool loop; one extra completion is used for the final schema.
    """
    catalog = tools if tools is not None else default_tools()
    messages: list[Message] = [
        Message(role="system", content=system_prompt),
        *(history or []),
        Message(role="user", content=user_message),
    ]
    executed: list[ExecutedToolCall] = []

    with start_span(
        f"invoke_agent {AGENT_NAME}",
        attributes={
            OP_NAME: "invoke_agent",
            AGENT_NAME_ATTR: AGENT_NAME,
            REQUEST_TEMPERATURE: temperature,
            PROVIDER_NAME: provider.name,
            REQUEST_MODEL: provider.model,
        },
    ) as agent_span:
        result = await _run_tool_loop(
            provider,
            messages=messages,
            catalog=catalog,
            executed=executed,
            max_iterations=max_iterations,
            temperature=temperature,
        )
        agent_span.set_attribute("gen_ai.tool.call.count", len(result.tool_calls))
        return result


async def _run_tool_loop(
    provider: LLMProvider,
    *,
    messages: list[Message],
    catalog: list[ToolSpec],
    executed: list[ExecutedToolCall],
    max_iterations: int,
    temperature: float,
) -> AgentResult:
    for _ in range(max_iterations):
        response = await _traced_complete(
            provider, messages, tools=catalog, temperature=temperature
        )
        if not response.tool_calls:
            parsed = _coerce_answer(response)
            if parsed is not None:
                return AgentResult(answer=parsed, tool_calls=executed, messages=messages)
            if response.content:
                messages.append(Message(role="assistant", content=response.content))
            break

        messages.append(
            Message(
                role="assistant",
                content=response.content,
                tool_calls=list(response.tool_calls),
            )
        )
        for call in response.tool_calls:
            result = await execute_tool(call.name, call.arguments, catalog, tool_call_id=call.id)
            executed.append(
                ExecutedToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    result=result,
                )
            )
            messages.append(
                Message(
                    role="tool",
                    content=result,
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
    else:
        # Hit the iteration cap while the model was still requesting tools.
        messages.append(
            Message(
                role="user",
                content=(
                    "You have reached the maximum number of tool calls. "
                    "Produce your best structured answer with the evidence you have."
                ),
            )
        )

    final = await _traced_complete(
        provider,
        messages,
        tools=None,
        response_schema=Answer,
        temperature=min(temperature, 0.2),
    )
    answer = _coerce_answer(final)
    if answer is None:
        answer = Answer(
            answer=(final.content or "I could not produce a structured answer.").strip(),
            citations=[],
            confidence=0.2,
        )
    return AgentResult(answer=answer, tool_calls=executed, messages=messages)


async def _traced_complete(
    provider: LLMProvider,
    messages: list[Message],
    *,
    tools: list[ToolSpec] | None = None,
    response_schema: type[BaseModel] | None = None,
    temperature: float | None = None,
) -> ProviderResponse:
    model = provider.model
    attrs = {
        OP_NAME: "chat",
        PROVIDER_NAME: provider.name,
        REQUEST_MODEL: model,
        REQUEST_TEMPERATURE: temperature,
        AGENT_NAME_ATTR: AGENT_NAME,
    }
    if response_schema is not None:
        attrs[OUTPUT_TYPE] = "json"
    with start_span(f"chat {model}", kind=SpanKind.CLIENT, attributes=attrs) as span:
        response = await provider.complete(
            messages,
            tools=tools,
            response_schema=response_schema,
            temperature=temperature,
        )
        if response.tool_calls:
            span.set_attribute(RESPONSE_FINISH, ("tool_calls",))
        maybe_set(span, USAGE_INPUT, response.input_tokens)
        maybe_set(span, USAGE_OUTPUT, response.output_tokens)
        return response


def _coerce_answer(response: ProviderResponse) -> Answer | None:
    if isinstance(response.parsed, Answer):
        return response.parsed
    if response.parsed is not None:
        try:
            return Answer.model_validate(response.parsed.model_dump())
        except ValidationError:
            pass
    if not response.content:
        return None
    text = response.content.strip()
    try:
        return Answer.model_validate_json(text)
    except ValidationError:
        pass
    try:
        return Answer.model_validate(json.loads(text))
    except (ValidationError, json.JSONDecodeError, TypeError):
        return None


# Public alias: the Day 4 LangGraph nodes reuse the same traced ``chat {model}`` span.
traced_complete = _traced_complete
