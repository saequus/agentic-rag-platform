"""OpenAI sampling parameters that some models reject."""

from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from openai import BadRequestError

from agentic_rag_platform.providers.openai_provider import (
    OpenAIProvider,
    accepts_custom_temperature,
)
from agentic_rag_platform.schemas import Message


@pytest.mark.parametrize(
    ("model", "accepted"),
    [
        ("gpt-4o", True),
        ("gpt-4.1-mini", True),
        ("gpt-5-nano", False),
        ("gpt-5", False),
        ("gpt-5.1", False),
        ("o3-mini", False),
        ("o1", False),
    ],
)
def test_which_models_accept_custom_temperature(model: str, accepted: bool) -> None:
    assert accepts_custom_temperature(model) is accepted


def _completion(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _temperature_error() -> BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(
        "Error code: 400 - Unsupported value: 'temperature' does not support 0.2 "
        "with this model. Only the default (1) value is supported.",
        response=cast(Any, response),
        body=None,
    )


@pytest.mark.asyncio
async def test_gpt5_nano_does_not_send_temperature() -> None:
    provider = OpenAIProvider("test-key", "gpt-5-nano", temperature=0.2)
    calls: list[dict[str, object]] = []

    async def create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return _completion("ok")

    provider._client.chat.completions.create = create  # type: ignore[assignment]
    result = await provider.complete([Message(role="user", content="hi")])

    assert result.content == "ok"
    assert calls == [
        {"model": "gpt-5-nano", "messages": [{"role": "user", "content": "hi"}]},
    ]


@pytest.mark.asyncio
async def test_custom_temperature_is_sent_for_gpt4o() -> None:
    provider = OpenAIProvider("test-key", "gpt-4o", temperature=0.2)
    calls: list[dict[str, object]] = []

    async def create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return _completion("ok")

    provider._client.chat.completions.create = create  # type: ignore[assignment]
    await provider.complete([Message(role="user", content="hi")], temperature=0.4)

    assert calls[0]["temperature"] == 0.4


@pytest.mark.asyncio
async def test_retries_once_when_api_rejects_temperature() -> None:
    provider = OpenAIProvider("test-key", "gpt-4o", temperature=0.2)
    calls: list[dict[str, object]] = []

    async def create(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        if len(calls) == 1:
            raise _temperature_error()
        return _completion("ok")

    provider._client.chat.completions.create = create  # type: ignore[assignment]
    result = await provider.complete([Message(role="user", content="hi")])

    assert result.content == "ok"
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
