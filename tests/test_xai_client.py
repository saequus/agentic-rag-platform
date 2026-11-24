"""The xAI gRPC channel has to be opened on a running event loop."""

import asyncio

import pytest

from agentic_rag_platform.providers.xai import XAIProvider


@pytest.mark.asyncio
async def test_provider_can_be_built_off_loop_and_opened_on_it() -> None:
    # FastAPI runs sync dependencies on an AnyIO worker thread, which has no loop.
    provider = await asyncio.to_thread(lambda: XAIProvider("test-key", "grok-4.7"))

    client = provider._client_on_running_loop()
    try:
        assert client is provider._client_on_running_loop()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_stream_text_yields_chunk_contents_without_network() -> None:
    from collections.abc import AsyncIterator
    from types import SimpleNamespace
    from typing import Any

    from agentic_rag_platform.schemas import Message

    created: dict[str, Any] = {}

    class FakeChat:
        def __init__(self) -> None:
            self.messages: list[Any] = []

        def append(self, message: Any) -> None:
            self.messages.append(message)

        async def stream(self) -> AsyncIterator[tuple[Any, Any]]:
            for text in ["Stopa ", "", "5,35% [1]"]:
                yield SimpleNamespace(), SimpleNamespace(content=text)

    def create(**kwargs: Any) -> FakeChat:
        chat = FakeChat()
        created.update(kwargs, chat=chat)
        return chat

    provider = XAIProvider("test-key", "grok-4.7")
    provider._client = SimpleNamespace(chat=SimpleNamespace(create=create))  # type: ignore[assignment]
    parts = [
        delta
        async for delta in provider.stream_text(
            [Message(role="system", content="s"), Message(role="user", content="q")],
            temperature=0.1,
        )
    ]
    assert parts == ["Stopa ", "5,35% [1]"]
    assert created["model"] == "grok-4.7"
    assert created["temperature"] == 0.1
    assert len(created["chat"].messages) == 2
