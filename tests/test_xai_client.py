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
