#!/usr/bin/env python3
"""Optional live check against the real xAI API.

Usage:
    cp .env.example .env   # then set XAI_API_KEY
    make smoke             # or: uv run python scripts/smoke.py

This script is never run in CI. It exits 2 if no key is configured.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running from the repo root without installing editable in odd layouts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag_platform.agent.loop import run_agent  # noqa: E402
from agentic_rag_platform.providers.factory import get_provider  # noqa: E402
from agentic_rag_platform.settings import get_settings  # noqa: E402


async def _main() -> int:
    load_dotenv()
    get_settings.cache_clear()
    settings = get_settings()
    if not settings.xai_api_key or not settings.xai_api_key.get_secret_value():
        print(
            "XAI_API_KEY is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 2

    provider = get_provider(settings)
    prompt = os.environ.get(
        "SMOKE_PROMPT",
        "What is the mid rate of 1 USD in PLN today? Cite your source.",
    )
    print(f"Provider={provider.name} model={provider.model}")
    print(f"Prompt: {prompt}")
    result = await run_agent(
        provider,
        prompt,
        max_iterations=settings.max_tool_iterations,
        temperature=settings.llm_temperature,
    )
    print(json.dumps(result.answer.model_dump(), indent=2, ensure_ascii=False))
    if result.tool_calls:
        print("Tool calls:")
        for call in result.tool_calls:
            print(f"  {call.name}({call.arguments}) -> {call.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
