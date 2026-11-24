"""The same retrieve -> answer flow in Pydantic AI (Day 4 comparison, ~50 lines of logic).

Compare with ``graph/``: here the model decides when to call ``search_documents``
(a tool), typed deps carry the retriever, and ``output_type`` validates the
answer. There is no explicit grading/rewrite loop, checkpointer or interrupt.

    uv run python -m agentic_rag_platform.agent.pydantic_ai_agent "Oprocentowanie EDO1136?"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from agentic_rag_platform.graph.nodes import RetrieveFn
from agentic_rag_platform.graph.prompts import DISCLAIMER
from agentic_rag_platform.settings import Settings, get_settings


@dataclass
class RagDeps:
    retrieve: RetrieveFn
    tenant_id: str = "default"
    k: int = 5


class CitedAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(description="doc_id and page of every source used")
    confidence: float = Field(ge=0.0, le=1.0)


INSTRUCTIONS = (
    "Answer questions about ECB bulletins and Polish treasury bonds. Always call "
    "search_documents first and answer only from its results; cite doc_id and page. "
    f"If the results do not answer the question, say you don't know. {DISCLAIMER}"
)


def build_agent(model: Model | str) -> Agent[RagDeps, CitedAnswer]:
    agent = Agent(model, deps_type=RagDeps, output_type=CitedAnswer, instructions=INSTRUCTIONS)

    @agent.tool
    async def search_documents(ctx: RunContext[RagDeps], query: str) -> list[dict[str, Any]]:
        """Search the indexed finance documents (hybrid retrieval)."""
        hits = await ctx.deps.retrieve(
            query, tenant_id=ctx.deps.tenant_id, k=ctx.deps.k, strategy=None, backend=None
        )
        return [
            {"doc_id": h.doc_id, "page": h.page, "text": h.text[:900], "score": h.score}
            for h in hits
        ]

    return agent


def default_model(settings: Settings) -> Model:
    """xAI is the default here too (pydantic-ai ships an xai-sdk based model)."""
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

    key = settings.xai_api_key.get_secret_value() if settings.xai_api_key else None
    return XaiModel(settings.xai_model, provider=XaiProvider(api_key=key))


def main(argv: list[str]) -> int:  # pragma: no cover - needs a key and an index
    from agentic_rag_platform.graph.routes import retrieve_fn_dep

    settings = get_settings()
    agent = build_agent(default_model(settings))
    deps = RagDeps(retrieve=retrieve_fn_dep(settings), tenant_id=settings.default_tenant_id)
    result = agent.run_sync(" ".join(argv) or "What is EDO1136?", deps=deps)
    print(result.output.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
