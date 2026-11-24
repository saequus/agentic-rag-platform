"""Checkpointer lifecycle: Postgres (default) or in-memory.

``AsyncPostgresSaver.setup()`` creates and migrates its own tables
(``checkpoints``, ``checkpoint_blobs``, ``checkpoint_writes``,
``checkpoint_migrations``) in ``DATABASE_URL``. That is why there is no
``migrations/003_*.sql`` for them: LangGraph versions that schema itself and
``setup()`` is idempotent, so it runs on every API start.

The pool needs ``autocommit=True`` (``setup()`` runs ``CREATE INDEX
CONCURRENTLY``) and ``prepare_threshold=0`` (safe behind PgBouncer), with
``dict_row`` rows, as the LangGraph docs require.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from psycopg.rows import dict_row

from agentic_rag_platform.graph.builder import AgentGraph, build_graph
from agentic_rag_platform.settings import Settings

log = logging.getLogger(__name__)


@dataclass
class AgentRuntime:
    graph: AgentGraph
    checkpointer: str
    _pool: Any = field(default=None, repr=False)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


_runtime: AgentRuntime | None = None
_lock = asyncio.Lock()


async def open_postgres_saver(database_url: str) -> tuple[BaseCheckpointSaver[Any], Any]:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool

    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=True, timeout=10)
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver, pool


async def get_agent_runtime(settings: Settings) -> AgentRuntime:
    """Build the compiled graph once per process (lazily, on the serving loop)."""
    global _runtime
    if _runtime is not None:
        return _runtime
    async with _lock:
        if _runtime is not None:
            return _runtime
        saver: BaseCheckpointSaver[Any]
        pool: Any = None
        kind = settings.agent_checkpointer
        if kind == "postgres":
            try:
                saver, pool = await open_postgres_saver(settings.database_url)
            except Exception as exc:
                # Keep /chat usable for a demo, but say loudly that memory is not durable.
                log.warning(
                    "Postgres checkpointer unavailable (%s: %s); using in-memory checkpoints",
                    type(exc).__name__,
                    exc,
                )
                saver, kind = InMemorySaver(), "memory"
        else:
            saver = InMemorySaver()
        graph = build_graph(saver, max_rewrites=settings.agent_max_rewrites)
        _runtime = AgentRuntime(graph=graph, checkpointer=kind, _pool=pool)
        return _runtime


async def close_agent_runtime() -> None:
    global _runtime
    if _runtime is not None:
        await _runtime.close()
        _runtime = None


def reset_agent_runtime() -> None:
    """Tests: drop the cached runtime (in-memory checkpoints are lost)."""
    global _runtime
    _runtime = None
