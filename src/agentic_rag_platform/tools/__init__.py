"""Built-in tools the agent can call. Day 1 ships ``get_fx_rate``."""

from agentic_rag_platform.tools.registry import ToolSpec, default_tools, execute_tool, get_tool

__all__ = ["ToolSpec", "default_tools", "execute_tool", "get_tool"]
