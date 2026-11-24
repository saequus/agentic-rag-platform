"""Demo *write* tool for the Day 4 human-in-the-loop interrupt.

``save_note`` stands in for any side-effecting action (create a ticket, send
an email, book a trade request). It is deliberately **not** in
``default_tools()``, so the Day 1 loop cannot call it. The LangGraph agent only
runs it after a human approves via ``POST /chat/resume``.

Storage is an in-process dict: enough to show the approval gate, not a real
notes service.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from agentic_rag_platform.tools.registry import ToolSpec

NOTES: dict[str, dict[str, Any]] = {}


class SaveNoteArgs(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=4000)
    tenant_id: str = "default"


async def _save_note_handler(**kwargs: Any) -> dict[str, Any]:
    args = SaveNoteArgs.model_validate(kwargs)
    note_id = uuid.uuid4().hex[:12]
    NOTES[note_id] = {
        "note_id": note_id,
        "tenant_id": args.tenant_id,
        "title": args.title,
        "text": args.text,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return {"status": "saved", "note_id": note_id, "title": args.title}


SAVE_NOTE = ToolSpec(
    name="save_note",
    description="Save a short note for the user. This is a write action and needs approval.",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "text": {"type": "string"},
            "tenant_id": {"type": "string"},
        },
        "required": ["title", "text"],
        "additionalProperties": False,
    },
    handler=_save_note_handler,
)

WRITE_TOOLS = frozenset({SAVE_NOTE.name})
