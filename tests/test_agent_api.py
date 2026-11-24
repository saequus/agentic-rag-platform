"""HTTP layer for the LangGraph agent: /chat, /chat/stream (SSE), /chat/resume."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentic_rag_platform.graph.routes import retrieve_fn_dep
from agentic_rag_platform.main import app, provider_dep
from agentic_rag_platform.tools.notes import NOTES
from tests.fakes import FakeAgentLLM, FakeRetrieve, hit

EDO = hit("pl-bond-edo-108-2026", "EDO1136: w pierwszym okresie odsetkowym 5,35%.", page=2)


@pytest.fixture
def agent(client: TestClient) -> Callable[..., FakeAgentLLM]:
    def _install(**kwargs: Any) -> FakeAgentLLM:
        llm = FakeAgentLLM(**kwargs)
        retrieve = FakeRetrieve([EDO])
        app.dependency_overrides[provider_dep] = lambda: llm
        app.dependency_overrides[retrieve_fn_dep] = lambda: retrieve
        return llm

    return _install


def _sse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_chat_returns_grounded_answer_with_citations(
    client: TestClient, agent: Callable[..., FakeAgentLLM]
) -> None:
    agent()
    response = client.post("/chat", json={"message": "Jakie jest oprocentowanie EDO1136?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["answer"] == "The first-period rate is 5.35% [1]."
    assert body["citations"][0] == {
        "n": 1,
        "source": "pl-bond-edo-108-2026 p. 2",
        "doc_id": "pl-bond-edo-108-2026",
        "chunk_no": 0,
        "page": 2,
        "section": None,
        "score": 0.5,
        "url": None,
    }
    assert body["grounded"] is True
    assert body["route"] == "retrieve"
    assert body["checkpointer"] == "memory"
    assert body["provider"] == "fake"
    assert len(body["thread_id"]) == 32


def test_thread_memory_over_http(client: TestClient, agent: Callable[..., FakeAgentLLM]) -> None:
    agent()
    first = client.post("/chat", json={"message": "EDO1136?", "thread_id": "demo-1"}).json()
    client.post("/chat", json={"message": "A COI?", "thread_id": "demo-1"})
    thread = client.get("/chat/demo-1").json()
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant", "user", "assistant"]
    assert thread["messages"][1]["content"] == first["answer"]
    assert client.get("/chat/demo-1", params={"tenant_id": "other"}).status_code == 404
    other = client.post("/chat", json={"message": "x", "thread_id": "demo-1", "tenant_id": "other"})
    assert other.status_code == 403


def test_chat_stream_emits_nodes_tokens_and_done(
    client: TestClient, agent: Callable[..., FakeAgentLLM]
) -> None:
    agent()
    with client.stream("POST", "/chat/stream", json={"message": "EDO1136?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    events = _sse_events(body)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    nodes = [data["node"] for kind, data in events if kind == "node"]
    assert nodes == ["route", "retrieve", "grade_docs", "generate", "check_grounded", "finalize"]
    tokens = "".join(data["text"] for kind, data in events if kind == "token")
    done = events[-1][1]
    assert tokens == done["answer"]
    assert done["citations"][0]["doc_id"] == "pl-bond-edo-108-2026"
    retrieve_event = next(data for kind, data in events if data.get("node") == "retrieve")
    assert "text" not in retrieve_event["docs"][0]  # node events carry refs, not passages


def test_interrupt_and_resume_over_http(
    client: TestClient, agent: Callable[..., FakeAgentLLM]
) -> None:
    NOTES.clear()
    agent(route="tool", tool_name="save_note")
    pending = client.post("/chat", json={"message": "Save a note", "thread_id": "hitl-1"}).json()
    assert pending["status"] == "interrupted"
    assert pending["answer"] is None
    assert pending["pending_approval"]["tool"] == "save_note"
    assert pending["pending_approval"]["arguments"]["text"] == "Check EDO1136 rate"

    blocked = client.post("/chat", json={"message": "hello?", "thread_id": "hitl-1"})
    assert blocked.status_code == 409
    assert client.get("/chat/hitl-1").json()["pending_approval"]["tool"] == "save_note"

    done = client.post("/chat/resume", json={"thread_id": "hitl-1", "approved": True}).json()
    assert done["status"] == "completed"
    assert done["answer"].startswith("Saved the note")
    assert len(NOTES) == 1

    again = client.post("/chat/resume", json={"thread_id": "hitl-1", "approved": True})
    assert again.status_code == 409


def test_interrupt_over_sse(client: TestClient, agent: Callable[..., FakeAgentLLM]) -> None:
    agent(route="tool", tool_name="save_note")
    with client.stream(
        "POST", "/chat/stream", json={"message": "Save a note", "thread_id": "hitl-2"}
    ) as response:
        body = "".join(response.iter_text())
    events = _sse_events(body)
    interrupt = next(data for kind, data in events if kind == "interrupt")
    assert interrupt["payload"]["tool"] == "save_note"
    assert events[-1][1]["status"] == "interrupted"


def test_resume_unknown_thread_is_409(
    client: TestClient, agent: Callable[..., FakeAgentLLM]
) -> None:
    agent()
    response = client.post("/chat/resume", json={"thread_id": "nope", "approved": True})
    assert response.status_code == 409


def test_chat_validates_thread_id(client: TestClient, agent: Callable[..., FakeAgentLLM]) -> None:
    agent()
    response = client.post("/chat", json={"message": "x", "thread_id": "bad id with spaces"})
    assert response.status_code == 422


def test_agent_chat_without_key_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "")
    app.dependency_overrides.pop(provider_dep, None)
    app.dependency_overrides[retrieve_fn_dep] = lambda: FakeRetrieve()
    response = TestClient(app).post("/chat", json={"message": "hello"})
    assert response.status_code == 503
    assert "XAI_API_KEY" in response.json()["detail"]
