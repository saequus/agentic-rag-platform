"""OpenTelemetry instrumentation: in-memory exporter, no network, no secrets."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agentic_rag_platform.agent.loop import run_agent
from agentic_rag_platform.ingest.embedder import HashEmbedder
from agentic_rag_platform.ingest.pipeline import _embed_queries
from agentic_rag_platform.main import app
from agentic_rag_platform.settings import Settings
from agentic_rag_platform.telemetry import configure_telemetry, instrument_app
from agentic_rag_platform.telemetry.tracing import AGENT_NAME
from agentic_rag_platform.tools.fx_rate import NBP_RATE_URL
from agentic_rag_platform.tools.registry import execute_tool
from tests.fakes import ScriptedProvider, answer, tool_request


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    assert configure_telemetry(Settings(), span_exporter=exporter) is not None
    instrument_app(app)
    yield exporter
    exporter.clear()


def _by_name(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _attrs(span: ReadableSpan) -> Mapping[str, object]:
    attributes = span.attributes
    assert attributes is not None
    return dict(attributes)


def test_sdk_disabled_is_a_noop() -> None:
    exporter = InMemorySpanExporter()
    assert configure_telemetry(Settings(otel_sdk_disabled=True), span_exporter=exporter) is None


@pytest.mark.asyncio
async def test_invoke_agent_and_chat_spans(spans: InMemorySpanExporter) -> None:
    provider = ScriptedProvider([answer("hello", confidence=0.8)])
    result = await run_agent(provider, "Say hello", tools=[])
    assert result.answer.answer == "hello"

    agent = _by_name(spans, f"invoke_agent {AGENT_NAME}")
    assert len(agent) == 1
    assert _attrs(agent[0])["gen_ai.operation.name"] == "invoke_agent"
    assert _attrs(agent[0])["gen_ai.agent.name"] == AGENT_NAME
    assert _attrs(agent[0])["gen_ai.provider.name"] == "fake"
    assert _attrs(agent[0])["gen_ai.tool.call.count"] == 0
    assert agent[0].status.status_code != StatusCode.ERROR

    chat = _by_name(spans, "chat fake-model")
    assert len(chat) == 1
    assert _attrs(chat[0])["gen_ai.operation.name"] == "chat"
    assert _attrs(chat[0])["gen_ai.request.model"] == "fake-model"
    # Prompts must not be attached by default (PII / data-minimisation).
    keys = set(_attrs(chat[0]).keys())
    assert "gen_ai.input.messages" not in keys
    assert "gen_ai.output.messages" not in keys
    assert "gen_ai.prompt" not in keys


@respx.mock
@pytest.mark.asyncio
async def test_tool_and_chat_spans_for_fx_rate(spans: InMemorySpanExporter) -> None:
    respx.get(NBP_RATE_URL.format(code="usd")).mock(
        return_value=httpx.Response(
            200,
            json={"rates": [{"effectiveDate": "2026-11-19", "mid": 4.25}]},
        )
    )
    provider = ScriptedProvider(
        [
            tool_request("call_1", "get_fx_rate", base="USD", quote="PLN"),
            answer("1 USD is 4.25 PLN.", confidence=0.93),
        ]
    )
    result = await run_agent(provider, "How many PLN is one US dollar?")
    assert result.tool_calls[0].name == "get_fx_rate"

    tool = _by_name(spans, "execute_tool get_fx_rate")
    assert len(tool) == 1
    assert _attrs(tool[0])["gen_ai.operation.name"] == "execute_tool"
    assert _attrs(tool[0])["gen_ai.tool.name"] == "get_fx_rate"
    assert _attrs(tool[0])["gen_ai.tool.call.id"] == "call_1"
    # Tool arguments and results stay off the span.
    keys = set(_attrs(tool[0]).keys())
    assert "gen_ai.tool.call.arguments" not in keys
    assert "gen_ai.tool.call.result" not in keys

    agent = _by_name(spans, f"invoke_agent {AGENT_NAME}")
    assert _attrs(agent[0])["gen_ai.tool.call.count"] == 1


@pytest.mark.asyncio
async def test_unknown_tool_marks_span_error(spans: InMemorySpanExporter) -> None:
    payload = await execute_tool("does_not_exist", {})
    assert json.loads(payload)["error"].startswith("Unknown tool")
    tool = _by_name(spans, "execute_tool does_not_exist")
    assert len(tool) == 1
    assert tool[0].status.status_code == StatusCode.ERROR
    assert _attrs(tool[0])["error.type"] == "unknown_tool"


def test_embeddings_span_has_model_not_input_text(spans: InMemorySpanExporter) -> None:
    embedder = HashEmbedder()
    vectors = _embed_queries(embedder, ["secret personal query that must not be attributed"])
    assert len(vectors) == 1
    embs = [s for s in spans.get_finished_spans() if s.name.startswith("embeddings ")]
    assert len(embs) == 1
    attrs = embs[0].attributes
    assert attrs is not None
    assert attrs["gen_ai.operation.name"] == "embeddings"
    assert attrs["gen_ai.request.model"] == embedder.name
    blob = " ".join(f"{k}={v!r}" for k, v in attrs.items())
    assert "secret personal query" not in blob


def test_health_is_excluded_from_http_spans(spans: InMemorySpanExporter) -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    http_spans = [
        span
        for span in spans.get_finished_spans()
        if "health" in span.name.lower() or (span.attributes or {}).get("http.route") == "/health"
    ]
    assert http_spans == []
