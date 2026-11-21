"""Manual span helpers aligned with OpenTelemetry GenAI semantic conventions.

Conventions are in Development status and live in the dedicated repository:
https://github.com/open-telemetry/semantic-conventions-genai
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

from agentic_rag_platform import __version__

TRACER_NAME = "agentic_rag_platform"
AGENT_NAME = "finance_assistant"

# Attribute keys from the GenAI conventions (Development, Nov 2026).
OP_NAME = "gen_ai.operation.name"
PROVIDER_NAME = "gen_ai.provider.name"
AGENT_NAME_ATTR = "gen_ai.agent.name"
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_TEMPERATURE = "gen_ai.request.temperature"
OUTPUT_TYPE = "gen_ai.output.type"
USAGE_INPUT = "gen_ai.usage.input_tokens"
USAGE_OUTPUT = "gen_ai.usage.output_tokens"
RESPONSE_FINISH = "gen_ai.response.finish_reasons"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_TYPE = "gen_ai.tool.type"
RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"
EMBEDDINGS_DIM = "gen_ai.embeddings.dimension.count"
ERROR_TYPE = "error.type"
TENANT_ID = "tenant.id"


def tracer() -> Tracer:
    return trace.get_tracer(TRACER_NAME, __version__)


def _clean_attrs(attributes: Mapping[str, AttributeValue | None]) -> dict[str, AttributeValue]:
    return {key: value for key, value in attributes.items() if value is not None}


@contextmanager
def start_span(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Mapping[str, AttributeValue | None] | None = None,
) -> Iterator[Span]:
    """Start a span, mark errors, and re-raise.

    Attributes that matter for sampling are passed at creation time.
    """
    attrs = _clean_attrs(attributes or {})
    with tracer().start_as_current_span(name, kind=kind, attributes=attrs) as span:
        try:
            yield span
        except Exception as exc:
            record_exception(span, exc)
            raise


def record_exception(span: Span, exc: BaseException) -> None:
    span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
    span.set_attribute(ERROR_TYPE, type(exc).__name__)
    span.record_exception(exc)


def set_status_error(span: Span, message: str, error_type: str = "error") -> None:
    span.set_status(Status(StatusCode.ERROR, message[:200]))
    span.set_attribute(ERROR_TYPE, error_type)


def maybe_set(span: Span, key: str, value: AttributeValue | None) -> None:
    if value is not None:
        span.set_attribute(key, value)


def document_refs(hits: Sequence[Any]) -> list[dict[str, Any]]:
    """Retrieval documents with id and score only — no chunk text (PII-safe)."""
    refs: list[dict[str, Any]] = []
    for hit in hits:
        doc_id = getattr(hit, "doc_id", None)
        chunk_no = getattr(hit, "chunk_no", None)
        ident = f"{doc_id}:{chunk_no}" if doc_id is not None else None
        if ident is None:
            continue
        refs.append({"id": ident, "score": float(getattr(hit, "score", 0.0))})
    return refs
