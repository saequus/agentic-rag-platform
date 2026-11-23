"""OpenTelemetry setup and GenAI semantic-convention helpers.

Tracing is off unless an exporter is configured (an OTLP endpoint, a console
exporter, or an in-memory exporter in tests). Official ``OTEL_*`` environment
variables are honoured; see ``docs/otel-notes.md``.
"""

from agentic_rag_platform.telemetry.setup import (
    configure_telemetry,
    instrument_app,
    shutdown_telemetry,
)
from agentic_rag_platform.telemetry.tracing import (
    AGENT_NAME,
    start_span,
    tracer,
)

__all__ = [
    "AGENT_NAME",
    "configure_telemetry",
    "instrument_app",
    "shutdown_telemetry",
    "start_span",
    "tracer",
]
