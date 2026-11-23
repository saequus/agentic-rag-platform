"""Tracer provider, exporters, and auto-instrumentation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_OFF,
    ALWAYS_ON,
    ParentBased,
    Sampler,
    TraceIdRatioBased,
)

from agentic_rag_platform import __version__

if TYPE_CHECKING:
    from fastapi import FastAPI

    from agentic_rag_platform.settings import Settings

logger = logging.getLogger(__name__)

_provider: TracerProvider | None = None
_libraries_instrumented = False


def _sampler_from_settings(settings: Settings) -> Sampler:
    name = settings.otel_traces_sampler.lower()
    arg = settings.otel_traces_sampler_arg
    if name in {"always_on", "alwayson"}:
        return ALWAYS_ON
    if name in {"always_off", "alwaysoff"}:
        return ALWAYS_OFF
    if name in {"traceidratio", "trace_id_ratio"}:
        return TraceIdRatioBased(arg)
    # Default and official "parentbased_traceidratio".
    if name in {"parentbased_always_on", "parentbasedalwayson"}:
        return ParentBased(ALWAYS_ON)
    if name in {"parentbased_always_off", "parentbasedalwaysoff"}:
        return ParentBased(ALWAYS_OFF)
    return ParentBased(TraceIdRatioBased(arg))


def configure_telemetry(
    settings: Settings | None = None,
    *,
    span_exporter: SpanExporter | None = None,
) -> TracerProvider | None:
    """Install a TracerProvider when an exporter is configured.

    ``span_exporter`` is for tests (typically ``InMemorySpanExporter``). In
    production, exporters come from ``OTEL_EXPORTER_OTLP_ENDPOINT`` and
    ``OTEL_TRACES_EXPORTER``. Honours ``OTEL_SDK_DISABLED``.
    """
    global _provider
    from agentic_rag_platform.settings import get_settings

    settings = settings or get_settings()
    if settings.otel_sdk_disabled:
        logger.info("OpenTelemetry SDK disabled (OTEL_SDK_DISABLED)")
        return None

    endpoint = (settings.otel_exporter_otlp_endpoint or "").strip() or None
    wants_otlp = endpoint is not None and "otlp" in settings.otel_traces_exporter.lower()
    wants_console = "console" in settings.otel_traces_exporter.lower()
    if span_exporter is None and not wants_otlp and not wants_console:
        return _provider

    if _provider is None:
        resource = Resource.create(
            {
                "service.name": settings.otel_service_name,
                "service.version": __version__,
                "deployment.environment": settings.otel_environment,
            }
        )
        provider = TracerProvider(resource=resource, sampler=_sampler_from_settings(settings))
        # If something else already installed a provider, keep adding to ours
        # only when we can actually set it.
        current = trace.get_tracer_provider()
        if not isinstance(current, TracerProvider):
            trace.set_tracer_provider(provider)
            _provider = provider
        else:
            _provider = current

    if span_exporter is not None:
        _provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    if wants_otlp and span_exporter is None:
        _add_otlp_exporter(settings)

    if wants_console and span_exporter is None:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    return _provider


def _add_otlp_exporter(settings: Settings) -> None:
    assert _provider is not None
    protocol = settings.otel_exporter_otlp_protocol.lower()
    # HTTP/protobuf is the default and the only protocol packaged here.
    # gRPC (port 4317) works via the Collector; add
    # opentelemetry-exporter-otlp-proto-grpc if you need to speak gRPC from the app.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    # Leave endpoint unset so the exporter reads OTEL_EXPORTER_OTLP_* itself
    # when the standard env var is already present. If only settings has it
    # (e.g. from a dotenv file), pass it explicitly.
    endpoint: str | None = None
    if settings.otel_exporter_otlp_endpoint:
        endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/")
        if protocol.startswith("http") and not endpoint.endswith("/v1/traces"):
            endpoint = f"{endpoint}/v1/traces"
    _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    logger.info("OTLP span exporter configured (%s)", protocol)


def instrument_app(app: FastAPI) -> None:
    """Instrument psycopg and httpx once.

    HTTP server spans come from FastAPI 0.165 native OpenTelemetry, configured
    on the app with an exclude list for /health and docs. We do not also call
    FastAPIInstrumentor, which would duplicate those spans.
    """
    global _libraries_instrumented
    if not _libraries_instrumented:
        try:
            from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

            PsycopgInstrumentor().instrument()
        except Exception:  # pragma: no cover - optional if the extra is missing
            logger.warning("psycopg instrumentation could not be enabled", exc_info=True)
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

            HTTPXClientInstrumentor().instrument()
        except Exception:  # pragma: no cover
            logger.warning("httpx instrumentation could not be enabled", exc_info=True)
        _libraries_instrumented = True


def shutdown_telemetry() -> None:
    """Flush, then shut down exporters.

    Under pytest we only flush: shutting the global provider down would break
    later tests in the same process.
    """
    import os

    if _provider is None:
        return
    _provider.force_flush()
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    _provider.shutdown()
