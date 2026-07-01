"""OpenTelemetry tracing setup with a console exporter by default.

Swapping to an OTLP exporter (Jaeger, Honeycomb, etc.) for production is a matter of
changing the SpanProcessor registered in configure_tracing() -- call sites never
touch the SDK directly, they just use traced_span().
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from sentinel.config import settings

_configured = False


def configure_tracing(service_name: str = "sentinel") -> None:
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if settings.otel_console_export:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


@contextmanager
def traced_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[None]:
    configure_tracing()
    tracer = trace.get_tracer("sentinel")
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            span.set_attribute(key, str(value))
        yield
