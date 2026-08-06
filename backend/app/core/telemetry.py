"""可选 OpenTelemetry 初始化；导出失败不得影响业务请求。"""

import logging
from contextlib import contextmanager, nullcontext
from typing import Iterator

from .config import settings

logger = logging.getLogger(__name__)
_configured = False


def configure_telemetry() -> None:
    """按配置初始化 OTLP 导出器，默认关闭。"""
    global _configured
    if _configured or not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": settings.otel_service_name})
        )
        if settings.otel_exporter_endpoint.strip():
            endpoint = settings.otel_exporter_endpoint.strip()
            exporter = OTLPSpanExporter(
                endpoint=endpoint,
                insecure=endpoint.startswith("http://"),
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _configured = True
        logger.info("OpenTelemetry 已启用")
    except Exception as exc:
        logger.warning("OpenTelemetry 初始化失败，不影响业务: %s", exc)


@contextmanager
def trace_span(name: str) -> Iterator[None]:
    """创建不含正文内容的阶段 Span；未启用时为空操作。"""
    span_context = nullcontext()
    if settings.otel_enabled:
        try:
            from opentelemetry import trace

            span_context = trace.get_tracer("zhiyu.agent").start_as_current_span(name)
        except Exception as exc:
            logger.debug("创建 OTel Span 失败: %s", exc)
    with span_context:
        yield
