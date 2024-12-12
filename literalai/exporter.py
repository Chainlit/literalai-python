from datetime import datetime, timezone
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from typing import Sequence
import logging

from literalai.helper import utc_now
from literalai.observability.step import Step

from literalai.context import active_root_run_var, active_steps_var, active_thread_var


class LoggingSpanExporter(SpanExporter):
    def __init__(self, logger_name: str = "span_exporter"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export the spans by logging them."""
        try:
            for span in spans:
                if (
                    span.attributes
                    and span.attributes.get("llm.is_streaming", None) is not None
                ):
                    step = self._create_step_from_span(span)
                    self.logger.info(f"Created step from span: {step.to_dict()}")

            return SpanExportResult.SUCCESS
        except Exception as e:
            self.logger.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self):
        """Shuts down the exporter."""
        pass

    def force_flush(self, timeout_millis: float = 30000) -> bool:
        """Force flush the exporter."""
        return True

    def _create_step_from_span(self, span: ReadableSpan) -> Step:
        """Convert a span to a Step object"""
        start_time = (
            datetime.fromtimestamp(span.start_time / 1e9, tz=timezone.utc).isoformat()
            if span.start_time
            else utc_now()
        )
        end_time = (
            datetime.fromtimestamp(span.end_time / 1e9, tz=timezone.utc).isoformat()
            if span.end_time
            else utc_now()
        )

        self.logger.info(span.attributes)
        parent_id = span.attributes.get("literal.parent_id")
        thread_id = span.attributes.get("literal.thread_id")
        root_run_id = span.attributes.get("literal.root_run_id")

        self.logger.info(
            f"From span attributes - Parent ID: {parent_id}, Thread ID: {thread_id}, Root Run ID: {root_run_id}"
        )

        step = Step(
            id=(str(span.context.span_id) if span.context else None),
            name=span.name,
            type="llm",
            start_time=start_time,
            end_time=end_time,
            thread_id=thread_id,
            parent_id=parent_id,
            root_run_id=root_run_id,
        )

        if span.status.is_ok:
            step.error = span.status.description or "Unknown error"

        # Handle input/output/generation based on span attributes
        # (We'll implement this in the next iteration)

        return step
