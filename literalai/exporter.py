from datetime import datetime, timezone
import json
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from typing import Dict, List, Optional, Sequence, cast
import logging


from literalai.event_processor import EventProcessor
from literalai.helper import utc_now
from literalai.observability.generation import GenerationType
from literalai.observability.step import Step, StepDict


class LoggingSpanExporter(SpanExporter):
    def __init__(
        self,
        logger_name: str = "span_exporter",
        event_processor: Optional[EventProcessor] = None,
    ):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.event_processor = event_processor

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
                    and span.attributes.get("gen_ai.request.model", None) is not None
                    and self.event_processor is not None
                ):
                    step = self._create_step_from_span(span)
                    self.event_processor.add_event(cast(StepDict, step.to_dict()))

            return SpanExportResult.SUCCESS
        except Exception as e:
            self.logger.error(f"Failed to export spans: {e}")
            return SpanExportResult.FAILURE

    def shutdown(self):
        """Shuts down the exporter."""
        if self.event_processor is not None:
            return self.event_processor.flush_and_stop()

    def force_flush(self, timeout_millis: float = 30000) -> bool:
        """Force flush the exporter."""
        return True

    #     # TODO: Add generation promptid
    #     # TODO: Add generation variables
    #     # TODO: Check missing variables
    #     # TODO: ttFirstToken
    #     # TODO: duration
    #     # TODO: tokenThroughputInSeconds
    #     # TODO: Add tools
    #     # TODO: error check with gemini error
    def _create_step_from_span(self, span: ReadableSpan) -> Step:
        """Convert a span to a Step object"""
        attributes = span.attributes or {}

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

        generation_type = attributes.get("llm.request.type")
        is_chat = generation_type == "chat"

        span_props = {
            "parent_id": attributes.get(
                "traceloop.association.properties.literal.parent_id"
            ),
            "thread_id": attributes.get(
                "traceloop.association.properties.literal.thread_id"
            ),
            "root_run_id": attributes.get(
                "traceloop.association.properties.literal.root_run_id"
            ),
            "metadata": attributes.get(
                "traceloop.association.properties.literal.metadata"
            ),
            "tags": attributes.get("traceloop.association.properties.literal.tags"),
            "name": attributes.get("traceloop.association.properties.literal.name"),
        }

        span_props = {
            k: str(v) for k, v in span_props.items() if v is not None and v != "None"
        }

        generation_content = {
            "messages": (
                self.extract_messages(cast(Dict, attributes)) if is_chat else None
            ),
            "message_completion": (
                self.extract_messages(cast(Dict, attributes), "gen_ai.completion.")[0]
                if is_chat
                else None
            ),
            "prompt": attributes.get("gen_ai.prompt.0.user"),
            "completion": attributes.get("gen_ai.completion.0.content"),
            "model": attributes.get("gen_ai.request.model"),
            "provider": attributes.get("gen_ai.system"),
        }
        generation_settings = {
            "max_tokens": attributes.get("gen_ai.request.max_tokens"),
            "stream": attributes.get("llm.is_streaming"),
            "token_count": attributes.get("llm.usage.total_tokens"),
            "input_token_count": attributes.get("gen_ai.usage.prompt_tokens"),
            "output_token_count": attributes.get("gen_ai.usage.completion_tokens"),
            "frequency_penalty": attributes.get("gen_ai.request.frequency_penalty"),
            "presence_penalty": attributes.get("gen_ai.request.presence_penalty"),
            "temperature": attributes.get("gen_ai.request.temperature"),
            "top_p": attributes.get("gen_ai.request.top_p"),
        }

        step_dict = {
            "id": str(span.context.span_id) if span.context else None,
            "name": span_props.get("name", span.name),
            "type": "llm",
            "metadata": self.extract_json(span_props.get("metadata", "{}")),
            "startTime": start_time,
            "endTime": end_time,
            "threadId": span_props.get("thread_id"),
            "parentId": span_props.get("parent_id"),
            "rootRunId": span_props.get("root_run_id"),
            "tags": self.extract_json(span_props.get("tags", "[]")),
            "input": {
                "content": (
                    generation_content["messages"]
                    if is_chat
                    else generation_content["prompt"]
                )
            },
            "output": {
                "content": (
                    generation_content["message_completion"]
                    if is_chat
                    else generation_content["completion"]
                )
            },
            "generation": {
                "type": GenerationType.CHAT if is_chat else GenerationType.COMPLETION,
                "prompt": generation_content["prompt"] if not is_chat else None,
                "completion": generation_content["completion"] if not is_chat else None,
                "model": generation_content["model"],
                "provider": generation_content["provider"],
                "settings": generation_settings,
                "tokenCount": generation_settings["token_count"],
                "inputTokenCount": generation_settings["input_token_count"],
                "outputTokenCount": generation_settings["output_token_count"],
                "messages": generation_content["messages"],
                "messageCompletion": generation_content["message_completion"],
            },
        }

        step = Step.from_dict(cast(StepDict, step_dict))

        if not span.status.is_ok:
            step.error = span.status.description or "Unknown error"

        return step

    def extract_messages(
        self, data: Dict, prefix: str = "gen_ai.prompt."
    ) -> List[Dict]:
        messages = []
        index = 0

        while True:
            role_key = f"{prefix}{index}.role"
            content_key = f"{prefix}{index}.content"

            if role_key not in data or content_key not in data:
                break

            messages.append(
                {
                    "role": data[role_key],
                    "content": self.extract_json(data[content_key]),
                }
            )

            index += 1

        return messages

    def extract_json(self, data: str) -> Dict | List | str:
        try:
            content = json.loads(data)
        except Exception:
            content = data

        return content
