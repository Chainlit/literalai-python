from datetime import datetime, timezone
import json
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from typing import Dict, List, Optional, Sequence, cast
import logging

from literalai.event_processor import EventProcessor
from literalai.helper import utc_now
from literalai.observability.generation import GenerationType
from literalai.observability.step import Step


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
                    self.event_processor.add_event(step.to_dict())
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
        parent_id = None
        thread_id = None
        root_run_id = None
        metadata = None
        name = None
        tags = None
        generation_messages = None
        generation_message_completion = None
        generation_prompt = None
        generation_completion = None
        generation_type = None
        generation_model = None
        generation_provider = None
        generation_max_tokens = None
        generation_stream = None
        generation_token_count = None
        generation_input_token_count = None
        generation_output_token_count = None
        generation_frequency_penalty = None
        generation_logit_bias = None
        generation_logprobs = None
        generation_top_logprobs = None
        generation_n = None
        generation_presence_penalty = None
        generation_response_format = None
        generation_seed = None
        generation_stop = None
        generation_temperature = None
        generation_top_p = None
        generation_tool_choice = None

        if span.attributes is not None:
            self.logger.info(span.attributes)
            parent_id = (
                span.attributes.get(
                    "traceloop.association.properties.literal.parent_id"
                )
                if span.attributes.get(
                    "traceloop.association.properties.literal.parent_id"
                )
                and span.attributes.get(
                    "traceloop.association.properties.literal.parent_id"
                )
                != "None"
                else None
            )
            thread_id = (
                span.attributes.get(
                    "traceloop.association.properties.literal.thread_id"
                )
                if span.attributes.get(
                    "traceloop.association.properties.literal.thread_id"
                )
                and span.attributes.get(
                    "traceloop.association.properties.literal.thread_id"
                )
                != "None"
                else None
            )
            root_run_id = (
                span.attributes.get(
                    "traceloop.association.properties.literal.root_run_id"
                )
                if span.attributes.get(
                    "traceloop.association.properties.literal.root_run_id"
                )
                and span.attributes.get(
                    "traceloop.association.properties.literal.root_run_id"
                )
                != "None"
                else None
            )
            metadata = (
                self.extract_json(
                    str(
                        span.attributes.get(
                            "traceloop.association.properties.literal.metadata"
                        )
                    )
                )
                if span.attributes.get(
                    "traceloop.association.properties.literal.metadata"
                )
                and span.attributes.get(
                    "traceloop.association.properties.literal.metadata"
                )
                != "None"
                else None
            )
            tags = (
                self.extract_json(
                    str(
                        span.attributes.get(
                            "traceloop.association.properties.literal.tags"
                        )
                    )
                )
                if span.attributes.get("traceloop.association.properties.literal.tags")
                and span.attributes.get("traceloop.association.properties.literal.tags")
                != "None"
                else None
            )
            name = (
                span.attributes.get("traceloop.association.properties.literal.name")
                if span.attributes.get("traceloop.association.properties.literal.name")
                and span.attributes.get("traceloop.association.properties.literal.name")
                != "None"
                else None
            )
            generation_type = span.attributes.get("llm.request.type")
            generation_messages = (
                self.extract_messages(span.attributes)
                if generation_type == "chat"
                else None
            )
            generation_message_completion = (
                self.extract_messages(span.attributes, "gen_ai.completion.")[0]
                if generation_type == "chat"
                else None
            )
            generation_prompt = span.attributes.get("gen_ai.prompt.0.user")
            generation_completion = span.attributes.get("gen_ai.completion.0.content")
            generation_model = span.attributes.get("gen_ai.request.model")
            generation_provider = span.attributes.get("gen_ai.system")
            generation_max_tokens = span.attributes.get("gen_ai.request.max_tokens")
            generation_stream = span.attributes.get("llm.is_streaming")
            generation_token_count = span.attributes.get("llm.usage.total_tokens")
            generation_input_token_count = span.attributes.get(
                "gen_ai.usage.prompt_tokens"
            )
            generation_output_token_count = span.attributes.get(
                "gen_ai.usage.completion_tokens"
            )
            # TODO: Validate settings
            generation_frequency_penalty = span.attributes.get(
                "gen_ai.request.frequency_penalty"
            )
            generation_logit_bias = span.attributes.get("gen_ai.request.logit_bias")
            generation_logprobs = span.attributes.get("gen_ai.request.logprobs")
            generation_top_logprobs = span.attributes.get("gen_ai.request.top_logprobs")
            generation_n = span.attributes.get("gen_ai.request.n")
            generation_presence_penalty = span.attributes.get(
                "gen_ai.request.presence_penalty"
            )
            generation_response_format = span.attributes.get(
                "gen_ai.request.response_format"
            )
            generation_seed = span.attributes.get("gen_ai.request.seed")
            generation_stop = span.attributes.get("gen_ai.request.stop")
            generation_temperature = span.attributes.get("gen_ai.request.temperature")
            generation_top_p = span.attributes.get("gen_ai.request.top_p")
            generation_tool_choice = span.attributes.get("gen_ai.request.tool_choice")

        step = Step.from_dict(
            {
                "id": (str(span.context.span_id) if span.context else None),
                "name": str(name) if name else span.name,
                "type": "llm",
                "metadata": cast(Dict, metadata),
                "startTime": start_time,
                "endTime": end_time,
                "threadId": str(thread_id) if thread_id else None,
                "parentId": str(parent_id) if parent_id else None,
                "rootRunId": str(root_run_id) if root_run_id else None,
                "input": (
                    {"content": generation_messages}
                    if generation_type == "chat"
                    else {"content": generation_prompt}
                ),
                "output": (
                    {"content": generation_message_completion}
                    if generation_type == "chat"
                    else {"content": generation_completion}
                ),
                "tags": cast(List, tags),
                "generation": {
                    "prompt": (
                        generation_prompt if generation_type == "completion" else None
                    ),
                    "completion": (
                        generation_completion
                        if generation_type == "completion"
                        else None
                    ),
                    "type": (
                        GenerationType.CHAT
                        if generation_type == "chat"
                        else GenerationType.COMPLETION
                    ),
                    "model": generation_model,
                    "provider": generation_provider,
                    "settings": {
                        "max_tokens": generation_max_tokens,
                        "frequency_penalty": generation_frequency_penalty,
                        "logit_bias": generation_logit_bias,
                        "logprobs": generation_logprobs,
                        "top_logprobs": generation_top_logprobs,
                        "n": generation_n,
                        "presence_penalty": generation_presence_penalty,
                        "response_format": generation_response_format,
                        "seed": generation_seed,
                        "stop": generation_stop,
                        "temperature": generation_temperature,
                        "top_p": generation_top_p,
                        "tool_choice": generation_tool_choice,
                        "stream": generation_stream,
                    },
                    "tokenCount": generation_token_count,
                    "inputTokenCount": generation_input_token_count,
                    "outputTokenCount": generation_output_token_count,
                    "messages": generation_messages,
                    "messageCompletion": generation_message_completion,
                },
            }
        )

        if not span.status.is_ok:
            step.error = span.status.description or "Unknown error"

        # TODO: Add generation promptid
        # TODO: Add generation variables
        # TODO: ttFirstToken
        # TODO: duration
        # TODO: tokenThroughputInSeconds
        # TODO: Add tools
        # TODO: error check with gemini error

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
