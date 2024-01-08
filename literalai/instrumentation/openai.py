import json
import logging
from typing import TYPE_CHECKING, Dict

from literalai.requirements import check_all_requirements

if TYPE_CHECKING:
    from literalai.client import LiteralClient
    from literalai.step import Step

from literalai.my_types import (
    ChatGeneration,
    CompletionGeneration,
    GenerationMessage,
    GenerationType,
)
from literalai.wrappers import AfterContext, BeforeContext, wrap_all

logger = logging.getLogger(__name__)

REQUIREMENTS = ["openai>=1.0.0"]

TO_WRAP = [
    {
        "module": "openai.resources.chat.completions",
        "object": "Completions",
        "method": "create",
        "metadata": {
            "type": GenerationType.CHAT,
        },
        "async": False,
    },
    {
        "module": "openai.resources.completions",
        "object": "Completions",
        "method": "create",
        "metadata": {
            "type": GenerationType.COMPLETION,
        },
        "async": False,
    },
    {
        "module": "openai.resources.chat.completions",
        "object": "AsyncCompletions",
        "method": "create",
        "metadata": {
            "type": GenerationType.CHAT,
        },
        "async": True,
    },
    {
        "module": "openai.resources.completions",
        "object": "AsyncCompletions",
        "method": "create",
        "metadata": {
            "type": GenerationType.COMPLETION,
        },
        "async": True,
    },
]


def instrument_openai(client: "LiteralClient"):
    if not check_all_requirements(REQUIREMENTS):
        raise Exception(f"Instrumentation requirements not satisfied: {REQUIREMENTS}")

    from openai import AsyncStream, Stream

    def update_step_before(step: "Step", generation_type: "GenerationType", kwargs):
        if generation_type == GenerationType.CHAT:
            settings = {
                "model": kwargs.get("model"),
                "frequency_penalty": kwargs.get("frequency_penalty"),
                "logit_bias": kwargs.get("logit_bias"),
                "logprobs": kwargs.get("logprobs"),
                "top_logprobs": kwargs.get("top_logprobs"),
                "max_tokens": kwargs.get("max_tokens"),
                "n": kwargs.get("n"),
                "presence_penalty": kwargs.get("presence_penalty"),
                "response_format": kwargs.get("response_format"),
                "seed": kwargs.get("seed"),
                "stop": kwargs.get("stop"),
                "stream": kwargs.get("stream"),
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
                "tools": kwargs.get("tools"),
                "tool_choice": kwargs.get("tool_choice"),
            }
            settings = {k: v for k, v in settings.items() if v is not None}
            step.generation = ChatGeneration(provider="openai", settings=settings)
            if kwargs.get("messages"):
                step.input = json.dumps(kwargs.get("messages"))
                step.generation.messages = [
                    GenerationMessage(
                        role=m.get("role", "user"),
                        formatted=m.get("content", ""),
                    )
                    for m in kwargs.get("messages", [])
                ]
        elif generation_type == GenerationType.COMPLETION:
            settings = {
                "model": kwargs.get("model"),
                "best_of": kwargs.get("best_of"),
                "echo": kwargs.get("echo"),
                "frequency_penalty": kwargs.get("frequency_penalty"),
                "logit_bias": kwargs.get("logit_bias"),
                "logprobs": kwargs.get("logprobs"),
                "max_tokens": kwargs.get("max_tokens"),
                "n": kwargs.get("n"),
                "presence_penalty": kwargs.get("presence_penalty"),
                "seed": kwargs.get("seed"),
                "stop": kwargs.get("stop"),
                "stream": kwargs.get("stream"),
                "suffix": kwargs.get("suffix"),
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
            }
            settings = {k: v for k, v in settings.items() if v is not None}
            step.input = kwargs.get("prompt")
            step.generation = CompletionGeneration(
                provider="openai", settings=settings, formatted=kwargs.get("prompt")
            )

    def update_step_after(step: "Step", generation_type: "GenerationType", result):
        if generation_type == GenerationType.CHAT:
            step.output = result.choices[0].message.content
            if step.generation and step.generation.type == GenerationType.CHAT:
                step.generation.completion = result.choices[0].message.content
        elif generation_type == GenerationType.COMPLETION:
            step.output = result.choices[0].text
            if step.generation and step.generation.type == GenerationType.COMPLETION:
                step.generation.completion = result.choices[0].text

        if step.generation:
            step.generation.token_count = result.usage.total_tokens

    def before_wrapper(metadata: Dict):
        def before(context: BeforeContext, *args, **kwargs):
            step = client.start_step(name=context["original_func"].__name__, type="llm")

            generation_type = metadata["type"]

            update_step_before(step, generation_type, kwargs)

            context["step"] = step

        return before

    def async_before_wrapper(metadata: Dict):
        async def before(context: BeforeContext, *args, **kwargs):
            step = client.start_step(name=context["original_func"].__name__, type="llm")

            generation_type = metadata["type"]

            update_step_before(step, generation_type, kwargs)

            context["step"] = step

        return before

    def streaming_response(step: "Step", generation_type: "GenerationType", result):
        content = ""
        for chunk in result:
            if generation_type == GenerationType.CHAT:
                if (
                    len(chunk.choices) > 0
                    and chunk.choices[0].delta.content is not None
                ):
                    content += chunk.choices[0].delta.content
                yield chunk
            elif generation_type == GenerationType.COMPLETION:
                if len(chunk.choices) > 0 and chunk.choices[0].text is not None:
                    content += chunk.choices[0].text
                yield chunk

        step.output = content
        if step.generation:
            step.generation.completion = content
        step.end()

    def after_wrapper(metadata: Dict):
        # Needs to be done in a separate function to avoid transforming all returned data into generators
        def after(result, context: AfterContext, *args, **kwargs):
            generation_type = metadata["type"]

            step = context["step"]

            if isinstance(result, Stream):
                return streaming_response(step, generation_type, result)

            update_step_after(step, generation_type, result)
            step.end()

            return result

        return after

    async def async_streaming_response(
        step: "Step", generation_type: "GenerationType", result
    ):
        # Needs to be done in a separate function to avoid transforming all returned data into generators
        content = ""
        async for chunk in result:
            if generation_type == GenerationType.CHAT:
                if (
                    len(chunk.choices) > 0
                    and chunk.choices[0].delta.content is not None
                ):
                    content += chunk.choices[0].delta.content
            elif generation_type == GenerationType.COMPLETION:
                if len(chunk.choices) > 0 and chunk.choices[0].text is not None:
                    content += chunk.choices[0].text

            yield chunk

        step.output = content
        if step.generation:
            step.generation.completion = content
        step.end()

    def async_after_wrapper(metadata: Dict):
        async def after(result, context: AfterContext, *args, **kwargs):
            generation_type = metadata["type"]

            step = context["step"]

            if isinstance(result, AsyncStream):
                return async_streaming_response(step, generation_type, result)

            update_step_after(step, generation_type, result)
            step.end()

            return result

        return after

    wrap_all(
        TO_WRAP,
        before_wrapper,
        after_wrapper,
        async_before_wrapper,
        async_after_wrapper,
    )
