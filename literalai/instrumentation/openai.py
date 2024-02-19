import logging
import time
from typing import TYPE_CHECKING, Dict

from literalai.requirements import check_all_requirements

if TYPE_CHECKING:
    from literalai.client import LiteralClient
    from literalai.step import Step

from literalai.helper import ensure_values_serializable
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

is_instrumented = False


def instrument_openai(client: "LiteralClient"):
    global is_instrumented
    if is_instrumented:
        return

    if not check_all_requirements(REQUIREMENTS):
        raise Exception(f"Instrumentation requirements not satisfied: {REQUIREMENTS}")

    from openai import AsyncStream, Stream
    from openai.types.chat.chat_completion_chunk import ChoiceDelta

    def update_step_before(step: "Step", generation_type: "GenerationType", kwargs):
        model = kwargs.get("model")
        tools = kwargs.get("tools")
        step.name = model or "openai"
        if generation_type == GenerationType.CHAT:
            messages = ensure_values_serializable(kwargs.get("messages"))
            settings = {
                "model": model,
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
                "tool_choice": kwargs.get("tool_choice"),
            }
            settings = {k: v for k, v in settings.items() if v is not None}
            step.generation = ChatGeneration(
                provider="openai",
                model=model,
                tools=tools,
                settings=settings,
                messages=messages,
            )
            if messages:
                step.input = {"content": messages}

        elif generation_type == GenerationType.COMPLETION:
            settings = {
                "model": model,
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
            step.input = {"content": kwargs.get("prompt")}
            step.generation = CompletionGeneration(
                provider="openai",
                model=model,
                settings=settings,
                prompt=kwargs.get("prompt"),
            )

    def update_step_after(step: "Step", result):
        if step.generation and isinstance(step.generation, ChatGeneration):
            step.output = result.choices[0].message.model_dump()
            step.generation.message_completion = result.choices[0].message.model_dump()

        elif step.generation and isinstance(step.generation, CompletionGeneration):
            step.output = {"content": result.choices[0].text}
            if step.generation and step.generation.type == GenerationType.COMPLETION:
                step.generation.completion = result.choices[0].text

        if step.generation:
            step.generation.input_token_count = result.usage.prompt_tokens
            step.generation.output_token_count = result.usage.completion_tokens
            step.generation.token_count = result.usage.total_tokens

    def before_wrapper(metadata: Dict):
        def before(context: BeforeContext, *args, **kwargs):
            step = client.start_step(name=context["original_func"].__name__, type="llm")

            generation_type = metadata["type"]

            update_step_before(step, generation_type, kwargs)

            context["step"] = step
            context["start"] = time.time()

        return before

    def async_before_wrapper(metadata: Dict):
        async def before(context: BeforeContext, *args, **kwargs):
            step = client.start_step(name=context["original_func"].__name__, type="llm")

            generation_type = metadata["type"]

            update_step_before(step, generation_type, kwargs)

            context["step"] = step
            context["start"] = time.time()

        return before

    def process_delta(new_delta: ChoiceDelta, message_completion: GenerationMessage):
        if new_delta.function_call:
            if new_delta.function_call.name:
                message_completion["function_call"] = {
                    "name": new_delta.function_call.name
                }

            if not message_completion["function_call"]:
                return False

            if new_delta.function_call.arguments:
                if "arguments" not in message_completion["function_call"]:
                    message_completion["function_call"]["arguments"] = ""
                message_completion["function_call"][
                    "arguments"
                ] += new_delta.function_call.arguments
            return True
        elif new_delta.tool_calls:
            if not message_completion["tool_calls"]:
                message_completion["tool_calls"] = []
            delta_tool_call = new_delta.tool_calls[0]
            delta_function = delta_tool_call.function
            if not delta_function:
                return False
            if delta_function.name:
                message_completion["tool_calls"].append(  # type: ignore
                    {
                        "id": delta_tool_call.id,
                        "type": "function",
                        "function": {
                            "name": delta_function.name,
                            "arguments": "",
                        },
                    }
                )
            if delta_function.arguments:
                message_completion["tool_calls"][delta_tool_call.index]["function"][  # type: ignore
                    "arguments"
                ] += delta_function.arguments

            return True
        elif new_delta.content:
            if isinstance(message_completion["content"], str):
                message_completion["content"] += new_delta.content
                return True
        else:
            return False

    def streaming_response(step: "Step", result, context: AfterContext):
        completion = ""
        message_completion = {
            "role": "assistant",
            "content": "",
        }  # type: GenerationMessage
        token_count = 0
        for chunk in result:
            if step.generation and isinstance(step.generation, ChatGeneration):
                if len(chunk.choices) > 0:
                    ok = process_delta(chunk.choices[0].delta, message_completion)
                    if not ok:
                        yield chunk
                        continue
                    if step.generation.tt_first_token is None:
                        step.generation.tt_first_token = (
                            time.time() - context["start"]
                        ) * 1000
                    token_count += 1
                yield chunk
            elif step.generation and isinstance(step.generation, CompletionGeneration):
                if len(chunk.choices) > 0 and chunk.choices[0].text is not None:
                    if step.generation.tt_first_token is None:
                        step.generation.tt_first_token = (
                            time.time() - context["start"]
                        ) * 1000
                    token_count += 1
                    completion += chunk.choices[0].text
                yield chunk

        if step.generation:
            step.generation.duration = time.time() - context["start"]
            if step.generation.duration and token_count:
                step.generation.token_throughput_in_s = (
                    token_count / step.generation.duration
                )
            if isinstance(step.generation, ChatGeneration):
                step.output = message_completion  # type: ignore
                step.generation.message_completion = message_completion
            else:
                step.output = {"content": completion}
                step.generation.completion = completion

        step.end()

    def after_wrapper(metadata: Dict):
        # Needs to be done in a separate function to avoid transforming all returned data into generators
        def after(result, context: AfterContext, *args, **kwargs):
            step = context.get("step")
            if not step:
                return result

            if isinstance(result, Stream):
                return streaming_response(step, result, context)

            if step.generation:
                step.generation.duration = time.time() - context["start"]
            update_step_after(step, result)
            step.end()
            return result

        return after

    async def async_streaming_response(step: "Step", result, context: AfterContext):
        completion = ""
        message_completion = {
            "role": "assistant",
            "content": "",
        }  # type: GenerationMessage
        token_count = 0
        async for chunk in result:
            if step.generation and isinstance(step.generation, ChatGeneration):
                if len(chunk.choices) > 0:
                    ok = process_delta(chunk.choices[0].delta, message_completion)
                    if not ok:
                        continue
                    if step.generation.tt_first_token is None:
                        step.generation.tt_first_token = (
                            time.time() - context["start"]
                        ) * 1000
                    token_count += 1
                yield chunk
            elif step.generation and isinstance(step.generation, CompletionGeneration):
                if len(chunk.choices) > 0 and chunk.choices[0].text is not None:
                    if step.generation.tt_first_token is None:
                        step.generation.tt_first_token = (
                            time.time() - context["start"]
                        ) * 1000
                    token_count += 1
                    completion += chunk.choices[0].text
                yield chunk

        if step.generation:
            step.generation.duration = time.time() - context["start"]
            if step.generation.duration and token_count:
                step.generation.token_throughput_in_s = (
                    token_count / step.generation.duration
                )
            if isinstance(step.generation, ChatGeneration):
                step.output = message_completion  # type: ignore
                step.generation.message_completion = message_completion
            else:
                step.output = {"content": completion}
                step.generation.completion = completion
        step.end()

    def async_after_wrapper(metadata: Dict):
        async def after(result, context: AfterContext, *args, **kwargs):
            step = context.get("step")

            if not step:
                return result

            if isinstance(result, AsyncStream):
                return async_streaming_response(step, result, context)

            if step.generation:
                step.generation.duration = time.time() - context["start"]

            update_step_after(step, result)
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

    is_instrumented = True
