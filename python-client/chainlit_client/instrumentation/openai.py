import json
from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING, Callable, Optional, TypedDict

if TYPE_CHECKING:
    from ..client import ChainlitClient

from packaging import version as packaging_version

from ..step import Step
from ..types import ChatGeneration, CompletionGeneration, GenerationType
from ..wrappers import async_wrapper, sync_wrapper


class BeforeContext(TypedDict):
    original_func: Callable
    step: Optional[Step]


class AfterContext(TypedDict):
    original_func: Callable
    step: Step


def instrument_openai(client: "ChainlitClient"):
    try:
        version("openai")
    except:
        # openai not installed, no need to patch
        return

    is_legacy = packaging_version.parse(version("openai")) < packaging_version.parse(
        "1.0.0"
    )

    def before_wrapper(generation_type: GenerationType = GenerationType.CHAT):
        def before(context: BeforeContext, *args, **kwargs):
            step = client.create_step(
                name=context["original_func"].__name__, type="LLM"
            )

            # TODO: Support AzureOpenAI

            # TODO: Capture all settings
            settings = {
                "model": kwargs.get("model"),
            }

            # TODO: Handle CompletionGeneration
            step.generation = ChatGeneration(provider="openai", settings=settings)
            if kwargs.get("messages"):
                step.input = json.dumps(kwargs.get("messages"))
                step.generation.messages = kwargs.get("messages")

            context["step"] = step

        return before

    def after_wrapper():
        def after(result, context: AfterContext, *args, **kwargs):
            step = context["step"]
            if is_legacy:
                step.output = result.choices[0].text
            else:
                step.output = result.choices[0].message.content

            if step.generation:
                step.generation.token_count = result.usage.total_tokens
            step.finalize()

        return after

    sync_patches = []
    async_patches = []
    if is_legacy:
        sync_patches = [
            {
                "module": "openai",
                "object": "Completion",
                "method": "create",
                "type": GenerationType.COMPLETION,
            },
            {
                "module": "openai",
                "object": "ChatCompletion",
                "method": "create",
                "type": GenerationType.CHAT,
            },
        ]
        async_patches = [
            {
                "module": "openai",
                "object": "Completion",
                "method": "acreate",
                "type": GenerationType.COMPLETION,
            },
            {
                "module": "openai",
                "object": "ChatCompletion",
                "method": "acreate",
                "type": GenerationType.CHAT,
            },
        ]
    else:
        sync_patches = [
            {
                "module": "openai.resources.chat.completions",
                "object": "Completions",
                "method": "create",
                "type": GenerationType.CHAT,
            },
            {
                "module": "openai.resources.completions",
                "object": "Completions",
                "method": "create",
                "type": GenerationType.COMPLETION,
            },
        ]
        async_patches = [
            {
                "module": "openai.resources.chat.completions",
                "object": "AsyncCompletions",
                "method": "create",
                "type": GenerationType.CHAT,
            },
            {
                "module": "openai.resources.completions",
                "object": "AsyncCompletions",
                "method": "create",
                "type": GenerationType.COMPLETION,
            },
        ]

    for patch in sync_patches:
        module = import_module(str(patch["module"]))
        target_object = getattr(module, str(patch["object"]))
        original_method = getattr(target_object, str(patch["method"]))

        generation_type = GenerationType(patch["type"])

        wrapped_method = sync_wrapper(
            before_func=before_wrapper(generation_type=generation_type),
            after_func=after_wrapper(),
        )(original_method)

        setattr(target_object, str(patch["method"]), wrapped_method)

    for patch in async_patches:
        module = import_module(str(patch["module"]))
        target_object = getattr(module, str(patch["object"]))
        original_method = getattr(target_object, str(patch["method"]))

        generation_type = GenerationType(patch["type"])

        wrapped_method = async_wrapper(
            before_func=before_wrapper(generation_type=generation_type),
            after_func=after_wrapper(),
        )(original_method)

        setattr(target_object, str(patch["method"]), wrapped_method)
