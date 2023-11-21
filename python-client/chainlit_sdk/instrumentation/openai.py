import json
from importlib import import_module
from importlib.metadata import version

from chainlit_sdk.observability_agent import ObservabilityAgent
from chainlit_sdk.wrappers import async_wrapper, sync_wrapper
from packaging import version as packaging_version


def instrument_openai(observer: ObservabilityAgent):
    try:
        version("openai")
    except:
        # openai not installed, no need to patch
        return

    is_legacy = packaging_version.parse(version("openai")) < packaging_version.parse(
        "1.0.0"
    )

    def before_wrapper(generation_type: str = "COMPLETION"):
        def before(context, *args, **kwargs):
            step = observer.create_step(
                name=context["original_func"].__name__, type="llm"
            )
            if kwargs.get("messages"):
                step.input = json.dumps(kwargs.get("messages"))
                step.generation["messages"] = kwargs.get("messages")
            step.generation["provider"] = "openai"
            step.generation["settings"] = {
                "model": kwargs.get("model"),
            }
            step.generation["type"] = generation_type
            context["step"] = step

        return before

    def after_wrapper():
        def after(result, context, *args, **kwargs):
            step = context["step"]
            if is_legacy:
                step.output = result.choices[0].text
            else:
                step.output = result.choices[0].message.content
            step.generation["tokenCount"] = result.usage.total_tokens
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
                "type": "COMPLETION",
            },
            {
                "module": "openai",
                "object": "ChatCompletion",
                "method": "create",
                "type": "CHAT",
            },
        ]
        async_patches = [
            {
                "module": "openai",
                "object": "Completion",
                "method": "acreate",
                "type": "COMPLETION",
            },
            {
                "module": "openai",
                "object": "ChatCompletion",
                "method": "acreate",
                "type": "CHAT",
            },
        ]
    else:
        sync_patches = [
            {
                "module": "openai.resources.chat.completions",
                "object": "Completions",
                "method": "create",
                "type": "CHAT",
            },
            {
                "module": "openai.resources.completions",
                "object": "Completions",
                "method": "create",
                "type": "COMPLETION",
            },
        ]
        async_patches = [
            {
                "module": "openai.resources.chat.completions",
                "object": "AsyncCompletions",
                "method": "create",
                "type": "CHAT",
            },
            {
                "module": "openai.resources.completions",
                "object": "AsyncCompletions",
                "method": "create",
                "type": "COMPLETION",
            },
        ]

    for patch in sync_patches:
        module = import_module(patch["module"])
        target_object = getattr(module, patch["object"])
        original_method = getattr(target_object, patch["method"])

        wrapped_method = sync_wrapper(
            before_func=before_wrapper(generation_type=patch["type"]),
            after_func=after_wrapper(),
        )(original_method)

        setattr(target_object, patch["method"], wrapped_method)

    for patch in async_patches:
        module = import_module(patch["module"])
        target_object = getattr(module, patch["object"])
        original_method = getattr(target_object, patch["method"])

        wrapped_method = async_wrapper(
            before_func=before_wrapper(generation_type=patch["type"]),
            after_func=after_wrapper(),
        )(original_method)

        setattr(target_object, patch["method"], wrapped_method)
