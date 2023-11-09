from chainlit_sdk.observability_agent import ObservabilityAgent
from chainlit_sdk.wrappers import sync_wrapper, async_wrapper
from importlib import import_module
from importlib.metadata import version
from packaging import version as packaging_version


agent = ObservabilityAgent()


def before_wrapper():
    def before(context, *args, **kwargs):
        span = agent.create_span(name=context["original_func"].__name__, type="llm")
        span.set_parameter("model", kwargs.get("model"))
        context["span"] = span

    return before


def after_wrapper():
    def after(result, context, *args, **kwargs):
        span = context["span"]
        span.set_parameter("response", result.choices[0].message.content)
        span.set_parameter("prompt_tokens", result.usage.prompt_tokens)
        span.set_parameter("completion_tokens", result.usage.completion_tokens)
        span.set_parameter("total_tokens", result.usage.total_tokens)
        span.finalize()

    return after


def instrument():
    try:
        version("openai")
    except:
        # openai not installed, no need to patch
        return

    sync_patches = []
    async_patches = []
    if packaging_version.parse(version("openai")) >= packaging_version.parse("1.0.0"):
        #     openai.ChatCompletion.create() -> client.chat.completions.create()
        # openai.Completion.create() -> client.completions.create()
        sync_patches = [
            {
                "module": "openai.resources.chat.completions",
                "object": "Completions",
                "method": "create",
            },
            {
                "module": "openai.resources.completions",
                "object": "Completions",
                "method": "create",
            },
        ]
        async_patches = [
            {
                "module": "openai.resources.chat.completions",
                "object": "AsyncCompletions",
                "method": "create",
            },
            {
                "module": "openai.resources.completions",
                "object": "AsyncCompletions",
                "method": "create",
            },
        ]
    else:
        sync_patches = [
            {
                "module": "openai",
                "object": "Completion",
                "method": "create",
            },
            {
                "module": "openai",
                "object": "ChatCompletion",
                "method": "create",
            },
        ]
        async_patches = [
            {
                "module": "openai",
                "object": "Completion",
                "method": "acreate",
            },
            {
                "module": "openai",
                "object": "ChatCompletion",
                "method": "acreate",
            },
        ]

    for patch in sync_patches:
        module = import_module(patch["module"])
        target_object = getattr(module, patch["object"])
        original_method = getattr(target_object, patch["method"])

        wrapped_method = sync_wrapper(
            before_func=before_wrapper(),
            after_func=after_wrapper(),
        )(original_method)

        setattr(target_object, patch["method"], wrapped_method)

    for patch in async_patches:
        module = import_module(patch["module"])
        target_object = getattr(module, patch["object"])
        original_method = getattr(target_object, patch["method"])

        wrapped_method = async_wrapper(
            before_func=before_wrapper(),
            after_func=after_wrapper(),
        )(original_method)

        setattr(target_object, patch["method"], wrapped_method)
