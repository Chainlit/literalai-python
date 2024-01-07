from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Callable, Optional, TypedDict

if TYPE_CHECKING:
    from literalai.step import Step


class BeforeContext(TypedDict):
    original_func: Callable
    step: Optional["Step"]


class AfterContext(TypedDict):
    original_func: Callable
    step: "Step"


def sync_wrapper(before_func=None, after_func=None):
    def decorator(original_func):
        @wraps(original_func)
        def wrapped(*args, **kwargs):
            context = {"original_func": original_func}
            # If a before_func is provided, call it with the shared context.
            if before_func:
                before_func(context, *args, **kwargs)

            result = original_func(*args, **kwargs)

            # If an after_func is provided, call it with the result and the shared context.
            if after_func:
                result = after_func(result, context, *args, **kwargs)

            return result

        return wrapped

    return decorator


def async_wrapper(before_func=None, after_func=None):
    def decorator(original_func):
        @wraps(original_func)
        async def wrapped(*args, **kwargs):
            context = {"original_func": original_func}
            # If a before_func is provided, call it with the shared context.
            if before_func:
                await before_func(context, *args, **kwargs)

            result = await original_func(*args, **kwargs)

            # If an after_func is provided, call it with the result and the shared context.
            if after_func:
                result = await after_func(result, context, *args, **kwargs)

            return result

        return wrapped

    return decorator


def wrap_all(
    to_wrap: list,
    before_wrapper,
    after_wrapper,
    async_before_wrapper,
    async_after_wrapper,
):
    for patch in to_wrap:
        module = import_module(str(patch["module"]))
        target_object = getattr(module, str(patch["object"]))
        original_method = getattr(target_object, str(patch["method"]))

        if patch["async"]:
            wrapped_method = async_wrapper(
                before_func=async_before_wrapper(metadata=patch["metadata"]),
                after_func=async_after_wrapper(metadata=patch["metadata"]),
            )(original_method)
        else:
            wrapped_method = sync_wrapper(
                before_func=before_wrapper(metadata=patch["metadata"]),
                after_func=after_wrapper(metadata=patch["metadata"]),
            )(original_method)

        setattr(target_object, str(patch["method"]), wrapped_method)
