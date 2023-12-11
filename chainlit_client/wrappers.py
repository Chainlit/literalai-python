from functools import wraps


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
                after_func(result, context, *args, **kwargs)

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
                before_func(context, *args, **kwargs)

            result = await original_func(*args, **kwargs)

            # If an after_func is provided, call it with the result and the shared context.
            if after_func:
                after_func(result, context, *args, **kwargs)

            return result

        return wrapped

    return decorator
