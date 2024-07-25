import inspect
from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional

from literalai.my_types import Environment

if TYPE_CHECKING:
    from literalai.client import BaseLiteralClient


class EnvContextManager:
    def __init__(self, client: "BaseLiteralClient", env: Environment = "prod"):
        self.client = client
        self.env = env
        self.original_env = client.api.environment

    def __call__(self, func):
        return env_decorator(
            self.client,
            func=func,
            ctx_manager=self,
        )

    async def __aenter__(self):
        self.client.api.environment = self.env

    async def __aexit__(self):
        self.client.api.environment = self.original_env

    def __enter__(self):
        self.client.api.environment = self.env

    def __exit__(self):
        self.client.api.environment = self.original_env


def env_decorator(
    client: "BaseLiteralClient",
    func: Callable,
    env: Environment = "prod",
    ctx_manager: Optional[EnvContextManager] = None,
    **decorator_kwargs,
):
    if not ctx_manager:
        ctx_manager = EnvContextManager(
            client=client,
            env=env,
            **decorator_kwargs,
        )

    # Handle async decorator
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with ctx_manager:
                await func(*args, **kwargs)

        return async_wrapper
    else:
        # Handle sync decorator
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with ctx_manager:
                func(*args, **kwargs)

        return sync_wrapper
