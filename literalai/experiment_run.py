import inspect
import uuid
from functools import wraps
from typing import TYPE_CHECKING, Callable, Optional

from literalai.context import active_experiment_run_id_var
from literalai.environment import EnvContextManager
from literalai.step import StepContextManager

if TYPE_CHECKING:
    from literalai.client import BaseLiteralClient


class ExperimentRunContextManager(EnvContextManager, StepContextManager):
    def __init__(
        self,
        client: "BaseLiteralClient",
    ):
        self.id = str(uuid.uuid4())
        EnvContextManager.__init__(self, client=client, env="experiment")
        StepContextManager.__init__(
            self, client=client, name="Experiment Run", type="run", id=self.id
        )

    def __call__(self, func):
        return experiment_run_decorator(
            self.client,
            func=func,
            ctx_manager=self,
        )

    async def __aenter__(self):
        super().__aenter__()
        active_experiment_run_id_var.set(self.id)

    async def __aexit__(self):
        super().__aexit__()
        active_experiment_run_id_var.set(None)

    def __enter__(self):
        super().__enter__()
        active_experiment_run_id_var.set(self.id)

    def __exit__(self):
        super().__exit__()
        active_experiment_run_id_var.set(None)


def experiment_run_decorator(
    client: "BaseLiteralClient",
    func: Callable,
    ctx_manager: Optional[ExperimentRunContextManager] = None,
    **decorator_kwargs,
):
    if not ctx_manager:
        ctx_manager = ExperimentRunContextManager(
            client=client,
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
