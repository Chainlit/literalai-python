import os
from functools import wraps

from .api import API
from .context import active_steps_var
from .event_processor import EventProcessor
from .instrumentation.openai import instrument_openai
from .types import Step, StepType, StepContextManager


class Chainlit:
    def __init__(self, batch_size: int = 1, api_key: str = None, endpoint: str = None):
        if not api_key:
            self.api_key = os.getenv("CHAINLIT_API_KEY", None)
            if not self.api_key:
                raise Exception("CHAINLIT_API_KEY not provided")
        if not endpoint:
            self.endpoint = os.getenv(
                "CHAINLIT_ENDPOINT", "https://cloud.chainlit.io/graphql"
            )

        self.api = API(api_key=self.api_key, endpoint=self.endpoint)
        self.event_processor = EventProcessor(
            api=self.api,
            batch_size=batch_size,
        )

    def instrument_openai(self):
        instrument_openai(self)

    def step_decorator(self, type=None, thread_id=None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.step(name=func.__name__, type=type, thread_id=thread_id):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    async def a_step_decorator(self, type=None, thread_id=None):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.step(name=func.__name__, type=type, thread_id=thread_id):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def run(self, thread_id=None):
        return self.step_decorator(type=StepType.RUN, thread_id=thread_id)

    def message(self, thread_id=None):
        return self.step_decorator(type=StepType.MESSAGE, thread_id=thread_id)

    def llm(self, thread_id=None):
        return self.step_decorator(type=StepType.LLM, thread_id=thread_id)

    def a_run(self, thread_id=None):
        return self.a_step_decorator(type=StepType.RUN, thread_id=thread_id)

    def a_message(self, thread_id=None):
        return self.a_step_decorator(type=StepType.MESSAGE, thread_id=thread_id)

    def a_llm(self, thread_id=None):
        return self.a_step_decorator(type=StepType.LLM, thread_id=thread_id)

    def create_step(self, name="", type=None, thread_id=None):
        step = Step(
            name=name, type=type, thread_id=thread_id, processor=self.event_processor
        )
        return step

    def step(self, name="", type=None, thread_id=None):
        return StepContextManager(self, name=name, type=type, thread_id=thread_id)

    def get_current_step(self):
        active_steps = active_steps_var.get()
        if active_steps and len(active_steps) > 0:
            return active_steps[-1]
        else:
            return None

    def wait_until_queue_empty(self):
        self.event_processor.wait_until_queue_empty()
