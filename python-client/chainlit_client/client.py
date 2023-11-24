import json
import os
import uuid
import inspect
from functools import wraps
from typing import Optional

from .api import API
from .context import active_steps_var, active_thread_id_var
from .event_processor import EventProcessor
from .instrumentation.openai import instrument_openai
from .types import ThreadContextManager, Step, StepContextManager, StepType


class ChainlitClient:
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

    def step_decorator(
        self,
        name: Optional[str] = None,
        type: Optional[StepType] = None,
        thread_id: Optional[str] = None,
    ):
        def decorator(func):
            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with StepContextManager(self, name=name or func.__name__, type=type, thread_id=thread_id) as step:
                        result = await func(*args, **kwargs)
                        try:
                            if step.output is None:
                                step.output = json.dumps(result)
                        except:
                            pass
                        return result
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with StepContextManager(self, name=name or func.__name__, type=type, thread_id=thread_id) as step:
                        result = func(*args, **kwargs)
                        try:
                            if step.output is None:
                                step.output = json.dumps(result)
                        except:
                            pass
                        return result
                return sync_wrapper
        return decorator


    def thread_decorator(self, original_function=None, *, thread_id: Optional[str] = None):
        def decorator(func):
            if inspect.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with ThreadContextManager(self, thread_id=thread_id) as step:
                        result = await func(*args, **kwargs)
                        return result
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with ThreadContextManager(self, thread_id=thread_id) as step:
                        return func(*args, **kwargs)
                return sync_wrapper

        # If the decorator is used without parenthesis, return the decorated function
        if original_function:
            return decorator(original_function)

        # If the decorator is used with parenthesis, return the decorator
        return decorator
 
    def run(
        self,
        thread_id: Optional[str] = None,
    ):
        return self.step_decorator(type=StepType.RUN, thread_id=thread_id)

    def llm(
        self,
        thread_id: Optional[str] = None,
    ):
        return self.step_decorator(type=StepType.LLM, thread_id=thread_id)

    def create_step(
        self,
        name: str = "",
        type: Optional[StepType] = None,
        thread_id: Optional[str] = None,
    ):
        step = Step(
            name=name, type=type, thread_id=thread_id, processor=self.event_processor
        )
        return step

    def get_current_step(self):
        active_steps = active_steps_var.get()
        if active_steps and len(active_steps) > 0:
            return active_steps[-1]
        else:
            return None

    def get_current_thread_id(self):
        return active_thread_id_var.get()

    def wait_until_queue_empty(self):
        self.event_processor.wait_until_queue_empty()
