from functools import wraps

from .types import Step, StepType

from .context import active_steps_var
from .event_processor import EventProcessor


class StepContextManager:
    def __init__(self, agent, name="", type=None, thread_id=None):
        self.agent = agent
        self.step_name = name
        self.step_type = type
        self.step: Step = None
        self.thread_id = thread_id

    def __enter__(self):
        self.step = self.agent.create_step(
            name=self.step_name, type=self.step_type, thread_id=self.thread_id
        )
        return self.step

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.step.finalize()


class ObservabilityAgent:
    processor: EventProcessor = None

    def __init__(self, processor: EventProcessor = None):
        self.processor = processor

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
        if self.processor is None:
            raise Exception("ObservabilityAgent not initialized.")
        step = Step(name=name, type=type, thread_id=thread_id, processor=self.processor)
        return step

    def step(self, name="", type=None, thread_id=None):
        return StepContextManager(self, name=name, type=type, thread_id=thread_id)

    def get_current_step(self):
        active_steps = active_steps_var.get()
        if active_steps and len(active_steps) > 0:
            return active_steps[-1]
        else:
            return None
