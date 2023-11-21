import time
import uuid
from contextvars import ContextVar
from enum import Enum, unique
from functools import wraps
from typing import Dict, Optional

from .event_processor import EventProcessor

active_steps_var = ContextVar("active_steps", default=[])


@unique
class StepType(Enum):
    RUN = "run"
    MESSAGE = "message"
    LLM = "llm"


@unique
class OperatorRole(Enum):
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"
    USER = "USER"


class Step:
    id: str = str(uuid.uuid4())
    name: str = ""
    operatorRole: Optional[OperatorRole] = None
    type: StepType = None
    metadata: Dict = {}
    parent_id: str = None
    start: float = time.time()
    end: float = None
    input: str = None
    output: str = None
    generation: Dict = {}

    def __init__(
        self,
        name="",
        type: StepType = None,
        thread_id: str = None,
        processor: EventProcessor = None,
    ):
        if processor is None:
            raise Exception("Step must be initialized with a processor.")

        self.name = name
        self.type = type
        self.processor = processor

        active_steps = active_steps_var.get()
        if active_steps:
            parent_step = active_steps[-1]
            self.parent_id = parent_step.id
            self.thread_id = parent_step.thread_id
        else:
            self.thread_id = thread_id
        if self.thread_id is None:
            raise Exception("Step must be initialized with a thread_id.")
        active_steps.append(self)
        active_steps_var.set(active_steps)

    def finalize(self):
        self.end = time.time()
        active_steps = active_steps_var.get()
        active_steps.pop()
        self.processor.add_event(self.to_dict())
        active_steps_var.set(active_steps)

    def to_dict(self):
        return {
            "id": self.id,
            "metadata": self.metadata,
            "parent": self.parent,
            "start": self.start,
            "end": self.end,
            "type": self.type,
            "thread_id": str(self.thread_id),
            "input": self.input,
            "output": self.output,
            "generation": self.generation if self.type == "llm" else None,
            "operatorRole": self.operatorRole.value if self.operatorRole else None,
        }


class StepContextManager:
    def __init__(self, agent, name="", type=None, thread_id=None):
        self.agent = agent
        self.step_name = name
        self.step_type = type
        self.step = None
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
