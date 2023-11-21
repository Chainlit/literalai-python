import time
import uuid
from enum import Enum, unique
from typing import Any, Dict, Optional

from .context import active_steps_var
from .event_processor import EventProcessor


@unique
class StepType(Enum):
    RUN = "RUN"
    MESSAGE = "MESSAGE"
    LLM = "LLM"


@unique
class OperatorRole(Enum):
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"
    USER = "USER"


@unique
class GenerationType(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


class Generation:
    template: Optional[str] = None
    formatted: Optional[str] = None
    template_format: Optional[str] = None
    provider: Optional[str] = None
    inputs: Optional[Dict] = None
    completion: Optional[str] = None
    settings: Optional[Dict] = None
    messages: Optional[Any] = None
    tokenCount: Optional[int] = None
    type: Optional[GenerationType] = None

    def to_dict(self):
        return {
            "template": self.template,
            "formatted": self.formatted,
            "template_format": self.template_format,
            "provider": self.provider,
            "inputs": self.inputs,
            "completion": self.completion,
            "settings": self.settings,
            "messages": self.messages,
            "tokenCount": self.tokenCount,
            "type": self.type,
        }


class Step:
    id: str = str(uuid.uuid4())
    name: str = ""
    operatorRole: Optional[OperatorRole] = None
    type: Optional[StepType] = None
    metadata: Dict = {}
    parent_id: Optional[str] = None
    start: float = time.time()
    end: Optional[float] = None
    input: Optional[str] = None
    output: Optional[str] = None
    generation: Optional[Generation] = None

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
        if type == StepType.LLM:
            self.generation = Generation()
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
            "parent_id": self.parent_id,
            "start": self.start,
            "end": self.end,
            "type": self.type.value if self.type else None,
            "thread_id": str(self.thread_id),
            "input": self.input,
            "output": self.output,
            "generation": self.generation.to_dict()
            if self.type == StepType.LLM
            else None,
            "operatorRole": self.operatorRole.value if self.operatorRole else None,
        }
