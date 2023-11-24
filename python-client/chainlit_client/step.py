import inspect
import json
import time
import uuid
from enum import Enum, unique
from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .client import ChainlitClient
    from .event_processor import EventProcessor

from .context import active_steps_var, active_thread_id_var
from .types import Attachment, Feedback, FeedbackStrategy, Generation, GenerationType


@unique
class StepType(Enum):
    RUN = "RUN"
    TOOL = "TOOL"
    LLM = "LLM"
    EMBEDDING = "EMBEDDING"
    RETRIEVAL = "RETRIEVAL"
    RERANK = "RERANK"
    UNDEFINED = "UNDEFINED"


class Step:
    id: Optional[str] = None
    name: Optional[str] = ""
    type: Optional[StepType] = None
    metadata: Dict = {}
    parent_id: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    input: Optional[str] = None
    output: Optional[str] = None

    generation: Optional[Generation] = None
    feedback: Optional[Feedback] = None
    attachments: List[Attachment] = []

    def __init__(
        self,
        name: str = "",
        type: Optional[StepType] = None,
        thread_id: Optional[str] = None,
        processor: Optional["EventProcessor"] = None,
    ):
        self.id = str(uuid.uuid4())
        self.start = int(time.time() * 1e3)
        self.name = name
        self.type = type
        if type == StepType.LLM:
            self.generation = Generation()
        self.processor = processor

        self.thread_id = thread_id

        # Overwrite the thread_id with the context as it's more trustworthy
        active_thread = active_thread_id_var.get()
        if active_thread:
            self.thread_id = active_thread

        active_steps = active_steps_var.get()
        if active_steps:
            parent_step = active_steps[-1]
            self.parent_id = parent_step.id
            # Overwrite the thread_id with the parent step as it's more trustworthy
            self.thread_id = parent_step.thread_id

        if self.thread_id is None:
            raise Exception("Step must be initialized with a thread_id.")

        active_steps.append(self)
        active_steps_var.set(active_steps)

    def finalize(self):
        self.end = int(time.time() * 1e3)
        active_steps = active_steps_var.get()
        active_steps.pop()
        active_steps_var.set(active_steps)
        if self.processor is None:
            raise Exception(
                "Step must be initialized with a processor to allow finalization."
            )
        self.processor.add_event(self.to_dict())

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
            "name": self.name,
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "attachments": [attachment.to_dict() for attachment in self.attachments],
        }

    @classmethod
    def from_dict(cls, step_dict: Dict) -> "Step":
        name = step_dict.get("name", "")
        step_type = StepType(step_dict.get("type")) if step_dict.get("type") else None
        thread_id = step_dict.get("thread_id")

        step = cls(name=name, type=step_type, thread_id=thread_id)

        step.id = step_dict.get("id")
        step.input = step_dict.get("input", None)
        step.output = step_dict.get("output", None)
        step.metadata = step_dict.get("metadata", {})

        if "generation" in step_dict and step_type == StepType.LLM:
            generation_dict = step_dict["generation"]
            generation = Generation()
            generation.template = generation_dict.get("template")
            generation.formatted = generation_dict.get("formatted")
            generation.template_format = generation_dict.get("template_format")
            generation.provider = generation_dict.get("provider")
            generation.inputs = generation_dict.get("inputs")
            generation.completion = generation_dict.get("completion")
            generation.settings = generation_dict.get("settings")
            generation.messages = generation_dict.get("messages")
            generation.tokenCount = generation_dict.get("tokenCount")
            generation.type = GenerationType(generation_dict.get("type"))
            step.generation = generation

        if "feedback" in step_dict:
            feedback_dict = step_dict["feedback"]
            if feedback_dict:
                value = feedback_dict.get("value")
                strategy = (
                    FeedbackStrategy(feedback_dict.get("strategy"))
                    if feedback_dict.get("strategy")
                    else None
                )
                comment = feedback_dict.get("comment")
                if strategy:
                    step.feedback = Feedback(
                        value=value, strategy=strategy, comment=comment
                    )

        if "attachments" in step_dict:
            attachments = step_dict["attachments"]
            if attachments:
                step.attachments = [
                    Attachment.from_dict(attachment) for attachment in attachments
                ]

        return step


class StepContextManager:
    def __init__(
        self,
        client: "ChainlitClient",
        name: str = "",
        type: StepType = StepType.UNDEFINED,
        thread_id: Optional[str] = None,
    ):
        self.client = client
        self.step_name = name
        self.step_type = type
        self.thread_id = thread_id

    def __call__(self, func):
        return step_decorator(
            self.client,
            func=func,
            type=self.step_type,
            name=self.step_name,
            thread_id=self.thread_id,
        )

    async def __aenter__(self):
        self.step = self.client.create_step(
            name=self.step_name, type=self.step_type, thread_id=self.thread_id
        )
        return self.step

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.step.finalize()

    def __enter__(self) -> Step:
        self.step = self.client.create_step(
            name=self.step_name, type=self.step_type, thread_id=self.thread_id
        )
        return self.step

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.step.finalize()


def step_decorator(
    client: "ChainlitClient",
    func: Callable,
    type: StepType = StepType.UNDEFINED,
    name: str = "",
    thread_id: Optional[str] = None,
):
    if not name:
        name = func.__name__

    # Handle async decorator
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with StepContextManager(
                client=client, type=type, name=name, thread_id=thread_id
            ) as step:
                result = await func(*args, **kwargs)
                try:
                    if step.output is None:
                        step.output = json.dumps(result)
                except:
                    pass
                return result

        return async_wrapper
    else:
        # Handle sync decorator
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with StepContextManager(
                client=client, type=type, name=name, thread_id=thread_id
            ) as step:
                result = func(*args, **kwargs)
                try:
                    if step.output is None:
                        step.output = json.dumps(result)
                except:
                    pass
                return result

        return sync_wrapper
