import datetime
import inspect
import json
import uuid
from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from .client import ChainlitClient
    from .event_processor import EventProcessor

from .context import active_steps_var, active_thread_id_var
from .types import Attachment, BaseGeneration, Feedback

StepType = Literal[
    "RUN", "TOOL", "LLM", "EMBEDDING", "RETRIEVAL", "RERANK", "UNDEFINED"
]


class Step:
    id: Optional[str] = None
    name: Optional[str] = ""
    type: Optional[StepType] = None
    metadata: Dict = {}
    parent_id: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    tags: Optional[List[str]] = None
    thread_id: Optional[str] = None

    generation: Optional[BaseGeneration] = None
    feedback: Optional[Feedback] = None
    attachments: List[Attachment] = []

    def __init__(
        self,
        name: str = "",
        type: Optional[StepType] = None,
        id: Optional[str] = None,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        processor: Optional["EventProcessor"] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.start = datetime.datetime.utcnow().isoformat()
        self.name = name
        self.type = type

        self.processor = processor

        # priority for thread_id: thread_id > parent_step.thread_id > active_thread
        self.thread_id = thread_id

        # priority for parent_id: parent_id > parent_step.id
        self.parent_id = parent_id

        active_steps = active_steps_var.get()
        if active_steps:
            parent_step = active_steps[-1]
            if not parent_id:
                self.parent_id = parent_step.id
            if not thread_id:
                self.thread_id = parent_step.thread_id

        if not self.thread_id:
            if active_thread := active_thread_id_var.get():
                self.thread_id = active_thread

        if not self.thread_id:
            raise Exception("Step must be initialized with a thread_id.")

        active_steps.append(self)
        active_steps_var.set(active_steps)

    def finalize(self):
        self.end = datetime.datetime.utcnow().isoformat()
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
            "type": self.type,
            "thread_id": self.thread_id,
            "input": self.input,
            "output": self.output,
            "generation": self.generation.to_dict()
            if self.generation and self.type == "LLM"
            else None,
            "name": self.name,
            "tags": self.tags,
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "attachments": [attachment.to_dict() for attachment in self.attachments],
        }

    @classmethod
    def from_dict(cls, step_dict: Dict) -> "Step":
        name = step_dict.get("name", "")
        step_type = step_dict.get("type", "UNDEFINED")  # type: StepType
        thread_id = step_dict.get("thread_id") or step_dict.get("threadId")

        step = cls(name=name, type=step_type, thread_id=thread_id)

        step.id = step_dict.get("id")
        step.input = step_dict.get("input", None)
        step.output = step_dict.get("output", None)
        step.metadata = step_dict.get("metadata", {})
        step.tags = step_dict.get("tags", [])
        step.parent_id = step_dict.get("parent_id", None)
        step.start = step_dict.get("start", None)
        step.end = step_dict.get("end", None)

        if "generation" in step_dict and step_type == "LLM":
            generation_dict = step_dict["generation"]
            if generation_dict:
                step.generation = BaseGeneration.from_dict(generation_dict)

        if "feedback" in step_dict:
            feedback_dict = step_dict["feedback"]
            if feedback_dict:
                step.feedback = Feedback.from_dict(feedback_dict)

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
        type: StepType = "UNDEFINED",
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        self.client = client
        self.step_name = name
        self.step_type = type
        self.id = id
        self.parent_id = parent_id
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
            name=self.step_name,
            type=self.step_type,
            id=self.id,
            parent_id=self.parent_id,
            thread_id=self.thread_id,
        )
        return self.step

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.step.finalize()

    def __enter__(self) -> Step:
        self.step = self.client.create_step(
            name=self.step_name,
            type=self.step_type,
            id=self.id,
            parent_id=self.parent_id,
            thread_id=self.thread_id,
        )
        return self.step

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.step.finalize()


def step_decorator(
    client: "ChainlitClient",
    func: Callable,
    type: StepType = "UNDEFINED",
    name: str = "",
    id: Optional[str] = None,
    parent_id: Optional[str] = None,
    thread_id: Optional[str] = None,
):
    if not name:
        name = func.__name__

    # Handle async decorator
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with StepContextManager(
                client=client,
                type=type,
                name=name,
                id=id,
                parent_id=parent_id,
                thread_id=thread_id,
            ) as step:
                try:
                    step.input = json.dumps({"args": args, "kwargs": kwargs})
                except:
                    pass
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
                client=client,
                type=type,
                name=name,
                id=id,
                parent_id=parent_id,
                thread_id=thread_id,
            ) as step:
                try:
                    step.input = json.dumps({"args": args, "kwargs": kwargs})
                except:
                    pass
                result = func(*args, **kwargs)
                try:
                    if step.output is None:
                        step.output = json.dumps(result)
                except:
                    pass
                return result

        return sync_wrapper
