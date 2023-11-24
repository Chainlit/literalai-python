import time
import uuid
from enum import Enum, unique
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .context import active_steps_var, active_thread_id_var

if TYPE_CHECKING:
    from .event_processor import EventProcessor
    from .client import ChainlitClient


@unique
class StepType(Enum):
    RUN = "RUN"
    TOOL = "TOOL"
    LLM = "LLM"
    EMBEDDING = "EMBEDDING"
    RETRIEVAL = "RETRIEVAL"
    RERANK = "RERANK"
    UNDEFINED = "UNDEFINED"


@unique
class MessageRole(Enum):
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"
    USER = "USER"


@unique
class GenerationType(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"

# TODO: Split in two classes: ChatGeneration and CompletionGeneration
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
            "type": self.type.value if self.type else None,
        }


@unique
class FeedbackStrategy(Enum):
    BINARY = "BINARY"
    STARS = "STARS"
    BIG_STARS = "BIG_STARS"
    LIKERT = "LIKERT"
    CONTINUOUS = "CONTINUOUS"
    LETTERS = "LETTERS"
    PERCENTAGE = "PERCENTAGE"


class Feedback:
    value: Optional[float] = None
    strategy: FeedbackStrategy = FeedbackStrategy.BINARY
    comment: Optional[str] = None

    def __init__(
        self,
        value: Optional[float] = None,
        strategy: FeedbackStrategy = FeedbackStrategy.BINARY,
        comment: Optional[str] = None,
    ):
        self.value = value
        self.strategy = strategy
        self.comment = comment

    def to_dict(self):
        return {
            "value": self.value,
            "strategy": self.strategy.value if self.strategy else None,
            "comment": self.comment,
        }


class Attachment:
    id: str = None
    mime: Optional[str] = None
    name: Optional[str] = None
    objectKey: Optional[str] = None
    url: Optional[str] = None

    def __init__(
        self,
        id: str = None,
        mime: Optional[str] = None,
        name: Optional[str] = None,
        objectKey: Optional[str] = None,
        url: Optional[str] = None,
    ):
        self.id = id
        if self.id is None:
            self.id = str(uuid.uuid4())
        self.mime = mime
        self.name = name
        self.objectKey = objectKey
        self.url = url

    def to_dict(self):
        return {
            "id": self.id,
            "mime": self.mime,
            "name": self.name,
            "objectKey": self.objectKey,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, attachment_dict: Dict) -> "Attachment":
        id = attachment_dict.get("id")
        mime = attachment_dict.get("mime")
        name = attachment_dict.get("name")
        objectKey = attachment_dict.get("objectKey")
        url = attachment_dict.get("url")

        attachment = cls(id=id, mime=mime, name=name, objectKey=objectKey, url=url)

        return attachment


class Step:
    id: str = None
    name: str = ""
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
        type: StepType = None,
        thread_id: str = None,
        processor: "EventProcessor" = None,
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
        type: Optional[StepType] = None,
        thread_id: Optional[str] = None,
    ):
        self.client = client
        self.step_name = name
        self.step_type = type
        self.step: Optional[Step] = None
        self.thread_id = thread_id

    def __enter__(self) -> Step:
        self.step: Step = self.client.create_step(
            name=self.step_name, type=self.step_type, thread_id=self.thread_id
        )
        return self.step

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.step.finalize()


class Thread:
    id: str = None
    steps: List[Step] = []

    def __init__(
        self,
        id: str,
        steps: List[Step] = [],
    ):
        self.id = id
        self.steps = steps

    def to_dict(self):
        return {
            "id": self.id,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, thread_dict: Dict) -> "Thread":
        id = thread_dict.get("id")
        steps = [Step.from_dict(step) for step in thread_dict.get("steps")]

        thread = cls(id=id, steps=steps)

        return thread


class ThreadContextManager:
    def __init__(
        self,
        client: "ChainlitClient",
        thread_id: Optional[str] = None,
    ):
        self.client = client
        if thread_id is None:
            self.thread_id = str(uuid.uuid4())
        
        active_thread_id_var.set(thread_id)
        self.thread: Thread(id=self.thread_id)

    def __enter__(self) -> Step:
        return self.thread_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass