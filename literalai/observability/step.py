import inspect
import uuid
from copy import deepcopy
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
)

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from literalai.client import BaseLiteralClient
    from literalai.event_processor import EventProcessor

from literalai.context import active_steps_var, active_thread_var
from literalai.helper import utc_now
from literalai.my_types import Environment, Utils
from literalai.observability.generation import (
    BaseGeneration,
    ChatGeneration,
    CompletionGeneration,
)

TrueStepType = Literal[
    "run", "tool", "llm", "embedding", "retrieval", "rerank", "undefined"
]

MessageStepType = Literal["user_message", "assistant_message", "system_message"]

StepType = Union[TrueStepType, MessageStepType]


ScoreType = Literal["HUMAN", "AI"]


class ScoreDict(TypedDict, total=False):
    id: Optional[str]
    name: str
    type: ScoreType
    value: float
    stepId: Optional[str]
    datasetExperimentItemId: Optional[str]
    comment: Optional[str]
    tags: Optional[List[str]]


class AttachmentDict(TypedDict, total=False):
    id: Optional[str]
    stepId: Optional[str]
    threadId: Optional[str]
    metadata: Optional[Dict]
    mime: Optional[str]
    name: Optional[str]
    objectKey: Optional[str]
    url: Optional[str]


@dataclass(repr=False)
class Score(Utils):
    name: str
    type: ScoreType
    value: float
    step_id: Optional[str]
    dataset_experiment_item_id: Optional[str]
    comment: Optional[str]
    tags: Optional[List[str]]
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "value": self.value,
            "stepId": self.step_id,
            "datasetExperimentItemId": self.dataset_experiment_item_id,
            "comment": self.comment,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, score_dict: ScoreDict) -> "Score":
        id = score_dict.get("id", "")
        name = score_dict.get("name", "")
        type = score_dict.get("type", "HUMAN")
        value = score_dict.get("value", 0.0)
        step_id = score_dict.get("stepId", "")
        dataset_experiment_item_id = score_dict.get("datasetExperimentItemId", "")
        comment = score_dict.get("comment", "")
        tags = score_dict.get("tags", [])

        score = cls(
            id=id,
            name=name,
            type=type,
            value=value,
            step_id=step_id,
            dataset_experiment_item_id=dataset_experiment_item_id,
            comment=comment,
            tags=tags,
        )

        return score


@dataclass(repr=False)
class Attachment(Utils):
    step_id: Optional[str] = None
    thread_id: Optional[str] = None
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[Dict] = Field(default_factory=lambda: {})
    mime: Optional[str] = None
    name: Optional[str] = None
    object_key: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "metadata": self.metadata,
            "mime": self.mime,
            "name": self.name,
            "objectKey": self.object_key,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, attachment_dict: AttachmentDict) -> "Attachment":
        id = attachment_dict.get("id", "")
        thread_id = attachment_dict.get("threadId", None)
        step_id = attachment_dict.get("stepId", None)
        metadata = attachment_dict.get("metadata", {})
        mime = attachment_dict.get("mime", "")
        name = attachment_dict.get("name", "")
        object_key = attachment_dict.get("objectKey", "")
        url = attachment_dict.get("url", "")

        attachment = cls(
            id=id,
            thread_id=thread_id,
            mime=mime,
            name=name,
            object_key=object_key,
            url=url,
            step_id=step_id,
            metadata=metadata,
        )

        return attachment


class StepDict(TypedDict, total=False):
    id: Optional[str]
    name: Optional[str]
    type: Optional[StepType]
    environment: Optional[Environment]
    threadId: Optional[str]
    error: Optional[str]
    input: Optional[Dict]
    output: Optional[Dict]
    metadata: Optional[Dict]
    tags: Optional[List[str]]
    parentId: Optional[str]
    createdAt: Optional[str]
    startTime: Optional[str]
    endTime: Optional[str]
    generation: Optional[Dict]
    scores: Optional[List[ScoreDict]]
    attachments: Optional[List[AttachmentDict]]


class Step(Utils):
    id: str
    name: Optional[str] = ""
    type: Optional[StepType] = None
    metadata: Optional[Dict] = None
    parent_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    created_at: Optional[str] = None
    error: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    thread_id: Optional[str] = None
    environment: Optional[Environment] = None

    generation: Optional[Union[ChatGeneration, CompletionGeneration]] = None
    scores: Optional[List[Score]] = []
    attachments: List[Attachment] = []

    def __init__(
        self,
        name: str = "",
        type: Optional[StepType] = None,
        id: Optional[str] = None,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        processor: Optional["EventProcessor"] = None,
        tags: Optional[List[str]] = None,
    ):
        from time import sleep

        sleep(0.001)
        self.id = id or str(uuid.uuid4())
        self.start_time = utc_now()
        self.name = name
        self.type = type

        self.processor = processor

        # priority for thread_id: thread_id > parent_step.thread_id > active_thread
        self.thread_id = thread_id

        # priority for parent_id: parent_id > parent_step.id
        self.parent_id = parent_id

        self.tags = tags

    def start(self):
        active_steps = active_steps_var.get()
        if len(active_steps) > 0:
            parent_step = active_steps[-1]
            if not self.parent_id:
                self.parent_id = parent_step.id
            if not self.thread_id:
                self.thread_id = parent_step.thread_id

        if not self.thread_id:
            if active_thread := active_thread_var.get():
                self.thread_id = active_thread.id

        active_steps.append(self)
        active_steps_var.set(active_steps)

    def end(self):
        self.end_time = utc_now()

        # Update active steps
        active_steps = active_steps_var.get()

        # Check if step is active
        if self not in active_steps:
            raise Exception("Step must be started before ending.")

        # Remove step from active steps
        active_steps.remove(self)
        active_steps_var.set(active_steps)

        if self.processor is None:
            raise Exception(
                "Step must be stopped with a processor to allow finalization."
            )
        self.processor.add_event(self.to_dict())

    def to_dict(self):
        return {
            "id": self.id,
            "metadata": self.metadata,
            "parentId": self.parent_id,
            "startTime": self.start_time,
            "endTime": self.end_time,
            "type": self.type,
            "threadId": self.thread_id,
            "error": self.error,
            "input": self.input,
            "output": self.output,
            "generation": self.generation.to_dict() if self.generation else None,
            "name": self.name,
            "tags": self.tags,
            "scores": [score.to_dict() for score in self.scores],
            "attachments": [attachment.to_dict() for attachment in self.attachments],
        }

    @classmethod
    def from_dict(cls, step_dict: StepDict) -> "Step":
        name = step_dict.get("name") or ""
        step_type = step_dict.get("type", cast(StepType, "undefined"))
        thread_id = step_dict.get("threadId")

        step = cls(name=name, type=step_type, thread_id=thread_id)

        step.id = step_dict.get("id") or ""
        step.input = step_dict.get("input", None)
        step.error = step_dict.get("error", None)
        step.output = step_dict.get("output", None)
        step.environment = step_dict.get("environment", None)
        step.metadata = step_dict.get("metadata", {})
        step.tags = step_dict.get("tags", [])
        step.parent_id = step_dict.get("parentId", None)
        step.start_time = step_dict.get("startTime", None)
        step.end_time = step_dict.get("endTime", None)
        step.created_at = step_dict.get("createdAt", None)

        if "generation" in step_dict and step_type == "llm":
            generation_dict = step_dict["generation"]
            if generation_dict:
                step.generation = BaseGeneration.from_dict(generation_dict)

        if "scores" in step_dict:
            scores = step_dict["scores"]
            if scores:
                step.scores = [Score.from_dict(score) for score in scores]

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
        client: "BaseLiteralClient",
        name: str = "",
        type: TrueStepType = "undefined",
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs,
    ):
        self.client = client
        self.step_name = name
        self.step_type = type
        self.id = id
        self.parent_id = parent_id
        self.thread_id = thread_id
        self.kwargs = kwargs

    def __call__(self, func):
        return step_decorator(
            self.client,
            func=func,
            name=self.step_name,
            ctx_manager=self,
        )

    async def __aenter__(self):
        self.step = self.client.start_step(
            name=self.step_name,
            type=self.step_type,
            id=self.id,
            parent_id=self.parent_id,
            thread_id=self.thread_id,
            **self.kwargs,
        )
        return self.step

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.step.error = str(exc_val)
            await self.client.event_processor.aflush()
        self.step.end()

    def __enter__(self) -> Step:
        self.step = self.client.start_step(
            name=self.step_name,
            type=self.step_type,
            id=self.id,
            parent_id=self.parent_id,
            thread_id=self.thread_id,
            **self.kwargs,
        )
        return self.step

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.step.error = str(exc_val)
            self.client.event_processor.flush()
        self.step.end()


def flatten_args_kwargs(func, *args, **kwargs):
    signature = inspect.signature(func)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return {k: deepcopy(v) for k, v in bound_arguments.arguments.items()}


def step_decorator(
    client: "BaseLiteralClient",
    func: Callable,
    type: TrueStepType = "undefined",
    name: str = "",
    id: Optional[str] = None,
    parent_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    ctx_manager: Optional[StepContextManager] = None,
    **decorator_kwargs,
):
    if not name:
        name = func.__name__
    if not ctx_manager:
        ctx_manager = StepContextManager(
            client=client,
            type=type,
            name=name,
            id=id,
            parent_id=parent_id,
            thread_id=thread_id,
            **decorator_kwargs,
        )
    else:
        ctx_manager.step_name = name
    # Handle async decorator
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with ctx_manager as step:
                try:
                    step.input = flatten_args_kwargs(func, *args, **kwargs)
                except Exception:
                    pass
                result = await func(*args, **kwargs)
                try:
                    if step.output is None:
                        if isinstance(result, dict):
                            step.output = deepcopy(result)
                        else:
                            step.output = {"content": deepcopy(result)}
                except Exception:
                    pass
                return result

        return async_wrapper
    else:
        # Handle sync decorator
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with ctx_manager as step:
                try:
                    step.input = flatten_args_kwargs(func, *args, **kwargs)
                except Exception:
                    pass
                result = func(*args, **kwargs)
                try:
                    if step.output is None:
                        if isinstance(result, dict):
                            step.output = deepcopy(result)
                        else:
                            step.output = {"content": deepcopy(result)}
                except Exception:
                    pass
                return result

        return sync_wrapper
