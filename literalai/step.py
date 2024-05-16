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
    TypedDict,
    Union,
)

if TYPE_CHECKING:
    from literalai.client import BaseLiteralClient
    from literalai.event_processor import EventProcessor

from literalai.context import active_steps_var, active_thread_var
from literalai.helper import utc_now
from literalai.my_types import (
    Attachment,
    AttachmentDict,
    BaseGeneration,
    ChatGeneration,
    CompletionGeneration,
    Score,
    ScoreDict,
    Utils,
)

TrueStepType = Literal[
    "run", "tool", "llm", "embedding", "retrieval", "rerank", "undefined"
]

MessageStepType = Literal["user_message", "assistant_message", "system_message"]

StepType = Union[TrueStepType, MessageStepType]


class StepDict(TypedDict, total=False):
    id: Optional[str]
    name: Optional[str]
    type: Optional[StepType]
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
    id: Optional[str] = None
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
        step_type = step_dict.get("type", "undefined")
        thread_id = step_dict.get("threadId")

        step = cls(name=name, type=step_type, thread_id=thread_id)

        step.id = step_dict.get("id")
        step.input = step_dict.get("input", None)
        step.error = step_dict.get("error", None)
        step.output = step_dict.get("output", None)
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
            **self.kwargs
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
            **self.kwargs
        )
        return self.step

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.step.error = str(exc_val)
            self.client.event_processor.flush()
        self.step.end()


def step_decorator(
    client: "BaseLiteralClient",
    func: Callable,
    type: TrueStepType = "undefined",
    name: str = "",
    id: Optional[str] = None,
    parent_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    ctx_manager: Optional[StepContextManager] = None,
    **decorator_kwargs
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
            **decorator_kwargs
        )
    else:
        ctx_manager.step_name = name
    # Handle async decorator
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with ctx_manager as step:
                try:
                    step.input = {"args": deepcopy(args), "kwargs": deepcopy(kwargs)}
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
                    step.input = {"args": deepcopy(args), "kwargs": deepcopy(kwargs)}
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
