import inspect
import uuid
from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, TypedDict

from pydantic.dataclasses import dataclass

from literalai.context import active_thread_var
from literalai.my_types import User, UserDict
from literalai.step import Step, StepDict

if TYPE_CHECKING:
    from literalai.client import LiteralClient


class ThreadDict(TypedDict, total=False):
    id: Optional[str]
    name: Optional[str]
    metadata: Optional[Dict]
    tags: Optional[List[str]]
    createdAt: Optional[str]
    steps: Optional[List[StepDict]]
    participant: Optional[UserDict]


class Thread:
    id: str
    name: Optional[str]
    metadata: Optional[Dict]
    tags: Optional[List[str]]
    steps: Optional[List[Step]]
    user: Optional["User"]
    created_at: Optional[str]  # read-only, set by server
    needs_upsert: Optional[bool]

    def __init__(
        self,
        id: str,
        steps: Optional[List[Step]] = [],
        name: Optional[str] = None,
        metadata: Optional[Dict] = {},
        tags: Optional[List[str]] = [],
        user: Optional["User"] = None,
    ):
        self.id = id
        self.steps = steps
        self.name = name
        self.metadata = metadata
        self.tags = tags
        self.user = user
        self.needs_upsert = bool(metadata or tags or user or name)

    def to_dict(self) -> ThreadDict:
        return {
            "id": self.id,
            "metadata": self.metadata,
            "tags": self.tags,
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps] if self.steps else [],
            "participant": self.user.to_dict() if self.user else None,
            "createdAt": getattr(self, "created_at", None),
        }

    @classmethod
    def from_dict(cls, thread_dict: Dict) -> "Thread":
        id = thread_dict.get("id", "")
        name = thread_dict.get("name", None)
        metadata = thread_dict.get("metadata", {})
        tags = thread_dict.get("tags", [])
        steps = [Step.from_dict(step) for step in thread_dict.get("steps", [])]
        user = thread_dict.get("participant", None)
        created_at = thread_dict.get("createdAt", None)

        if user:
            user = User.from_dict(user)

        thread = cls(
            id=id, steps=steps, name=name, metadata=metadata, tags=tags, user=user
        )

        thread.created_at = created_at

        return thread


class ThreadContextManager:
    def __init__(
        self,
        client: "LiteralClient",
        thread_id: "Optional[str]" = None,
        name: "Optional[str]" = None,
        **kwargs,
    ):
        self.client = client
        self.thread_id = thread_id
        self.name = name
        self.kwargs = kwargs

    def upsert(self):
        thread = active_thread_var.get()
        thread_data = thread.to_dict()
        thread_data_to_upsert = {
            "thread_id": thread_data["id"],
            "name": thread_data["name"],
        }
        if metadata := thread_data.get("metadata"):
            thread_data_to_upsert["metadata"] = metadata
        if tags := thread_data.get("tags"):
            thread_data_to_upsert["tags"] = tags
        if user := thread_data.get("user"):
            thread_data_to_upsert["participant_id"] = user
        self.client.api.upsert_thread_sync(**thread_data_to_upsert)

    def __call__(self, func):
        return thread_decorator(
            self.client, func=func, name=self.name, ctx_manager=self
        )

    def __enter__(self) -> "Optional[Thread]":
        thread_id = self.thread_id if self.thread_id else str(uuid.uuid4())
        active_thread_var.set(Thread(id=thread_id, name=self.name, **self.kwargs))
        return active_thread_var.get()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (thread := active_thread_var.get()) and thread.needs_upsert:
            self.upsert()
        active_thread_var.set(None)

    async def __aenter__(self):
        thread_id = self.thread_id if self.thread_id else str(uuid.uuid4())
        active_thread_var.set(Thread(id=thread_id, name=self.name, **self.kwargs))
        return active_thread_var.get()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if (thread := active_thread_var.get()) and thread.needs_upsert:
            self.upsert()
        active_thread_var.set(None)


def thread_decorator(
    client: "LiteralClient",
    func: Callable,
    thread_id: Optional[str] = None,
    name: Optional[str] = None,
    ctx_manager: Optional[ThreadContextManager] = None,
    **decorator_kwargs,
):
    if not ctx_manager:
        ctx_manager = ThreadContextManager(
            client, thread_id=thread_id, name=name, **decorator_kwargs
        )
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with ctx_manager:
                result = await func(*args, **kwargs)
                return result

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with ctx_manager:
                return func(*args, **kwargs)

        return sync_wrapper


StringOperators = Literal["eq", "ilike", "like", "neq", "nilike", "nlike"]
StringListOperators = Literal["in", "nin"]
NumberListOperators = Literal["in", "nin"]
NumberOperators = Literal["eq", "gt", "gte", "lt", "lte", "neq"]
DateTimeOperators = Literal["gt", "gte", "lt", "lte"]


@dataclass
class StringFilter:
    operator: StringOperators
    value: str

    def to_dict(self):
        return {
            "operator": self.operator,
            "value": self.value,
        }


@dataclass
class StringListFilter:
    operator: StringListOperators
    value: List[str]

    def to_dict(self):
        return {
            "operator": self.operator,
            "value": self.value,
        }


@dataclass
class NumberListFilter:
    operator: NumberListOperators
    value: List[float]

    def to_dict(self):
        return {
            "operator": self.operator,
            "value": self.value,
        }


@dataclass
class NumberFilter:
    operator: NumberOperators
    value: float

    def to_dict(self):
        return {
            "operator": self.operator,
            "value": self.value,
        }


@dataclass
class DateTimeFilter:
    operator: DateTimeOperators
    value: str

    def to_dict(self):
        return {
            "operator": self.operator,
            "value": self.value,
        }


@dataclass
class ThreadFilter:
    # attachmentsName: Optional[StringListFilter] = None
    createdAt: Optional[DateTimeFilter] = None
    afterCreatedAt: Optional[DateTimeFilter] = None
    beforeCreatedAt: Optional[DateTimeFilter] = None
    # duration: Optional[NumberFilter] = None
    environment: Optional[StringFilter] = None
    feedbacksValue: Optional[NumberListFilter] = None
    participantsIdentifier: Optional[StringListFilter] = None
    search: Optional[StringFilter] = None
    # tokenCount: Optional[NumberFilter] = None

    def to_dict(self):
        return {
            # "attachmentsName": self.attachmentsName.to_dict()
            # if self.attachmentsName
            # else None,
            "createdAt": self.createdAt.to_dict() if self.createdAt else None,
            "afterCreatedAt": self.afterCreatedAt.to_dict()
            if self.afterCreatedAt
            else None,
            "beforeCreatedAt": self.beforeCreatedAt.to_dict()
            if self.beforeCreatedAt
            else None,
            # "duration": self.duration.to_dict() if self.duration else None,
            "environment": self.environment.to_dict() if self.environment else None,
            "feedbacksValue": self.feedbacksValue.to_dict()
            if self.feedbacksValue
            else None,
            "participantsIdentifier": self.participantsIdentifier.to_dict()
            if self.participantsIdentifier
            else None,
            "search": self.search.to_dict() if self.search else None,
            # "tokenCount": self.tokenCount.to_dict() if self.tokenCount else None,
        }
