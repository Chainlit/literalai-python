import inspect
import uuid
from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional

from pydantic.dataclasses import dataclass

from chainlit_client.context import active_thread_id_var
from chainlit_client.my_types import User
from chainlit_client.step import Step

if TYPE_CHECKING:
    from chainlit_client.client import ChainlitClient


class Thread:
    id: str
    metadata: Optional[Dict]
    tags: Optional[List[str]]
    steps: Optional[List[Step]]
    user: Optional["User"]
    created_at: Optional[str]  # read-only, set by server

    def __init__(
        self,
        id: str,
        steps: Optional[List[Step]] = [],
        metadata: Optional[Dict] = {},
        tags: Optional[List[str]] = [],
        user: Optional["User"] = None,
    ):
        self.id = id
        self.steps = steps
        self.metadata = metadata
        self.tags = tags
        self.user = user

    def to_dict(self):
        return {
            "id": self.id,
            "metadata": self.metadata,
            "tags": self.tags,
            "steps": [step.to_dict() for step in self.steps],
            "participant": self.user.to_dict() if self.user else None,
            "createdAt": self.created_at,
        }

    @classmethod
    def from_dict(cls, thread_dict: Dict) -> "Thread":
        id = thread_dict.get("id", "")
        metadata = thread_dict.get("metadata", {})
        tags = thread_dict.get("tags", [])
        steps = [Step.from_dict(step) for step in thread_dict.get("steps", [])]
        user = thread_dict.get("participant", None)
        created_at = thread_dict.get("createdAt", None)

        if user:
            user = User.from_dict(user)

        thread = cls(id=id, steps=steps, metadata=metadata, tags=tags, user=user)

        thread.created_at = created_at

        return thread


class ThreadContextManager:
    def __init__(
        self,
        client: "ChainlitClient",
        thread_id: Optional[str] = None,
    ):
        self.client = client
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        self.thread_id = thread_id
        active_thread_id_var.set(thread_id)
        self.thread = Thread(id=thread_id)

    def __call__(self, func):
        return thread_decorator(self.client, func=func, thread_id=self.thread_id)

    def __enter__(self) -> Thread:
        return self.thread

    def __exit__(self, exc_type, exc_val, exc_tb):
        active_thread_id_var.set(None)

    async def __aenter__(self):
        return self.thread

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        active_thread_id_var.set(None)


def thread_decorator(
    client: "ChainlitClient", func: Callable, thread_id: Optional[str] = None
):
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with ThreadContextManager(client, thread_id=thread_id):
                result = await func(*args, **kwargs)
                return result

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with ThreadContextManager(client, thread_id=thread_id):
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
    attachmentsName: Optional[StringListFilter] = None
    createdAt: Optional[DateTimeFilter] = None
    afterCreatedAt: Optional[DateTimeFilter] = None
    beforeCreatedAt: Optional[DateTimeFilter] = None
    duration: Optional[NumberFilter] = None
    environment: Optional[StringFilter] = None
    feedbacksValue: Optional[NumberListFilter] = None
    participantsIdentifier: Optional[StringListFilter] = None
    search: Optional[StringFilter] = None
    tokenCount: Optional[NumberFilter] = None

    def to_dict(self):
        return {
            "attachmentsName": self.attachmentsName.to_dict()
            if self.attachmentsName
            else None,
            "createdAt": self.createdAt.to_dict() if self.createdAt else None,
            "afterCreatedAt": self.afterCreatedAt.to_dict()
            if self.afterCreatedAt
            else None,
            "beforeCreatedAt": self.beforeCreatedAt.to_dict()
            if self.beforeCreatedAt
            else None,
            "duration": self.duration.to_dict() if self.duration else None,
            "environment": self.environment.to_dict() if self.environment else None,
            "feedbacksValue": self.feedbacksValue.to_dict()
            if self.feedbacksValue
            else None,
            "participantsIdentifier": self.participantsIdentifier.to_dict()
            if self.participantsIdentifier
            else None,
            "search": self.search.to_dict() if self.search else None,
            "tokenCount": self.tokenCount.to_dict() if self.tokenCount else None,
        }
