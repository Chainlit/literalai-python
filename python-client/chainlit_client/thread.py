import inspect
import uuid
from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from .context import active_thread_id_var
from .step import Step

if TYPE_CHECKING:
    from .client import ChainlitClient
    from .types import User


class Thread:
    id: str
    metadata: Optional[Dict]
    tags: Optional[List[str]]
    steps: Optional[List[Step]]
    user: Optional["User"]

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
        }

    @classmethod
    def from_dict(cls, thread_dict: Dict) -> "Thread":
        id = thread_dict.get("id", "")
        metadata = thread_dict.get("metadata", {})
        tags = thread_dict.get("tags", [])
        steps = [Step.from_dict(step) for step in thread_dict.get("steps", [])]
        user = thread_dict.get("participant", None)

        if user:
            user = User.from_dict(user)

        thread = cls(id=id, steps=steps, metadata=metadata, tags=tags, user=user)

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
