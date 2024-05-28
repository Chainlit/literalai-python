import inspect
import logging
import traceback
import uuid
from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, TypedDict

from literalai.context import active_thread_var
from literalai.my_types import UserDict, Utils
from literalai.step import Step, StepDict

if TYPE_CHECKING:
    from literalai.client import BaseLiteralClient

logger = logging.getLogger(__name__)


class ThreadDict(TypedDict, total=False):
    """
    A dictionary representation of a Thread.

    Attributes (all optional):
        id (str): The unique identifier for the thread.
        name (str): The name of the thread.
        metadata (Dict[str, Any]): Additional metadata for the thread.
        tags (List[str]): A list of tags associated with the thread.
        createdAt (str): The timestamp when the thread was created.
        steps (List[StepDict]): The steps in the thread.
        participant (UserDict): The participant in the thread.
    """
    id: Optional[str]
    name: Optional[str]
    metadata: Optional[Dict]
    tags: Optional[List[str]]
    createdAt: Optional[str]
    steps: Optional[List[StepDict]]
    participant: Optional[UserDict]


class Thread(Utils):
    """
    A class representing a thread of steps.

    Attributes:
        id (str): The unique identifier for the thread.
        name (Optional[str]): The name of the thread. Defaults to None.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the thread. Defaults to None.
        tags (Optional[List[str]]): A list of tags associated with the thread. Defaults to None.
        steps (Optional[List[Step]]): The steps in the thread. Defaults to None.
        participant_id (Optional[str]): The identifier of the participant in the thread. Defaults to None.
        participant_identifier (Optional[str]): The identifier of the participant in the thread. Defaults to None.
        created_at (Optional[str]): The timestamp when the thread was created. Defaults to None.
        needs_upsert (Optional[bool]): A flag indicating whether the thread needs to be updated. Defaults to None.
    """
    id: str
    name: Optional[str]
    metadata: Optional[Dict]
    tags: Optional[List[str]]
    steps: Optional[List[Step]]
    participant_id: Optional[str]
    participant_identifier: Optional[str] = None
    created_at: Optional[str]  # read-only, set by server
    needs_upsert: Optional[bool]

    def __init__(
        self,
        id: str,
        steps: Optional[List[Step]] = [],
        name: Optional[str] = None,
        metadata: Optional[Dict] = {},
        tags: Optional[List[str]] = [],
        participant_id: Optional[str] = None,
    ):
        """
        Initializes a Thread object.

        Args:
            id (str): The unique identifier for the thread.
            steps (Optional[List[Step]], optional): The steps in the thread. Defaults to [].
            name (Optional[str], optional): The name of the thread. Defaults to None.
            metadata (Optional[Dict[str, Any]], optional): Additional metadata for the thread. Defaults to {}.
            tags (Optional[List[str]], optional): A list of tags associated with the thread. Defaults to [].
            participant_id (Optional[str], optional): The identifier of the participant in the thread. Defaults to None.
        """
        self.id = id
        self.steps = steps
        self.name = name
        self.metadata = metadata
        self.tags = tags
        self.participant_id = participant_id
        self.needs_upsert = bool(metadata or tags or participant_id or name)

    def to_dict(self) -> ThreadDict:
        """
        Converts the Thread object to a ThreadDict dictionary.

        Returns:
            ThreadDict: The dictionary representation of the Thread object.
        """
        ...
        return {
            "id": self.id,
            "metadata": self.metadata,
            "tags": self.tags,
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps] if self.steps else [],
            "participant": UserDict(
                id=self.participant_id, identifier=self.participant_identifier
            )
            if self.participant_id
            else UserDict(),
            "createdAt": getattr(self, "created_at", None),
        }

    @classmethod
    def from_dict(cls, thread_dict: ThreadDict) -> "Thread":
        """
        Creates a Thread object from a ThreadDict dictionary.

        Args:
            thread_dict (ThreadDict): The dictionary representation of the Thread object.

        Returns:
            Thread: The Thread object created from the dictionary.
        """
        step_dict_list = thread_dict.get("steps", None) or []
        id = thread_dict.get("id", None) or ""
        name = thread_dict.get("name", None)
        metadata = thread_dict.get("metadata", {})
        tags = thread_dict.get("tags", [])
        steps = [Step.from_dict(step_dict) for step_dict in step_dict_list]
        participant = thread_dict.get("participant", None)
        participant_id = participant.get("id", None) if participant else None
        participant_identifier = (
            participant.get("identifier", None) if participant else None
        )
        created_at = thread_dict.get("createdAt", None)

        thread = cls(
            id=id,
            steps=steps,
            name=name,
            metadata=metadata,
            tags=tags,
            participant_id=participant_id,
        )

        thread.created_at = created_at
        thread.participant_identifier = participant_identifier

        return thread


class ThreadContextManager:
    """
    A context manager for handling threads.

    Attributes:
        client (BaseLiteralClient): The client to handle the thread.
        thread_id (Optional[str]): The unique identifier for the thread. Defaults to None.
        name (Optional[str]): The name of the thread. Defaults to None.
        kwargs (Dict[str, Any]): Additional keyword arguments for the thread.
    """

    def __init__(
        self,
        client: "BaseLiteralClient",
        thread_id: "Optional[str]" = None,
        name: "Optional[str]" = None,
        **kwargs,
    ):
        """
        Initializes a ThreadContextManager object.

        Args:
            client (BaseLiteralClient): The client to handle the thread.
            thread_id (Optional[str], optional): The unique identifier for the thread. Defaults to None.
            name (Optional[str], optional): The name of the thread. Defaults to None.
            **kwargs: Additional keyword arguments for the thread.
        """
        self.client = client
        self.thread_id = thread_id
        self.name = name
        self.kwargs = kwargs

    def upsert(self):
        """
        Updates the thread on the server, if necessary.
        """
        if self.client.disabled:
            return

        thread = active_thread_var.get()
        thread_data = thread.to_dict()
        thread_data_to_upsert = {
            "id": thread_data["id"],
            "name": thread_data["name"],
        }
        if metadata := thread_data.get("metadata"):
            thread_data_to_upsert["metadata"] = metadata
        if tags := thread_data.get("tags"):
            thread_data_to_upsert["tags"] = tags
        if participant_id := thread_data.get("participant", {}).get("id"):
            thread_data_to_upsert["participant_id"] = participant_id

        try:
            self.client.to_sync().api.upsert_thread(**thread_data_to_upsert)
        except Exception:
            logger.error(f"Failed to upsert thread: {traceback.format_exc()}")

    def __call__(self, func):
        """
        Calls the function with the thread context.

        Args:
            func (Callable): The function to call.

        Returns:
            Callable: The function with the thread context.
        """
        return thread_decorator(
            self.client, func=func, name=self.name, ctx_manager=self
        )

    def __enter__(self) -> "Optional[Thread]":
        """
        Enters the synchronous context and sets the active thread.

        Returns:
            Optional[Thread]: The active thread.
        """
        thread_id = self.thread_id if self.thread_id else str(uuid.uuid4())
        active_thread_var.set(Thread(id=thread_id, name=self.name, **self.kwargs))
        return active_thread_var.get()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the synchronous context, updates the thread if necessary, and resets the active thread.

        Args:
            exc_type: The type of the exception, if any.
            exc_val: The exception object, if any.
            exc_tb: The traceback object, if any.
        """
        if (thread := active_thread_var.get()) and thread.needs_upsert:
            self.upsert()
        active_thread_var.set(None)

    async def __aenter__(self):
        """
        Enters the asynchronous context and sets the active thread.

        Returns:
            Optional[Thread]: The active thread.
        """
        thread_id = self.thread_id if self.thread_id else str(uuid.uuid4())
        active_thread_var.set(Thread(id=thread_id, name=self.name, **self.kwargs))
        return active_thread_var.get()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the asynchronous context, updates the thread if necessary, and resets the active thread.

        Args:
            exc_type: The type of the exception, if any.
            exc_val: The exception object, if any.
            exc_tb: The traceback object, if any.
        """
        if (thread := active_thread_var.get()) and thread.needs_upsert:
            self.upsert()
        active_thread_var.set(None)


def thread_decorator(
    client: "BaseLiteralClient",
    func: Callable,
    thread_id: Optional[str] = None,
    name: Optional[str] = None,
    ctx_manager: Optional[ThreadContextManager] = None,
    **decorator_kwargs,
):
    """
    A decorator for handling threads.

    Args:
        client (BaseLiteralClient): The client to handle the thread.
        func (Callable): The function to decorate.
        thread_id (Optional[str], optional): The unique identifier for the thread. Defaults to None.
        name (Optional[str], optional): The name of the thread. Defaults to None.
        ctx_manager (Optional[ThreadContextManager], optional): The context manager for the thread. Defaults to None.
        **decorator_kwargs: Additional keyword arguments for the decorator.

    Returns:
        Callable: The decorated function.
    """
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
