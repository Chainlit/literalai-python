import datetime
import uuid
from functools import wraps
from typing import TYPE_CHECKING, Dict, Literal, Optional

if TYPE_CHECKING:
    from .event_processor import EventProcessor

from .context import active_steps_var, active_thread_id_var

EventRole = Literal["ASSISTANT", "SYSTEM", "USER"]


class Event:
    id: Optional[str] = None
    name: Optional[str] = ""
    metadata: Dict = {}
    message: Optional[str] = None
    role: Optional[EventRole] = None

    thread_id: Optional[str] = None
    step_id: Optional[str] = None
    timestamp: Optional[str] = None

    def __init__(
        self,
        name: str = "",
        id: Optional[str] = None,
        thread_id: Optional[str] = None,
        step_id: Optional[str] = None,
        message: Optional[str] = None,
        role: Optional[EventRole] = None,
        processor: Optional["EventProcessor"] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.timestamp = datetime.datetime.utcnow().isoformat()
        self.name = name
        self.message = message
        self.role = role

        self.processor = processor

        # priority for thread_id: thread_id > parent_step.thread_id > active_thread
        self.thread_id = thread_id

        # priority for step_id: step_id > parent_step.id
        self.step_id = step_id

        active_steps = active_steps_var.get()
        if active_steps:
            parent_step = active_steps[-1]
            if not step_id:
                self.step_id = parent_step.id
            if not thread_id:
                self.thread_id = parent_step.thread_id

        if not self.thread_id:
            if active_thread := active_thread_id_var.get():
                self.thread_id = active_thread

        if not self.thread_id:
            raise Exception("Event must be initialized with a thread_id.")

    def finalize(self):
        if self.processor is None:
            raise Exception(
                "Event must be initialized with a processor to allow finalization."
            )
        # self.processor.add_event(self.to_dict()) # FIXME: add event to the event queue when the API is ready

    def to_dict(self):
        return {
            "id": self.id,
            "metadata": self.metadata,
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "thread_id": self.thread_id,
            "name": self.name,
            "message": self.message,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, step_dict: Dict) -> "Event":
        id = step_dict.get("id", "")
        name = step_dict.get("name", "")
        thread_id = step_dict.get("thread_id")
        step_id = step_dict.get("step_id")
        message = step_dict.get("message")
        role = step_dict.get("role")

        event = cls(
            id=id,
            name=name,
            thread_id=thread_id,
            step_id=step_id,
            message=message,
            role=role,
        )

        event.timestamp = step_dict.get("timestamp", "")
        event.metadata = step_dict.get("metadata", {})

        return event
