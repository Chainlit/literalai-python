import json
import datetime
import uuid
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from .event_processor import EventProcessor

from .context import active_steps_var, active_thread_id_var
from .types import Attachment, Feedback

MessageType = Literal["USER_MESSAGE", "ASSISTANT_MESSAGE", "SYSTEM_MESSAGE"]


class Message:
    id: Optional[str] = None
    name: Optional[str] = ""
    type: Optional[MessageType] = None
    metadata: Dict = {}
    parent_id: Optional[str] = None
    start: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None

    feedback: Optional[Feedback] = None
    attachments: List[Attachment] = []

    def __init__(
        self,
        content: str = "",
        id: Optional[str] = None,
        type: Optional[MessageType] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        feedback: Optional[Feedback] = None,
        attachments: List[Attachment] = [],
        metadata: Dict = {},
        start: Optional[str] = None,
        processor: Optional["EventProcessor"] = None,
    ):
        self.id = id or str(uuid.uuid4())
        if not start:
            self.start = datetime.datetime.utcnow().isoformat()
        else:
            self.start = start
        self.name = name
        self.type = type
        self.output = content
        self.feedback = feedback
        self.attachments = attachments
        self.metadata = metadata

        self.processor = processor

        # priority for thread_id: thread_id > parent_step.thread_id > active_thread
        self.thread_id = thread_id

        # priority for parent_id: parent_id > parent_step.id
        self.parent_id = parent_id

        if active_steps := active_steps_var.get():
            parent_step = active_steps[-1]
            if not parent_id:
                self.parent_id = parent_step.id
            if not thread_id:
                self.thread_id = parent_step.thread_id

        if not self.thread_id:
            if active_thread := active_thread_id_var.get():
                self.thread_id = active_thread

        if not self.thread_id:
            raise Exception("Message must be initialized with a thread_id.")

    def finalize(self):
        if self.processor is None:
            raise Exception(
                "Message must be initialized with a processor to allow finalization."
            )
        self.processor.add_event(self.to_dict())

    def to_dict(self):
        # Create a correct step Dict from a message
        return {
            "id": self.id,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "start": self.start,
            "end": self.start,  # start = end in Message
            "type": self.type,
            "thread_id": self.thread_id,
            "output": self.output,  # no input, output = content in Message
            "name": self.name,
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "attachments": [attachment.to_dict() for attachment in self.attachments],
        }

    @classmethod
    def from_dict(cls, message_dict: Dict) -> "Message":
        id = message_dict.get("id", "")
        metadata = message_dict.get("metadata", {})
        parent_id = message_dict.get("parent_id", "")
        start = message_dict.get("start", "")
        type = message_dict.get("type", "")
        thread_id = message_dict.get("thread_id", "")
        output = message_dict.get("output", "")
        name = message_dict.get("name", "")
        feedback = message_dict.get("feedback", "")
        attachments = message_dict.get("attachments", "")

        message = cls(
            id=id,
            metadata=metadata,
            parent_id=parent_id,
            start=start,
            type=type,
            thread_id=thread_id,
            content=output,
            name=name,
            feedback=feedback,
            attachments=attachments,
        )

        return message
