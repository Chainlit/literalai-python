import datetime
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from chainlit_client.event_processor import EventProcessor

from chainlit_client.context import active_steps_var, active_thread_id_var
from chainlit_client.my_types import Attachment, Feedback
from chainlit_client.step import MessageStepType, StepDict


class Message:
    id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[MessageStepType] = None
    metadata: Optional[Dict] = {}
    parent_id: Optional[str] = None
    timestamp: Optional[str] = None
    content: str
    thread_id: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: Optional[str] = None

    feedback: Optional[Feedback] = None
    attachments: List[Attachment] = []

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        type: Optional[MessageStepType] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        feedback: Optional[Feedback] = None,
        attachments: List[Attachment] = [],
        metadata: Optional[Dict] = {},
        timestamp: Optional[str] = None,
        tags: Optional[List[str]] = [],
        processor: Optional["EventProcessor"] = None,
    ):
        from time import sleep

        sleep(0.001)

        self.id = id or str(uuid.uuid4())
        if not timestamp:
            self.timestamp = datetime.datetime.utcnow().isoformat()
        else:
            self.timestamp = timestamp
        self.name = name
        self.type = type
        self.content = content
        self.feedback = feedback
        self.attachments = attachments
        self.metadata = metadata
        self.tags = tags

        self.processor = processor

        # priority for thread_id: thread_id > parent_step.thread_id > active_thread
        self.thread_id = thread_id

        # priority for parent_id: parent_id > parent_step.id
        self.parent_id = parent_id

    def end(self):
        active_steps = active_steps_var.get()
        if len(active_steps) > 0:
            parent_step = active_steps[-1]
            if not self.parent_id:
                self.parent_id = parent_step.id
            if not self.thread_id:
                self.thread_id = parent_step.thread_id

        if not self.thread_id:
            if active_thread := active_thread_id_var.get():
                self.thread_id = active_thread

        if not self.thread_id:
            raise Exception("Message must be initialized with a thread_id.")

        if self.processor is None:
            raise Exception(
                "Message must be initialized with a processor to allow finalization."
            )
        self.processor.add_event(self.to_dict())

    def to_dict(self) -> "StepDict":
        # Create a correct step Dict from a message
        return {
            "id": self.id,
            "metadata": self.metadata,
            "parentId": self.parent_id,
            "startTime": self.timestamp,
            "endTime": self.timestamp,  # startTime = endTime in Message
            "type": self.type,
            "threadId": self.thread_id,
            "output": self.content,  # no input, output = content in Message
            "name": self.name,
            "tags": self.tags,
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "attachments": [attachment.to_dict() for attachment in self.attachments],
        }

    @classmethod
    def from_dict(cls, message_dict: Dict) -> "Message":
        id = message_dict.get("id", None)
        type = message_dict.get("type", None)
        thread_id = message_dict.get("threadId", None)

        metadata = message_dict.get("metadata", None)
        parent_id = message_dict.get("parentId", None)
        timestamp = message_dict.get("startTime", None)
        content = message_dict.get("output", None)
        name = message_dict.get("name", None)
        feedback = message_dict.get("feedback", None)
        attachments = message_dict.get("attachments", [])

        message = cls(
            id=id,
            metadata=metadata,
            parent_id=parent_id,
            timestamp=timestamp,
            type=type,
            thread_id=thread_id,
            content=content,
            name=name,
            feedback=feedback,
            attachments=attachments,
        )

        message.created_at = message_dict.get("createdAt", None)

        return message
