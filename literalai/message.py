import uuid
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from literalai.event_processor import EventProcessor

from literalai.context import active_steps_var, active_thread_var
from literalai.helper import utc_now
from literalai.my_types import Attachment, Score, Utils
from literalai.step import MessageStepType, StepDict


class Message(Utils):
    """
    Class representing a message in the system.

    Attributes:
        id (Optional[str]): The unique identifier for the message. Defaults to None.
        name (Optional[str]): The name of the message. Defaults to None.
        type (Optional[MessageStepType]): The type of the message. Defaults to None.
        metadata (Optional[Dict]): The metadata associated with the message. Defaults to {}.
        parent_id (Optional[str]): The identifier of the parent message. Defaults to None.
        timestamp (Optional[str]): The timestamp of the message. Defaults to None.
        content (str): The content of the message.
        thread_id (Optional[str]): The identifier of the thread the message belongs to. Defaults to None.
        tags (Optional[List[str]]): The tags associated with the message. Defaults to None.
        created_at (Optional[str]): The time when the message was created. Defaults to None.
        scores (List[Score]): The scores associated with the message. Defaults to [].
        attachments (List[Attachment]): The attachments associated with the message. Defaults to [].
    """
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

    scores: List[Score] = []
    attachments: List[Attachment] = []

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        type: Optional[MessageStepType] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        scores: List[Score] = [],
        attachments: List[Attachment] = [],
        metadata: Optional[Dict] = {},
        timestamp: Optional[str] = None,
        tags: Optional[List[str]] = [],
        processor: Optional["EventProcessor"] = None,
    ):
        """
        Initializes the Message object.

        Args:
            content (str): The content of the message.
            id (Optional[str], optional): The unique identifier for the message. Defaults to None.
            type (Optional[MessageStepType], optional): The type of the message. Defaults to None.
            name (Optional[str], optional): The name of the message. Defaults to None.
            thread_id (Optional[str], optional): The identifier of the thread the message belongs to. Defaults to None.
            parent_id (Optional[str], optional): The identifier of the parent message. Defaults to None.
            scores (List[Score], optional): The scores associated with the message. Defaults to [].
            attachments (List[Attachment], optional): The attachments associated with the message. Defaults to [].
            metadata (Optional[Dict], optional): The metadata associated with the message. Defaults to {}.
            timestamp (Optional[str], optional): The timestamp of the message. Defaults to None.
            tags (Optional[List[str]], optional): The tags associated with the message. Defaults to None.
            processor (Optional["EventProcessor"], optional): The processor to handle the message. Defaults to None.
        """
        from time import sleep

        sleep(0.001)

        self.id = id or str(uuid.uuid4())
        if not timestamp:
            self.timestamp = utc_now()
        else:
            self.timestamp = timestamp
        self.name = name
        self.type = type
        self.content = content
        self.scores = scores
        self.attachments = attachments
        self.metadata = metadata
        self.tags = tags

        self.processor = processor

        # priority for thread_id: thread_id > parent_step.thread_id > active_thread
        self.thread_id = thread_id

        # priority for parent_id: parent_id > parent_step.id
        self.parent_id = parent_id

    def end(self):
        """
        Ends the message processing and adds the event to the processor.
        """
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

        if not self.thread_id:
            raise Exception("Message must be initialized with a thread_id.")

        if self.processor is None:
            raise Exception(
                "Message must be initialized with a processor to allow finalization."
            )
        self.processor.add_event(self.to_dict())

    def to_dict(self) -> "StepDict":
        """
        Converts the Message object to a StepDict dictionary.

        Returns:
            StepDict: The dictionary representation of the Message object.
        """
        return {
            "id": self.id,
            "metadata": self.metadata,
            "parentId": self.parent_id,
            "startTime": self.timestamp,
            "endTime": self.timestamp,  # startTime = endTime in Message
            "type": self.type,
            "threadId": self.thread_id,
            "output": {
                "content": self.content
            },  # no input, output = content in Message
            "name": self.name,
            "tags": self.tags,
            "scores": [score.to_dict() for score in self.scores],
            "attachments": [attachment.to_dict() for attachment in self.attachments],
        }

    @classmethod
    def from_dict(cls, message_dict: Dict) -> "Message":
        """
        Creates a Message object from a dictionary.

        Args:
            message_dict (Dict): The dictionary representation of the Message object.

        Returns:
            Message: The Message object created from the dictionary.
        """
        id = message_dict.get("id", None)
        type = message_dict.get("type", None)
        thread_id = message_dict.get("threadId", None)

        metadata = message_dict.get("metadata", None)
        parent_id = message_dict.get("parentId", None)
        timestamp = message_dict.get("startTime", None)
        content = message_dict.get("output", None)
        name = message_dict.get("name", None)
        scores = message_dict.get("scores", None)
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
            scores=scores,
            attachments=attachments,
        )

        message.created_at = message_dict.get("createdAt", None)

        return message
