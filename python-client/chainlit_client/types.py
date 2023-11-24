import uuid
from enum import Enum, unique
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional


@unique
class MessageRole(Enum):
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"
    USER = "USER"


@unique
class GenerationType(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


# TODO: Split in two classes: ChatGeneration and CompletionGeneration
class Generation:
    template: Optional[str] = None
    formatted: Optional[str] = None
    template_format: Optional[str] = None
    provider: Optional[str] = None
    inputs: Optional[Dict] = None
    completion: Optional[str] = None
    settings: Optional[Dict] = None
    messages: Optional[Any] = None
    tokenCount: Optional[int] = None
    type: Optional[GenerationType] = None

    def to_dict(self):
        return {
            "template": self.template,
            "formatted": self.formatted,
            "template_format": self.template_format,
            "provider": self.provider,
            "inputs": self.inputs,
            "completion": self.completion,
            "settings": self.settings,
            "messages": self.messages,
            "tokenCount": self.tokenCount,
            "type": self.type.value if self.type else None,
        }


@unique
class FeedbackStrategy(Enum):
    BINARY = "BINARY"
    STARS = "STARS"
    BIG_STARS = "BIG_STARS"
    LIKERT = "LIKERT"
    CONTINUOUS = "CONTINUOUS"
    LETTERS = "LETTERS"
    PERCENTAGE = "PERCENTAGE"


class Feedback:
    value: Optional[float] = None
    strategy: FeedbackStrategy = FeedbackStrategy.BINARY
    comment: Optional[str] = None

    def __init__(
        self,
        value: Optional[float] = None,
        strategy: FeedbackStrategy = FeedbackStrategy.BINARY,
        comment: Optional[str] = None,
    ):
        self.value = value
        self.strategy = strategy
        self.comment = comment

    def to_dict(self):
        return {
            "value": self.value,
            "strategy": self.strategy.value if self.strategy else None,
            "comment": self.comment,
        }


class Attachment:
    id: Optional[str] = None
    mime: Optional[str] = None
    name: Optional[str] = None
    objectKey: Optional[str] = None
    url: Optional[str] = None

    def __init__(
        self,
        id: Optional[str] = None,
        mime: Optional[str] = None,
        name: Optional[str] = None,
        objectKey: Optional[str] = None,
        url: Optional[str] = None,
    ):
        self.id = id
        if self.id is None:
            self.id = str(uuid.uuid4())
        self.mime = mime
        self.name = name
        self.objectKey = objectKey
        self.url = url

    def to_dict(self):
        return {
            "id": self.id,
            "mime": self.mime,
            "name": self.name,
            "objectKey": self.objectKey,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, attachment_dict: Dict) -> "Attachment":
        id = attachment_dict.get("id", "")
        mime = attachment_dict.get("mime", "")
        name = attachment_dict.get("name", "")
        objectKey = attachment_dict.get("objectKey", "")
        url = attachment_dict.get("url", "")

        attachment = cls(id=id, mime=mime, name=name, objectKey=objectKey, url=url)

        return attachment
