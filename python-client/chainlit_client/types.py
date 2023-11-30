import uuid
from enum import Enum, unique
from typing import Any, Dict, Literal, Optional, List
from .event import Event

from pydantic.dataclasses import Field, dataclass

MessageType = Literal["USER_MESSAGE", "ASSISTANT_MESSAGE", "SYSTEM_MESSAGE"]

GenerationMessageRole = Literal["user", "assistant", "tool", "function", "system"]
MessageRole = Literal["ASSISTANT", "SYSTEM", "USER", "TOOL"]
FeedbackStrategy = Literal[
    "BINARY", "STARS", "BIG_STARS", "LIKERT", "CONTINUOUS", "LETTERS", "PERCENTAGE"
]


@unique
class GenerationType(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


@dataclass
class GenerationMessage:
    template: Optional[str] = None
    formatted: Optional[str] = None
    # This is used for Langchain's MessagesPlaceholder
    placeholder_size: Optional[int] = None
    # This is used for OpenAI's function message
    name: Optional[str] = None
    role: Optional[GenerationMessageRole] = None
    template_format: str = "f-string"

    def to_openai(self):
        msg_dict = {"role": self.role, "content": self.formatted}
        if self.role == "function":
            msg_dict["name"] = self.name or ""
        return msg_dict

    def to_string(self):
        return f"{self.role}: {self.formatted}"

    def to_dict(self):
        return {
            "template": self.template,
            "formatted": self.formatted,
            "placeholder_size": self.placeholder_size,
            "name": self.name,
            "role": self.role if self.role else None,
            "templateFormat": self.template_format,
        }

    @classmethod
    def from_dict(self, message_dict: Dict):
        return GenerationMessage(
            template=message_dict.get("template"),
            formatted=message_dict.get("formatted"),
            placeholder_size=message_dict.get("placeholder_size"),
            name=message_dict.get("name"),
            role=message_dict.get("role", "ASSISTANT"),
        )


@dataclass
class BaseGeneration:
    provider: Optional[str] = None
    inputs: Optional[Dict] = None
    completion: Optional[str] = None
    settings: Optional[Dict] = None
    token_count: Optional[int] = None
    functions: Optional[List[Dict]] = None

    @classmethod
    def from_dict(self, generation_dict: Dict) -> "BaseGeneration":
        type = GenerationType(generation_dict.get("type"))
        if type == GenerationType.CHAT:
            return ChatGeneration.from_dict(generation_dict)
        elif type == GenerationType.COMPLETION:
            return CompletionGeneration.from_dict(generation_dict)
        else:
            raise ValueError(f"Unknown generation type: {type}")

    def to_dict(self):
        raise NotImplementedError()


@dataclass
class CompletionGeneration(BaseGeneration):
    template: Optional[str] = None
    formatted: Optional[str] = None
    template_format: str = "f-string"
    type = GenerationType.COMPLETION

    def to_dict(self):
        return {
            "template": self.template,
            "formatted": self.formatted,
            "template_format": self.template_format,
            "provider": self.provider,
            "inputs": self.inputs,
            "completion": self.completion,
            "settings": self.settings,
            "tokenCount": self.token_count,
            "type": self.type.value,
        }

    @classmethod
    def from_dict(self, generation_dict: Dict) -> "CompletionGeneration":
        return CompletionGeneration(
            template=generation_dict.get("template"),
            formatted=generation_dict.get("formatted"),
            template_format=generation_dict.get("template_format", "f-string"),
            provider=generation_dict.get("provider"),
            completion=generation_dict.get("completion"),
            settings=generation_dict.get("settings"),
            token_count=generation_dict.get("tokenCount"),
        )


@dataclass
class ChatGeneration(BaseGeneration):
    messages: List[GenerationMessage] = Field(default_factory=list)
    type = GenerationType.CHAT

    def to_dict(self):
        return {
            "messages": [m.to_dict() for m in self.messages],
            "provider": self.provider,
            "inputs": self.inputs,
            "completion": self.completion,
            "settings": self.settings,
            "tokenCount": self.token_count,
            "type": self.type.value,
        }

    @classmethod
    def from_dict(self, generation_dict: Dict) -> "ChatGeneration":
        return ChatGeneration(
            messages=[
                GenerationMessage.from_dict(m)
                for m in generation_dict.get("messages", [])
            ],
            provider=generation_dict.get("provider"),
            completion=generation_dict.get("completion"),
            settings=generation_dict.get("settings"),
            token_count=generation_dict.get("tokenCount"),
        )


@dataclass
class Feedback:
    value: Optional[float] = None
    strategy: FeedbackStrategy = "BINARY"
    comment: Optional[str] = None

    def to_dict(self):
        return {
            "value": self.value,
            "strategy": self.strategy,
            "comment": self.comment,
        }


@dataclass
class Attachment:
    step_id: str
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    mime: Optional[str] = None
    name: Optional[str] = None
    objectKey: Optional[str] = None
    url: Optional[str] = None
    display: Optional[str] = None
    type: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "mime": self.mime,
            "name": self.name,
            "objectKey": self.objectKey,
            "url": self.url,
            "display": self.display,
            "step_id": self.step_id,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, attachment_dict: Dict) -> "Attachment":
        id = attachment_dict.get("id", "")
        mime = attachment_dict.get("mime", "")
        name = attachment_dict.get("name", "")
        objectKey = attachment_dict.get("objectKey", "")
        url = attachment_dict.get("url", "")
        display = attachment_dict.get("display", "")
        step_id = attachment_dict.get("step_id", "")
        type = attachment_dict.get("type", "")

        attachment = cls(
            id=id,
            mime=mime,
            name=name,
            objectKey=objectKey,
            url=url,
            display=display,
            step_id=step_id,
            type=type,
        )

        return attachment


@dataclass
class Participant:
    identifier: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict = Field(default_factory=lambda: {})

    def to_dict(self):
        return {
            "identifier": self.identifier,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, user_dict: Dict) -> "Participant":
        identifier = user_dict.get("identifier", "")
        metadata = user_dict.get("metadata", {})

        participant = cls(
            identifier=identifier,
            metadata=metadata,
        )

        return participant
