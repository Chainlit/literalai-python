import uuid
from enum import Enum, unique
from typing import (
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
)

from pydantic.dataclasses import Field, dataclass

GenerationMessageRole = Literal["user", "assistant", "tool", "function", "system"]
FeedbackStrategy = Literal[
    "BINARY", "STARS", "BIG_STARS", "LIKERT", "CONTINUOUS", "LETTERS", "PERCENTAGE"
]


@dataclass
class PageInfo:
    hasNextPage: bool
    endCursor: Optional[str]

    def to_dict(self):
        return {
            "hasNextPage": self.hasNextPage,
            "endCursor": self.endCursor,
        }

    @classmethod
    def from_dict(cls, page_info_dict: Dict) -> "PageInfo":
        hasNextPage = page_info_dict.get("hasNextPage", False)
        endCursor = page_info_dict.get("endCursor", None)
        return cls(hasNextPage=hasNextPage, endCursor=endCursor)


T = TypeVar("T", covariant=True)


class HasFromDict(Protocol[T]):
    @classmethod
    def from_dict(cls, obj_dict: Dict) -> T:
        raise NotImplementedError()


@dataclass
class PaginatedResponse(Generic[T]):
    pageInfo: PageInfo
    data: List[T]

    def to_dict(self):
        return {
            "pageInfo": self.pageInfo.to_dict(),
            "data": [
                (d.to_dict() if hasattr(d, "to_dict") and callable(d.to_dict) else d)
                for d in self.data
            ],
        }

    @classmethod
    def from_dict(
        cls, paginated_response_dict: Dict, the_class: HasFromDict[T]
    ) -> "PaginatedResponse[T]":
        pageInfo = PageInfo.from_dict(paginated_response_dict.get("pageInfo", {}))

        data = [the_class.from_dict(d) for d in paginated_response_dict.get("data", [])]

        return cls(pageInfo=pageInfo, data=data)


@dataclass
class PaginatedRestResponse(Generic[T]):
    totalCount: int
    totalPage: int
    hasNextPage: bool
    hasPreviousPage: bool
    data: List[T]


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
            "placeholderSize": self.placeholder_size,
            "name": self.name,
            "role": self.role if self.role else None,
            "templateFormat": self.template_format,
        }

    @classmethod
    def from_dict(self, message_dict: Dict):
        return GenerationMessage(
            template=message_dict.get("template"),
            formatted=message_dict.get("formatted"),
            placeholder_size=message_dict.get("placeholderSize"),
            template_format=message_dict.get("templateFormat") or "f-string",
            name=message_dict.get("name"),
            role=message_dict.get("role") or "assistant",
        )


@dataclass
class BaseGeneration:
    provider: Optional[str] = "Unknown"
    inputs: Optional[Dict] = Field(default_factory=dict)
    completion: Optional[str] = None
    settings: Optional[Dict] = Field(default_factory=dict)
    token_count: Optional[int] = None
    functions: Optional[List[Dict]] = None

    @classmethod
    def from_dict(
        self, generation_dict: Dict
    ) -> Union["ChatGeneration", "CompletionGeneration"]:
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
            "templateFormat": self.template_format,
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
            template_format=generation_dict.get("templateFormat") or "f-string",
            provider=generation_dict.get("provider"),
            completion=generation_dict.get("completion"),
            settings=generation_dict.get("settings"),
            inputs=generation_dict.get("inputs", {}),
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
            inputs=generation_dict.get("inputs", {}),
            provider=generation_dict.get("provider"),
            completion=generation_dict.get("completion"),
            settings=generation_dict.get("settings"),
            token_count=generation_dict.get("tokenCount"),
        )


class FeedbackDict(TypedDict, total=False):
    id: Optional[str]
    threadId: Optional[str]
    stepId: Optional[str]
    value: Optional[float]
    strategy: FeedbackStrategy
    comment: Optional[str]


class AttachmentDict(TypedDict, total=False):
    id: Optional[str]
    threadId: str
    stepId: str
    metadata: Optional[Dict]
    mime: Optional[str]
    name: Optional[str]
    objectKey: Optional[str]
    url: Optional[str]


@dataclass
class Feedback:
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: Optional[str] = None
    step_id: Optional[str] = None
    value: Optional[float] = None
    strategy: FeedbackStrategy = "BINARY"
    comment: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "threadId": self.thread_id,
            "stepId": self.step_id,
            "value": self.value,
            "strategy": self.strategy,
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, feedback_dict: FeedbackDict) -> "Feedback":
        id = feedback_dict.get("id", "")
        thread_id = feedback_dict.get("threadId", "")
        step_id = feedback_dict.get("stepId", "")
        value = feedback_dict.get("value", None)
        strategy = feedback_dict.get("strategy", "BINARY")
        comment = feedback_dict.get("comment", None)

        feedback = cls(
            id=id,
            thread_id=thread_id,
            step_id=step_id,
            value=value,
            strategy=strategy,
            comment=comment,
        )

        return feedback


@dataclass
class Attachment:
    thread_id: str
    step_id: str
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[Dict] = Field(default_factory=lambda: {})
    mime: Optional[str] = None
    name: Optional[str] = None
    object_key: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "threadId": self.thread_id,
            "stepId": self.step_id,
            "metadata": self.metadata,
            "mime": self.mime,
            "name": self.name,
            "objectKey": self.object_key,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, attachment_dict: AttachmentDict) -> "Attachment":
        id = attachment_dict.get("id", "")
        thread_id = attachment_dict.get("threadId", "")
        step_id = attachment_dict.get("stepId", "")
        metadata = attachment_dict.get("metadata", {})
        mime = attachment_dict.get("mime", "")
        name = attachment_dict.get("name", "")
        object_key = attachment_dict.get("objectKey", "")
        url = attachment_dict.get("url", "")

        attachment = cls(
            id=id,
            thread_id=thread_id,
            mime=mime,
            name=name,
            object_key=object_key,
            url=url,
            step_id=step_id,
            metadata=metadata,
        )

        return attachment


class UserDict(TypedDict, total=False):
    id: Optional[str]
    metadata: Optional[Dict]
    identifier: Optional[str]
    createdAt: Optional[str]


@dataclass
class User:
    id: Optional[str] = None
    created_at: Optional[str] = None
    identifier: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict = Field(default_factory=lambda: {})

    def to_dict(self) -> UserDict:
        return {
            "id": self.id,
            "identifier": self.identifier,
            "metadata": self.metadata,
            "createdAt": self.created_at,
        }

    @classmethod
    def from_dict(cls, user_dict: Dict) -> "User":
        id = user_dict.get("id", "")
        identifier = user_dict.get("identifier", "")
        metadata = user_dict.get("metadata", {})
        created_at = user_dict.get("createdAt", "")

        user = cls(
            id=id, identifier=identifier, metadata=metadata, created_at=created_at
        )

        return user
