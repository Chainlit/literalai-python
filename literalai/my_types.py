import sys
import uuid
from enum import Enum, unique
from typing import Dict, Generic, List, Literal, Optional, Protocol, TypeVar, Union

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from pydantic.dataclasses import Field, dataclass

GenerationMessageRole = Literal["user", "assistant", "tool", "function", "system"]
ScoreType = Literal["HUMAN", "AI"]


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


@unique
class GenerationType(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


class TextContent(TypedDict, total=False):
    type: Literal["text"]
    text: str


class ImageUrlContent(TypedDict, total=False):
    type: Literal["image_url"]
    image_url: str


class GenerationMessage(TypedDict, total=False):
    uuid: Optional[str]
    templated: Optional[bool]
    name: Optional[str]
    role: Optional[GenerationMessageRole]
    content: Optional[Union[str, List[Union[TextContent, ImageUrlContent]]]]
    function_call: Optional[Dict]
    tool_calls: Optional[List[Dict]]
    tool_call_id: Optional[str]


@dataclass
class BaseGeneration:
    prompt_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    settings: Optional[Dict] = Field(default_factory=dict)
    variables: Optional[Dict] = Field(default_factory=dict)
    tags: Optional[List[str]] = Field(default_factory=list)
    tools: Optional[List[Dict]] = None
    token_count: Optional[int] = None
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None
    tt_first_token: Optional[float] = None
    token_throughput_in_s: Optional[float] = None
    duration: Optional[float] = None

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
        return {
            "promptId": self.prompt_id,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
            "settings": self.settings,
            "variables": self.variables,
            "tags": self.tags,
            "tools": self.tools,
            "tokenCount": self.token_count,
            "inputTokenCount": self.input_token_count,
            "outputTokenCount": self.output_token_count,
            "ttFirstToken": self.tt_first_token,
            "tokenThroughputInSeconds": self.token_throughput_in_s,
            "duration": self.duration,
        }


@dataclass
class CompletionGeneration(BaseGeneration):
    prompt: Optional[str] = None
    completion: Optional[str] = None
    type = GenerationType.COMPLETION

    def to_dict(self):
        _dict = super().to_dict()
        _dict.update(
            {
                "prompt": self.prompt,
                "completion": self.completion,
                "type": self.type.value,
            }
        )
        return _dict

    @classmethod
    def from_dict(self, generation_dict: Dict) -> "CompletionGeneration":
        return CompletionGeneration(
            prompt_id=generation_dict.get("promptId"),
            error=generation_dict.get("error"),
            tags=generation_dict.get("tags"),
            provider=generation_dict.get("provider"),
            model=generation_dict.get("model"),
            variables=generation_dict.get("variables"),
            tools=generation_dict.get("tools"),
            settings=generation_dict.get("settings"),
            token_count=generation_dict.get("tokenCount"),
            input_token_count=generation_dict.get("inputTokenCount"),
            output_token_count=generation_dict.get("outputTokenCount"),
            tt_first_token=generation_dict.get("ttFirstToken"),
            token_throughput_in_s=generation_dict.get("tokenThroughputInSeconds"),
            duration=generation_dict.get("duration"),
            prompt=generation_dict.get("prompt"),
            completion=generation_dict.get("completion"),
        )


@dataclass
class ChatGeneration(BaseGeneration):
    type = GenerationType.CHAT
    messages: Optional[List[GenerationMessage]] = Field(default_factory=list)
    message_completion: Optional[GenerationMessage] = None

    def to_dict(self):
        _dict = super().to_dict()
        _dict.update(
            {
                "messages": self.messages,
                "messageCompletion": self.message_completion,
                "type": self.type.value,
            }
        )
        return _dict

    @classmethod
    def from_dict(self, generation_dict: Dict) -> "ChatGeneration":
        return ChatGeneration(
            prompt_id=generation_dict.get("promptId"),
            error=generation_dict.get("error"),
            tags=generation_dict.get("tags"),
            provider=generation_dict.get("provider"),
            model=generation_dict.get("model"),
            variables=generation_dict.get("variables"),
            tools=generation_dict.get("tools"),
            settings=generation_dict.get("settings"),
            token_count=generation_dict.get("tokenCount"),
            input_token_count=generation_dict.get("inputTokenCount"),
            output_token_count=generation_dict.get("outputTokenCount"),
            tt_first_token=generation_dict.get("ttFirstToken"),
            token_throughput_in_s=generation_dict.get("tokenThroughputInSeconds"),
            duration=generation_dict.get("duration"),
            messages=generation_dict.get("messages", []),
            message_completion=generation_dict.get("message_completion"),
        )


class ScoreDict(TypedDict, total=False):
    id: Optional[str]
    name: str
    type: ScoreType
    value: float
    stepId: Optional[str]
    generationId: Optional[str]
    datasetExperimentItemId: Optional[str]
    comment: Optional[str]
    tags: Optional[List[str]]


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
class Score:
    name: str
    type: ScoreType
    value: float
    step_id: Optional[str]
    generation_id: Optional[str]
    dataset_experiment_item_id: Optional[str]
    comment: Optional[str]
    tags: Optional[List[str]]
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "value": self.value,
            "stepId": self.step_id,
            "generationId": self.generation_id,
            "datasetExperimentItemId": self.dataset_experiment_item_id,
            "comment": self.comment,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, score_dict: ScoreDict) -> "Score":
        id = score_dict.get("id", "")
        name = score_dict.get("name", "")
        type = score_dict.get("type", "HUMAN")
        value = score_dict.get("value", 0.0)
        step_id = score_dict.get("stepId", "")
        generation_id = score_dict.get("generationId", "")
        dataset_experiment_item_id = score_dict.get("datasetExperimentItemId", "")
        comment = score_dict.get("comment", "")
        tags = score_dict.get("tags", [])

        score = cls(
            id=id,
            name=name,
            type=type,
            value=value,
            step_id=step_id,
            generation_id=generation_id,
            dataset_experiment_item_id=dataset_experiment_item_id,
            comment=comment,
            tags=tags,
        )

        return score


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
