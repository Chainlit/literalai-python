from enum import unique, Enum
from typing import Optional, Union, List, Dict, Literal

from pydantic import Field
from pydantic.dataclasses import dataclass

from typing_extensions import TypedDict

from literalai.my_types import TextContent, ImageUrlContent, Utils


GenerationMessageRole = Literal["user", "assistant", "tool", "function", "system"]


@unique
class GenerationType(Enum):
    CHAT = "CHAT"
    COMPLETION = "COMPLETION"


class GenerationMessage(TypedDict, total=False):
    uuid: Optional[str]
    templated: Optional[bool]
    name: Optional[str]
    role: Optional[GenerationMessageRole]
    content: Optional[Union[str, List[Union[TextContent, ImageUrlContent]]]]
    function_call: Optional[Dict]
    tool_calls: Optional[List[Dict]]
    tool_call_id: Optional[str]


@dataclass(repr=False)
class BaseGeneration(Utils):
    id: Optional[str] = None
    prompt_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    settings: Optional[Dict] = Field(default_factory=dict)
    variables: Optional[Dict] = Field(default_factory=dict)
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict] = Field(default_factory=dict)
    tools: Optional[List[Dict]] = None
    token_count: Optional[int] = None
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None
    tt_first_token: Optional[float] = None
    token_throughput_in_s: Optional[float] = None
    duration: Optional[float] = None

    @classmethod
    def from_dict(
        cls, generation_dict: Dict
    ) -> Union["ChatGeneration", "CompletionGeneration"]:
        type = GenerationType(generation_dict.get("type"))
        if type == GenerationType.CHAT:
            return ChatGeneration.from_dict(generation_dict)
        elif type == GenerationType.COMPLETION:
            return CompletionGeneration.from_dict(generation_dict)
        else:
            raise ValueError(f"Unknown generation type: {type}")

    def to_dict(self):
        _dict = {
            "promptId": self.prompt_id,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
            "settings": self.settings,
            "variables": self.variables,
            "tags": self.tags,
            "metadata": self.metadata,
            "tools": self.tools,
            "tokenCount": self.token_count,
            "inputTokenCount": self.input_token_count,
            "outputTokenCount": self.output_token_count,
            "ttFirstToken": self.tt_first_token,
            "tokenThroughputInSeconds": self.token_throughput_in_s,
            "duration": self.duration,
        }
        if self.id:
            _dict["id"] = self.id
        return _dict


@dataclass(repr=False)
class CompletionGeneration(BaseGeneration, Utils):
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
    def from_dict(cls, generation_dict: Dict):
        return CompletionGeneration(
            id=generation_dict.get("id"),
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


@dataclass(repr=False)
class ChatGeneration(BaseGeneration, Utils):
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
    def from_dict(self, generation_dict: Dict):
        return ChatGeneration(
            id=generation_dict.get("id"),
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
            message_completion=generation_dict.get("messageCompletion"),
        )

