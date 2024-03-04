import sys
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

import chevron

if TYPE_CHECKING:
    from literalai.api import API

from literalai.my_types import GenerationMessage, GenerationType


class ProviderSettings(TypedDict, total=False):
    provider: str
    model: str
    frequency_penalty: float
    max_tokens: int
    presence_penalty: float
    stop: Optional[List[str]]
    temperature: float
    top_p: float


class PromptVariable(TypedDict, total=False):
    name: str
    language: Literal["json", "plaintext"]


class LiteralMessageDict(dict):
    def __init__(self, prompt_id: str, variables: Dict, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize as a regular dict
        if "uuid" in self:
            uuid = self.pop("uuid")
            self.__literal_prompt__ = {
                "uuid": uuid,
                "prompt_id": prompt_id,
                "variables": variables,
            }


class PromptDict(TypedDict, total=False):
    id: str
    lineage: Dict
    createdAt: str
    updatedAt: str
    type: "GenerationType"
    name: str
    version: int
    versionDesc: Optional[str]
    templateMessages: List["GenerationMessage"]
    tools: Optional[List[Dict]]
    provider: str
    settings: ProviderSettings
    variables: List[PromptVariable]
    variablesDefaultValues: Optional[Dict[str, Any]]


@dataclass
class Prompt:
    api: "API"
    id: str
    created_at: str
    updated_at: str
    type: "GenerationType"
    name: str
    version: int
    version_desc: Optional[str]
    template_messages: List["GenerationMessage"]
    tools: Optional[List[Dict]]
    provider: str
    settings: ProviderSettings
    variables: List[PromptVariable]
    variables_default_values: Optional[Dict[str, Any]]

    def to_dict(self) -> PromptDict:
        # Convert the Prompt instance to a dictionary matching the PromptDict structure
        return {
            "id": self.id,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "type": self.type,
            "name": self.name,
            "version": self.version,
            "versionDesc": self.version_desc,
            "templateMessages": self.template_messages,  # Assuming this is a list of dicts or similar serializable objects
            "tools": self.tools,
            "provider": self.provider,
            "settings": self.settings,
            "variables": self.variables,
            "variablesDefaultValues": self.variables_default_values,
        }

    @classmethod
    def from_dict(cls, api: "API", prompt_dict: PromptDict) -> "Prompt":
        # Create a Prompt instance from a dictionary (PromptDict)
        provider = prompt_dict.get("settings", {}).pop("provider", "")

        return cls(
            api=api,
            id=prompt_dict.get("id", ""),
            name=prompt_dict.get("lineage", {}).get("name", ""),
            version=prompt_dict.get("version", 0),
            created_at=prompt_dict.get("createdAt", ""),
            updated_at=prompt_dict.get("updatedAt", ""),
            type=prompt_dict.get("type", GenerationType.CHAT),
            version_desc=prompt_dict.get("versionDesc"),
            template_messages=prompt_dict.get("templateMessages", []),
            tools=prompt_dict.get("tools", None) or None,
            provider=provider,
            settings=prompt_dict.get("settings", {}),
            variables=prompt_dict.get("variables", []),
            variables_default_values=prompt_dict.get("variablesDefaultValues"),
        )

    def format(
        self, variables: Optional[Dict[str, Any]] = None
    ) -> List[LiteralMessageDict]:
        variables_with_defaults = {
            **(self.variables_default_values or {}),
            **(variables or {}),
        }
        formatted_messages = []

        for message in self.template_messages:
            formatted_message = LiteralMessageDict(
                self.id, variables_with_defaults, message.copy()
            )
            if isinstance(formatted_message["content"], str):
                formatted_message["content"] = chevron.render(
                    message["content"], variables_with_defaults
                )
            else:
                for content in formatted_message["content"]:
                    if content["type"] == "text":
                        content["text"] = chevron.render(
                            content["text"], variables_with_defaults
                        )

            formatted_messages.append(formatted_message)

        return formatted_messages

    def to_langchain_chat_prompt_template(self):
        try:
            version("langchain")
        except Exception:
            raise Exception(
                "Please install langchain to use the langchain callback. "
                "You can install it with `pip install langchain`"
            )

        from langchain_core.messages import (
            AIMessage,
            BaseMessage,
            HumanMessage,
            SystemMessage,
        )
        from langchain_core.prompts import (
            AIMessagePromptTemplate,
            ChatPromptTemplate,
            HumanMessagePromptTemplate,
            SystemMessagePromptTemplate,
        )

        class CustomChatPromptTemplate(ChatPromptTemplate):
            orig_messages: Optional[List[GenerationMessage]]
            default_vars: Optional[Dict] = None
            prompt_id: Optional[str]

            def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
                variables_with_defaults = {
                    **(self.default_vars or {}),
                    **(kwargs or {}),
                }

                rendered_messages: List[BaseMessage] = []

                for index, message in enumerate(self.messages):
                    template = message.prompt.template  # type: ignore
                    content = chevron.render(template, variables_with_defaults)
                    additonal_kwargs = {}
                    if self.orig_messages and index < len(self.orig_messages):
                        additonal_kwargs = {
                            "uuid": self.orig_messages[index].get("uuid")
                            if self.orig_messages
                            else None,
                            "prompt_id": self.prompt_id,
                            "variables": variables_with_defaults,
                        }

                    if isinstance(message, HumanMessagePromptTemplate):
                        rendered_messages.append(
                            HumanMessage(
                                content=content, additional_kwargs=additonal_kwargs
                            )
                        )
                    if isinstance(message, AIMessagePromptTemplate):
                        rendered_messages.append(
                            AIMessage(
                                content=content, additional_kwargs=additonal_kwargs
                            )
                        )
                    if isinstance(message, SystemMessagePromptTemplate):
                        rendered_messages.append(
                            SystemMessage(
                                content=content, additional_kwargs=additonal_kwargs
                            )
                        )

                return rendered_messages

        lc_messages = [(m["role"], m["content"]) for m in self.template_messages]

        chat_template = CustomChatPromptTemplate.from_messages(lc_messages)
        chat_template.default_vars = self.variables_default_values
        chat_template.orig_messages = self.template_messages
        chat_template.prompt_id = self.id

        return chat_template
