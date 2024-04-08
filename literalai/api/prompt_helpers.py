from typing import TYPE_CHECKING, Dict, List, Optional

from literalai.my_types import GenerationMessage
from literalai.prompt import Prompt

if TYPE_CHECKING:
    from literalai.api import LiteralAPI

from . import gql


def create_prompt_lineage_helper(name: str, description: Optional[str] = None):
    variables = {"name": name, "description": description}

    def process_response(response):
        prompt = response["data"]["createPromptLineage"]
        return prompt

    description = "create prompt lineage"

    return gql.CREATE_PROMPT_LINEAGE, description, variables, process_response


def create_prompt_helper(
    api: "LiteralAPI",
    lineage_id: str,
    template_messages: List[GenerationMessage],
    settings: Optional[Dict] = None,
):
    variables = {
        "lineageId": lineage_id,
        "templateMessages": template_messages,
        "settings": settings,
    }

    def process_response(response):
        prompt = response["data"]["createPromptVersion"]
        return Prompt.from_dict(api, prompt) if prompt else None

    description = "create prompt version"

    return gql.CREATE_PROMPT_VERSION, description, variables, process_response


def get_prompt_helper(api: "LiteralAPI", name: str, version: Optional[int] = None):
    variables = {"name": name, "version": version}

    def process_response(response):
        prompt = response["data"]["promptVersion"]
        return Prompt.from_dict(api, prompt) if prompt else None

    description = "get prompt"

    return gql.GET_PROMPT_VERSION, description, variables, process_response
