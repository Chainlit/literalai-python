from typing import TYPE_CHECKING, Dict, List, Optional, TypedDict

from literalai.observability.generation import GenerationMessage
from literalai.prompt_engineering.prompt import Prompt, ProviderSettings

if TYPE_CHECKING:
    from literalai.api import LiteralAPI

from literalai.api import gql


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
    settings: Optional[ProviderSettings] = None,
    tools: Optional[List[Dict]] = None,
):
    variables = {
        "lineageId": lineage_id,
        "templateMessages": template_messages,
        "settings": settings,
        "tools": tools,
    }

    def process_response(response):
        prompt = response["data"]["createPromptVersion"]
        return Prompt.from_dict(api, prompt) if prompt else None

    description = "create prompt version"

    return gql.CREATE_PROMPT_VERSION, description, variables, process_response


def get_prompt_helper(
    api: "LiteralAPI",
    id: Optional[str] = None,
    name: Optional[str] = None,
    version: Optional[int] = 0,
):
    variables = {"id": id, "name": name, "version": version}

    def process_response(response):
        prompt = response["data"]["promptVersion"]
        return Prompt.from_dict(api, prompt) if prompt else None

    description = "get prompt"

    return gql.GET_PROMPT_VERSION, description, variables, process_response


class PromptRollout(TypedDict):
    version: int
    rollout: int


def get_prompt_ab_testing_helper(
    name: Optional[str] = None,
):
    variables = {"lineageName": name}

    def process_response(response) -> List[PromptRollout]:
        response_data = response["data"]["promptLineageRollout"]
        return list(map(lambda x: x["node"], response_data["edges"]))

    description = "get prompt A/B testing"

    return gql.GET_PROMPT_AB_TESTING, description, variables, process_response


def update_prompt_ab_testing_helper(name: str, rollouts: List[PromptRollout]):
    variables = {"name": name, "rollouts": rollouts}

    def process_response(response) -> Dict:
        return response["data"]["updatePromptLineageRollout"]

    description = "update prompt A/B testing"

    return gql.UPDATE_PROMPT_AB_TESTING, description, variables, process_response
