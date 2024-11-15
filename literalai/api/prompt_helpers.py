from typing import TYPE_CHECKING, Optional, TypedDict, Callable

from literalai.observability.generation import GenerationMessage
from literalai.prompt_engineering.prompt import Prompt, ProviderSettings

if TYPE_CHECKING:
    from literalai.api import LiteralAPI
    from literalai.api import SharedPromptCache

from literalai.api import gql


def create_prompt_lineage_helper(name: str, description: Optional[str] = None):
    variables = {"name": name, "description": description}

    def process_response(response):
        prompt = response["data"]["createPromptLineage"]
        return prompt

    description = "create prompt lineage"

    return gql.CREATE_PROMPT_LINEAGE, description, variables, process_response


def get_prompt_lineage_helper(name: str):
    variables = {"name": name}

    def process_response(response):
        prompt = response["data"]["promptLineage"]
        return prompt

    description = "get prompt lineage"

    return gql.GET_PROMPT_LINEAGE, description, variables, process_response


def create_prompt_helper(
    api: "LiteralAPI",
    lineage_id: str,
    template_messages: list[GenerationMessage],
    settings: Optional[ProviderSettings] = None,
    tools: Optional[list[dict]] = None,
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
    timeout: Optional[int] = None,
    prompt_cache: Optional[SharedPromptCache] = None,
) -> tuple[str, str, dict, Callable]:
    """Helper function for getting prompts with caching logic"""
    if not (id or name):
        raise ValueError("Either the `id` or the `name` must be provided.")

    cached_prompt = None
    if prompt_cache:
        cached_prompt = prompt_cache.get(id, name, version)
        timeout = 1 if cached_prompt else timeout

    variables = {"id": id, "name": name, "version": version}

    def process_response(response):
        prompt = Prompt.from_dict(api, response["data"]["prompt"])
        if prompt_cache:
            prompt_cache.put(prompt)
        return prompt

    description = "get prompt"

    return gql.GET_PROMPT_VERSION, description, variables, process_response, cached_prompt


def create_prompt_variant_helper(
    from_lineage_id: Optional[str] = None,
    template_messages: list[GenerationMessage] = [],
    settings: Optional[ProviderSettings] = None,
    tools: Optional[list[dict]] = None,
):
    variables = {
        "fromLineageId": from_lineage_id,
        "templateMessages": template_messages,
        "settings": settings,
        "tools": tools,
    }

    def process_response(response):
        variant = response["data"]["createPromptExperiment"]
        return variant["id"] if variant else None

    description = "create prompt variant"

    return gql.CREATE_PROMPT_VARIANT, description, variables, process_response


class PromptRollout(TypedDict):
    version: int
    rollout: int


def get_prompt_ab_testing_helper(
    name: Optional[str] = None,
):
    variables = {"lineageName": name}

    def process_response(response) -> list[PromptRollout]:
        response_data = response["data"]["promptLineageRollout"]
        return list(map(lambda x: x["node"], response_data["edges"]))

    description = "get prompt A/B testing"

    return gql.GET_PROMPT_AB_TESTING, description, variables, process_response


def update_prompt_ab_testing_helper(name: str, rollouts: list[PromptRollout]):
    variables = {"name": name, "rollouts": rollouts}

    def process_response(response) -> dict:
        return response["data"]["updatePromptLineageRollout"]

    description = "update prompt A/B testing"

    return gql.UPDATE_PROMPT_AB_TESTING, description, variables, process_response
