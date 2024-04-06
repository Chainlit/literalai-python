from typing import TYPE_CHECKING, Optional

from literalai.prompt import Prompt

if TYPE_CHECKING:
    from literalai.api import LiteralAPI

from . import gql


def get_prompt_helper(api: "LiteralAPI", name: str, version: Optional[int] = None):
    variables = {"name": name, "version": version}

    def process_response(response):
        prompt = response["data"]["promptVersion"]
        return Prompt.from_dict(api, prompt) if prompt else None

    description = "get prompt"

    return gql.GET_PROMPT, description, variables, process_response
