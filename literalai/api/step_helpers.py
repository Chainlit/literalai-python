from typing import Dict, List, Optional, Union

from literalai.step import Step, StepDict, StepType

from . import gql


def create_step_helper(
    thread_id: Optional[str] = None,
    type: Optional[StepType] = "undefined",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    input: Optional[Dict] = None,
    output: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    parent_id: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    variables = {
        "threadId": thread_id,
        "type": type,
        "startTime": start_time,
        "endTime": end_time,
        "input": input,
        "output": output,
        "metadata": metadata,
        "parentId": parent_id,
        "name": name,
        "tags": tags,
    }

    def process_response(response):
        return Step.from_dict(response["data"]["createStep"])

    description = "create step"

    return gql.CREATE_STEP, description, variables, process_response


def update_step_helper(
    id: str,
    type: Optional[StepType] = None,
    input: Optional[str] = None,
    output: Optional[str] = None,
    metadata: Optional[Dict] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    parent_id: Optional[str] = None,
):
    variables = {
        "id": id,
        "type": type,
        "input": input,
        "output": output,
        "metadata": metadata,
        "name": name,
        "tags": tags,
        "startTime": start_time,
        "endTime": end_time,
        "parentId": parent_id,
    }

    def process_response(response):
        return Step.from_dict(response["data"]["updateStep"])

    description = "update step"

    return gql.UPDATE_STEP, description, variables, process_response


def get_step_helper(id: str):
    variables = {"id": id}

    def process_response(response):
        step = response["data"]["step"]
        return Step.from_dict(step) if step else None

    description = "get step"

    return gql.GET_STEP, description, variables, process_response


def delete_step_helper(id: str):
    variables = {"id": id}

    def process_response(response):
        return bool(response["data"]["deleteStep"])

    description = "delete step"

    return gql.DELETE_STEP, description, variables, process_response


def send_steps_helper(steps: List[Union[StepDict, "Step"]]):
    query = gql.steps_query_builder(steps)
    variables = gql.steps_variables_builder(steps)

    description = "send steps"

    def process_response(response: Dict):
        return response

    return query, description, variables, process_response
