from typing import Any, Dict, List, Optional, TypedDict

from literalai.filter import scores_filters, scores_order_by
from literalai.my_types import PaginatedResponse, Score, ScoreType

from . import gql


def get_scores_helper(
    first: Optional[int] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    filters: Optional[scores_filters] = None,
    order_by: Optional[scores_order_by] = None,
):
    variables: Dict[str, Any] = {}

    if first:
        variables["first"] = first
    if after:
        variables["after"] = after
    if before:
        variables["before"] = before
    if filters:
        variables["filters"] = filters
    if order_by:
        variables["orderBy"] = order_by

    def process_response(response):
        response_data = response["data"]["scores"]
        response_data["data"] = list(map(lambda x: x["node"], response_data["edges"]))
        del response_data["edges"]
        return PaginatedResponse[Score].from_dict(response_data, Score)  # type: ignore

    description = "get scores"

    return gql.GET_SCORES, description, variables, process_response


def create_score_helper(
    name: str,
    value: int,
    type: ScoreType,
    step_id: Optional[str] = None,
    generation_id: Optional[str] = None,
    dataset_experiment_item_id: Optional[str] = None,
    comment: Optional[str] = None,
    tags: Optional[List[str]] = None,
):
    variables = {
        "name": name,
        "type": type,
        "value": value,
        "stepId": step_id,
        "generationId": generation_id,
        "datasetExperimentItemId": dataset_experiment_item_id,
        "comment": comment,
        "tags": tags,
    }

    def process_response(response):
        return Score.from_dict(response["data"]["createScore"])

    description = "create score"

    return gql.CREATE_SCORE, description, variables, process_response


class ScoreUpdate(TypedDict, total=False):
    comment: Optional[str]
    value: float


def update_score_helper(
    id: str,
    update_params: ScoreUpdate,
):
    variables = {"id": id, **update_params}

    def process_response(response):
        return Score.from_dict(response["data"]["updateScore"])

    description = "update score"

    return gql.UPDATE_SCORE, description, variables, process_response


def delete_score_helper(id: str):
    variables = {"id": id}

    def process_response(response):
        return response["data"]["deleteScore"]

    description = "delete score"

    return gql.DELETE_SCORE, description, variables, process_response
