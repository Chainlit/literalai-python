import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, TypeVar, Callable

from literalai.dataset import Dataset, DatasetType
from literalai.dataset_item import DatasetItem
from literalai.filter import (
    generations_filters,
    generations_order_by,
    scores_filters,
    scores_order_by,
    threads_filters,
    threads_order_by,
    users_filters,
)
from literalai.prompt import Prompt
from .user_helpers import get_user_helper, get_users_helper, create_user_helper, delete_user_helper, update_user_helper
from .thread_helpers import get_threads_helper, list_threads_helper, get_thread_helper, create_thread_helper, update_thread_helper, upsert_thread_helper, delete_thread_helper
from .score_helpers import get_scores_helper, create_score_helper, update_score_helper, delete_score_helper, ScoreUpdate
from .attachment_helpers import create_attachment_helper, update_attachment_helper, AttachmentUpload, get_attachment_helper, delete_attachment_helper
from .step_helpers import create_step_helper, update_step_helper, get_step_helper, delete_step_helper, send_steps_helper


if TYPE_CHECKING:
    from typing import Tuple  # noqa: F401

import httpx

from literalai.my_types import (
    Attachment,
    BaseGeneration,
    ChatGeneration,
    CompletionGeneration,
    PaginatedResponse,
    Score,
    ScoreType,
    User,
)
from literalai.step import Step, StepDict, StepType
from literalai.thread import Thread

logger = logging.getLogger(__name__)


class BaseLiteralAPI:
    def __init__(self, api_key: Optional[str]=None, url: Optional[str]=None):
        if url and url[-1] == "/":
            url = url[:-1]

        if api_key is None:
            raise Exception("LITERAL_API_KEY not set")
        if url is None:
            raise Exception("LITERAL_API_URL not set")

        self.api_key = api_key
        self.url = url

        self.graphql_endpoint = self.url + "/api/graphql"
        self.rest_endpoint = self.url + "/api"

    @property
    def headers(self):
        from literalai.version import __version__

        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "x-client-name": "py-literal-client",
            "x-client-version": __version__,
        }

class LiteralAPI(BaseLiteralAPI):
    R = TypeVar('R')

    def make_gql_call(
        self, description: str, query: str, variables: Dict[str, Any]
    ) -> Dict:
        def raise_error(error):
            logger.error(f"Failed to {description}: {error}")
            raise Exception(error)

        with httpx.Client() as client:
            response = client.post(
                self.graphql_endpoint,
                json={"query": query, "variables": variables},
                headers=self.headers,
                timeout=10,
            )

            if response.status_code >= 400:
                raise_error(response.text)

            json = response.json()

            if json.get("errors"):
                raise_error(json["errors"])

            if json.get("data"):
                if isinstance(json["data"], dict):
                    for key, value in json["data"].items():
                        if value and value.get("ok") is False:
                            raise_error(
                                f"Failed to {description}: {value.get('message')}"
                            )

            return json
        
        # This should not be reached, exceptions should be thrown beforehands
        # Added because of mypy
        raise Exception("Unknown error")

    def make_rest_call(self, subpath: str, body: Dict[str, Any]) -> Dict:
        with httpx.Client() as client:
            response = client.post(
                self.rest_endpoint + subpath,
                json=body,
                headers=self.headers,
                timeout=20,
            )

            response.raise_for_status()
            json = response.json()

            return json

    def gql_helper(self, query: str, description: str, variables: Dict, process_response: Callable[..., R]) -> R:
        response = self.make_gql_call(description, query, variables)
        return process_response(response)

    # User API

    def get_users(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[users_filters] = None,
    ):
        return self.gql_helper(*get_users_helper(first, after, before, filters))

    def get_user(
        self,
        id: Optional[str] = None, identifier: Optional[str] = None
    ):
        return self.gql_helper(*get_user_helper(id, identifier))


    def create_user(
        self, identifier: str, metadata: Optional[Dict] = None
    ):
        return self.gql_helper(*create_user_helper(identifier, metadata))

    def update_user(
        self, id: str, identifier: Optional[str] = None, metadata: Optional[Dict] = None
    ):
        return self.gql_helper(*update_user_helper(id, identifier, metadata))

    def delete_user(
        self, id: str
    ):
        return self.gql_helper(*delete_user_helper(id))

    def get_or_create_user(
        self, identifier: str, metadata: Optional[Dict] = None
    ):
        user = self.get_user(identifier=identifier)
        if user:
            return user

        return self.create_user(identifier, metadata)

    # Thread API

    def get_threads(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[threads_filters] = None,
        order_by: Optional[threads_order_by] = None,
    ):
        return self.gql_helper(*get_threads_helper(first, after, before, filters,order_by))


    def list_threads(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[threads_filters] = None,
        order_by: Optional[threads_order_by] = None,
    ):
        return self.gql_helper(*list_threads_helper(first, after, before, filters,order_by))

    def get_thread(self, id: str):
        return self.gql_helper(*get_thread_helper(id))


    def create_thread(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return self.gql_helper(*create_thread_helper(name, metadata, participant_id, environment, tags))

    def upsert_thread(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return self.gql_helper(*upsert_thread_helper(id, name, metadata, participant_id, environment, tags))

    def update_thread(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return self.gql_helper(*update_thread_helper(id, name, metadata, participant_id, environment, tags))

    def delete_thread(
        self, id: str
    ):
        return self.gql_helper(*delete_thread_helper(id))

    # Score API

    def get_scores(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[scores_filters] = None,
        order_by: Optional[scores_order_by] = None,
    ):
        return self.gql_helper(*get_scores_helper(first, after, before, filters, order_by))

    def create_score(
        self,
        name: str,
        value: int,
        type: ScoreType,
        step_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        dataset_experiment_item_id: Optional[str] = None,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return self.gql_helper(*create_score_helper(name, value, type, step_id, generation_id, dataset_experiment_item_id, comment, tags))

    def update_score(
        self,
        id: str,
        update_params: ScoreUpdate,
    ):
        return self.gql_helper(*update_score_helper(id, update_params))


    def delete_score(self, id: str):
        return self.gql_helper(*delete_score_helper(id))

    # Attachment API

    def upload_file(
        self,
        content: Union[bytes, str],
        thread_id: str,
        mime: Optional[str] = "application/octet-stream",
    ) -> Dict:
        id = str(uuid.uuid4())
        body = {"fileName": id, "contentType": mime, "threadId": thread_id}

        path = "/api/upload/file"

        with httpx.Client() as client:
            response = client.post(
                f"{self.url}{path}",
                json=body,
                headers=self.headers,
            )
            if response.status_code >= 400:
                reason = response.text
                logger.error(f"Failed to sign upload url: {reason}")
                return {"object_key": None, "url": None}
            json_res = response.json()
        method = "put" if "put" in json_res else "post"
        request_dict: Dict[str, Any] = json_res.get(method, {})
        url: Optional[str] = request_dict.get("url")

        if not url:
            raise Exception("Invalid server response")
        headers: Optional[Dict] = request_dict.get("headers")
        fields: Dict = request_dict.get("fields", {})
        object_key: Optional[str] = fields.get("key")
        upload_type: Literal["raw", "multipart"] = request_dict.get(
            "uploadType", "multipart"
        )
        signed_url: Optional[str] = json_res.get("signedUrl")

        # Prepare form data
        form_data = (
            {}
        )  # type: Dict[str, Union[Tuple[Union[str, None], Any], Tuple[Union[str, None], Any, Any]]]
        for field_name, field_value in fields.items():
            form_data[field_name] = (None, field_value)

        # Add file to the form_data
        # Note: The content_type parameter is not needed here, as the correct MIME type should be set in the 'Content-Type' field from upload_details
        form_data["file"] = (id, content, mime)

        with httpx.Client() as client:
            if upload_type == "raw":
                upload_response = client.request(
                    url=url, headers=headers, method=method, data=content  # type: ignore
                )
            else:
                upload_response = client.request(
                    url=url,
                    headers=headers,
                    method=method,
                    files=form_data,
                )  # type: ignore
            try:
                upload_response.raise_for_status()
                return {"object_key": object_key, "url": signed_url}
            except Exception as e:
                logger.error(f"Failed to upload file: {str(e)}")
                return {"object_key": None, "url": None}

    def create_attachment(
        self,
        thread_id: str,
        step_id: str,
        id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        mime: Optional[str] = None,
        name: Optional[str] = None,
        object_key: Optional[str] = None,
        url: Optional[str] = None,
        content: Optional[Union[bytes, str]] = None,
        path: Optional[str] = None,
    ) -> "Attachment":
        query, description, variables, content, process_response = create_attachment_helper(thread_id=thread_id, step_id=step_id, id=id, metadata=metadata, mime=mime, name=name, object_key=object_key, url=url, content=content, path=path)
        
        if content:
            uploaded = self.upload_file(
                content=content, thread_id=thread_id, mime=mime
            )

            if uploaded["object_key"] is None or uploaded["url"] is None:
                raise Exception("Failed to upload file")

            object_key = uploaded["object_key"]
            if object_key:
                variables["objectKey"] = object_key
            else:
                variables["url"] = uploaded["url"]

        response = self.make_gql_call(description, query, variables)
        return process_response(response)

    def update_attachment(self, id: str, update_params: AttachmentUpload):
        return self.gql_helper(*update_attachment_helper(id, update_params))
    
    def get_attachment(self, id: str):
        return self.gql_helper(*get_attachment_helper(id))

    def delete_attachment(self, id: str):
        return self.gql_helper(*delete_attachment_helper(id))

    # Step API
    
    def create_step(
        self,
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
        return self.gql_helper(*create_step_helper(thread_id=thread_id, type=type, start_time=start_time, end_time=end_time, input=input, output=output, metadata=metadata, parent_id=parent_id, name=name, tags=tags))

    def update_step(
        self,
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
        return self.gql_helper(*update_step_helper(id=id, type=type, input=input, output=output, metadata=metadata, name=name, tags=tags, start_time=start_time, end_time=end_time, parent_id=parent_id))

    def get_step(
        self,
        id: str,
    ):
        return self.gql_helper(*get_step_helper(id=id))

    def delete_step(
        self,
        id: str,
    ):
        return self.gql_helper(*delete_step_helper(id=id))


    def send_steps(
        self,
        steps: List[Union[StepDict, "Step"]]
    ):
        return self.gql_helper(*send_steps_helper(steps=steps))


class AsyncLiteralAPI(BaseLiteralAPI):
    R = TypeVar('R')

    async def make_gql_call(
        self, description: str, query: str, variables: Dict[str, Any]
    ) -> Dict:
        def raise_error(error):
            logger.error(f"Failed to {description}: {error}")
            raise Exception(error)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.graphql_endpoint,
                json={"query": query, "variables": variables},
                headers=self.headers,
                timeout=10,
            )

            if response.status_code >= 400:
                raise_error(response.text)

            json = response.json()

            if json.get("errors"):
                raise_error(json["errors"])

            if json.get("data"):
                if isinstance(json["data"], dict):
                    for key, value in json["data"].items():
                        if value and value.get("ok") is False:
                            raise_error(
                                f"Failed to {description}: {value.get('message')}"
                            )

            return json

        # This should not be reached, exceptions should be thrown beforehands
        # Added because of mypy
        raise Exception("Unkown error")
   
    async def make_rest_call(self, subpath: str, body: Dict[str, Any]) -> Dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.rest_endpoint + subpath,
                json=body,
                headers=self.headers,
                timeout=20,
            )

            response.raise_for_status()
            json = response.json()

            return json
 
    async def gql_helper(self, query: str, description: str, variables: Dict, process_response: Callable[..., R]) -> R:
        response = await self.make_gql_call(description, query, variables)
        return process_response(response)
 
    async def get_users(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[users_filters] = None,
    ):
        return await self.gql_helper(*get_users_helper(first, after, before, filters))

    # User API
    
    async def get_user(
        self,
        id: Optional[str] = None, identifier: Optional[str] = None
    ):
        return await self.gql_helper(*get_user_helper(id, identifier))

    async def create_user(
        self, identifier: str, metadata: Optional[Dict] = None
    ):
        return await self.gql_helper(*create_user_helper(identifier, metadata))


    async def update_user(
        self, id: str, identifier: Optional[str] = None, metadata: Optional[Dict] = None
    ):
        return await self.gql_helper(*update_user_helper(id, identifier, metadata))


    async def delete_user(
        self, id: str
    ):
        return await self.gql_helper(*delete_user_helper(id))


    async def get_or_create_user(
        self, identifier: str, metadata: Optional[Dict] = None
    ):
        user = await self.get_user(identifier=identifier)
        if user:
            return user

        return await self.create_user(identifier, metadata)

    # Thread API

    async def get_threads(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[threads_filters] = None,
        order_by: Optional[threads_order_by] = None,
    ):
        return await self.gql_helper(*get_threads_helper(first, after, before, filters,order_by))


    async def list_threads(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[threads_filters] = None,
        order_by: Optional[threads_order_by] = None,
    ):
        return await self.gql_helper(*list_threads_helper(first, after, before, filters,order_by))

    async def get_thread(self, id: str):
        return await self.gql_helper(*get_thread_helper(id))


    async def create_thread(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return await self.gql_helper(*create_thread_helper(name, metadata, participant_id, environment, tags))

    async def upsert_thread(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return await self.gql_helper(*upsert_thread_helper(id, name, metadata, participant_id, environment, tags))

    async def update_thread(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return await self.gql_helper(*update_thread_helper(id, name, metadata, participant_id, environment, tags))

    async def delete_thread(
        self, id: str
    ):
        return await self.gql_helper(*delete_thread_helper(id))


    # Score API

    async def get_scores(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[scores_filters] = None,
        order_by: Optional[scores_order_by] = None,
    ):
        return await self.gql_helper(*get_scores_helper(first, after, before, filters, order_by))

    async def create_score(
        self,
        name: str,
        value: int,
        type: ScoreType,
        step_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        dataset_experiment_item_id: Optional[str] = None,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return await self.gql_helper(*create_score_helper(name, value, type, step_id, generation_id, dataset_experiment_item_id, comment, tags))

    async def update_score(
        self,
        id: str,
        update_params: ScoreUpdate,
    ):
        return await self.gql_helper(*update_score_helper(id, update_params))


    async def delete_score(self, id: str):
        return await self.gql_helper(*delete_score_helper(id))

    # Attachment API

    async def upload_file(
        self,
        content: Union[bytes, str],
        thread_id: str,
        mime: Optional[str] = "application/octet-stream",
    ) -> Dict:
        id = str(uuid.uuid4())
        body = {"fileName": id, "contentType": mime, "threadId": thread_id}

        path = "/api/upload/file"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.url}{path}",
                json=body,
                headers=self.headers,
            )
            if response.status_code >= 400:
                reason = response.text
                logger.error(f"Failed to sign upload url: {reason}")
                return {"object_key": None, "url": None}
            json_res = response.json()
        method = "put" if "put" in json_res else "post"
        request_dict: Dict[str, Any] = json_res.get(method, {})
        url: Optional[str] = request_dict.get("url")

        if not url:
            raise Exception("Invalid server response")
        headers: Optional[Dict] = request_dict.get("headers")
        fields: Dict = request_dict.get("fields", {})
        object_key: Optional[str] = fields.get("key")
        upload_type: Literal["raw", "multipart"] = request_dict.get(
            "uploadType", "multipart"
        )
        signed_url: Optional[str] = json_res.get("signedUrl")

        # Prepare form data
        form_data = (
            {}
        )  # type: Dict[str, Union[Tuple[Union[str, None], Any], Tuple[Union[str, None], Any, Any]]]
        for field_name, field_value in fields.items():
            form_data[field_name] = (None, field_value)

        # Add file to the form_data
        # Note: The content_type parameter is not needed here, as the correct MIME type should be set in the 'Content-Type' field from upload_details
        form_data["file"] = (id, content, mime)

        async with httpx.AsyncClient() as client:
            if upload_type == "raw":
                upload_response = await client.request(
                    url=url, headers=headers, method=method, data=content  # type: ignore
                )
            else:
                upload_response = await client.request(
                    url=url,
                    headers=headers,
                    method=method,
                    files=form_data,
                )  # type: ignore
            try:
                upload_response.raise_for_status()
                return {"object_key": object_key, "url": signed_url}
            except Exception as e:
                logger.error(f"Failed to upload file: {str(e)}")
                return {"object_key": None, "url": None}

    async def create_attachment(
        self,
        thread_id: str,
        step_id: str,
        id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        mime: Optional[str] = None,
        name: Optional[str] = None,
        object_key: Optional[str] = None,
        url: Optional[str] = None,
        content: Optional[Union[bytes, str]] = None,
        path: Optional[str] = None,
    ) -> "Attachment":
        query, description, variables, content, process_response = create_attachment_helper(thread_id=thread_id, step_id=step_id, id=id, metadata=metadata, mime=mime, name=name, object_key=object_key, url=url, content=content, path=path)
        
        if content:
            uploaded = await self.upload_file(
                content=content, thread_id=thread_id, mime=mime
            )

            if uploaded["object_key"] is None or uploaded["url"] is None:
                raise Exception("Failed to upload file")

            object_key = uploaded["object_key"]
            if object_key:
                variables["objectKey"] = object_key
            else:
                variables["url"] = uploaded["url"]

        response = await self.make_gql_call(description, query, variables)
        return process_response(response)

    async def update_attachment(self, id: str, update_params: AttachmentUpload):
        return await self.gql_helper(*update_attachment_helper(id, update_params))
    
    async def get_attachment(self, id: str):
        return await self.gql_helper(*get_attachment_helper(id))

    async def delete_attachment(self, id: str):
        return await self.gql_helper(*delete_attachment_helper(id))


    # Step API

    async def create_step(
        self,
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
        return await self.gql_helper(*create_step_helper(thread_id=thread_id, type=type, start_time=start_time, end_time=end_time, input=input, output=output, metadata=metadata, parent_id=parent_id, name=name, tags=tags))

    async def update_step(
        self,
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
        return await self.gql_helper(*update_step_helper(id=id, type=type, input=input, output=output, metadata=metadata, name=name, tags=tags, start_time=start_time, end_time=end_time, parent_id=parent_id))

    async def get_step(
        self,
        id: str,
    ):
        return await self.gql_helper(*get_step_helper(id=id))

    async def delete_step(
        self,
        id: str,
    ):
        return await self.gql_helper(*delete_step_helper(id=id))


    async def send_steps(
        self,
        steps: List[Union[StepDict, "Step"]]
    ):
        return await self.gql_helper(*send_steps_helper(steps=steps))


#     # Generation API

#     async def get_generations(
#         self,
#         first: Optional[int] = None,
#         after: Optional[str] = None,
#         before: Optional[str] = None,
#         filters: Optional[generations_filters] = None,
#         order_by: Optional[generations_order_by] = None,
#     ) -> PaginatedResponse:
#         query = """
#         query GetGenerations(
#             $after: ID,
#             $before: ID,
#             $cursorAnchor: DateTime,
#             $filters: [generationsInputType!],
#             $orderBy: GenerationsOrderByInput,
#             $first: Int,
#             $last: Int,
#             $projectId: String,
#             ) {
#             generations(
#                 after: $after,
#                 before: $before,
#                 cursorAnchor: $cursorAnchor,
#                 filters: $filters,
#                 orderBy: $orderBy,
#                 first: $first,
#                 last: $last,
#                 projectId: $projectId,
#                 ) {
#                 pageInfo {
#                     startCursor
#                     endCursor
#                     hasNextPage
#                     hasPreviousPage
#                 }
#                 totalCount
#                 edges {
#                     cursor
#                     node {
#                         id
#                         projectId
#                         prompt
#                         completion
#                         createdAt
#                         provider
#                         model
#                         variables
#                         messages
#                         messageCompletion
#                         tools
#                         settings
#                         stepId
#                         tokenCount
#                         duration
#                         inputTokenCount
#                         outputTokenCount
#                         ttFirstToken
#                         duration
#                         tokenThroughputInSeconds
#                         error
#                         type
#                         tags
#                         step {
#                             threadId
#                             thread {
#                             participant {
#                                 identifier
#                                     }
#                                 }
#                             }
#                         }
#                     }
#                 }
#             }
#     """
#         variables: Dict[str, Any] = {}

#         if first:
#             variables["first"] = first
#         if after:
#             variables["after"] = after
#         if before:
#             variables["before"] = before
#         if filters:
#             variables["filters"] = filters
#         if order_by:
#             variables["orderBy"] = order_by

#         result = await self.make_api_call("get generations", query, variables)

#         response = result["data"]["generations"]

#         response["data"] = list(map(lambda x: x["node"], response["edges"]))
#         del response["edges"]

#         return PaginatedResponse[ChatGeneration].from_dict(response, ChatGeneration)

#     async def create_generation(
#         self, generation: Union[ChatGeneration, CompletionGeneration]
#     ):
#         mutation = """
#         mutation CreateGeneration($generation: GenerationPayloadInput!) {
#             createGeneration(generation: $generation) {
#                 id,
#                 type
#             }
#         }
#         """

#         variables = {
#             "generation": generation.to_dict(),
#         }

#         result = await self.make_api_call("create generation", mutation, variables)

#         return BaseGeneration.from_dict(result["data"]["createGeneration"])

#     def create_generation_sync(
#         self, generation: Union[ChatGeneration, CompletionGeneration]
#     ):
#         mutation = """
#         mutation CreateGeneration($generation: GenerationPayloadInput!) {
#             createGeneration(generation: $generation) {
#                 id,
#                 type
#             }
#         }
#         """

#         variables = {
#             "generation": generation.to_dict(),
#         }

#         result = self.make_api_call_sync("create generation", mutation, variables)

#         return BaseGeneration.from_dict(result["data"]["createGeneration"])

#     # Upload API

#     async def upload_file(
#         self,
#         content: Union[bytes, str],
#         thread_id: str,
#         mime: Optional[str] = "application/octet-stream",
#     ) -> Dict:
#         id = str(uuid.uuid4())
#         body = {"fileName": id, "contentType": mime, "threadId": thread_id}

#         path = "/api/upload/file"

#         async with httpx.AsyncClient() as client:
#             response = await client.post(
#                 f"{self.url}{path}",
#                 json=body,
#                 headers=self.headers,
#             )
#             if response.status_code >= 400:
#                 reason = response.text
#                 logger.error(f"Failed to sign upload url: {reason}")
#                 return {"object_key": None, "url": None}
#             json_res = response.json()
#         method = "put" if "put" in json_res else "post"
#         request_dict: Dict[str, Any] = json_res.get(method, {})
#         url: Optional[str] = request_dict.get("url")

#         if not url:
#             raise Exception("Invalid server response")
#         headers: Optional[Dict] = request_dict.get("headers")
#         fields: Dict = request_dict.get("fields", {})
#         object_key: Optional[str] = fields.get("key")
#         upload_type: Literal["raw", "multipart"] = request_dict.get(
#             "uploadType", "multipart"
#         )
#         signed_url: Optional[str] = json_res.get("signedUrl")

#         # Prepare form data
#         form_data = (
#             {}
#         )  # type: Dict[str, Union[Tuple[Union[str, None], Any], Tuple[Union[str, None], Any, Any]]]
#         for field_name, field_value in fields.items():
#             form_data[field_name] = (None, field_value)

#         # Add file to the form_data
#         # Note: The content_type parameter is not needed here, as the correct MIME type should be set in the 'Content-Type' field from upload_details
#         form_data["file"] = (id, content, mime)

#         async with httpx.AsyncClient() as client:
#             if upload_type == "raw":
#                 upload_response = await client.request(
#                     url=url, headers=headers, method=method, data=content  # type: ignore
#                 )
#             else:
#                 upload_response = await client.request(
#                     url=url,
#                     headers=headers,
#                     method=method,
#                     files=form_data,
#                 )  # type: ignore
#             try:
#                 upload_response.raise_for_status()
#                 return {"object_key": object_key, "url": signed_url}
#             except Exception as e:
#                 logger.error(f"Failed to upload file: {str(e)}")
#                 return {"object_key": None, "url": None}

#     # Dataset API

#     async def create_dataset(
#         self,
#         name: str,
#         description: Optional[str] = None,
#         metadata: Optional[Dict] = None,
#         type: DatasetType = "key_value",
#     ) -> Dataset:
#         query = """
#             mutation createDataset(
#                 $name: String!
#                 $description: String
#                 $metadata: Json
#                 $type: DatasetType
#             ) {
#                 createDataset(
#                     name: $name
#                     description: $description
#                     metadata: $metadata
#                     type: $type
#                 ) {
#                     id
#                     createdAt
#                     name
#                     description
#                     metadata
#                     type
#                 }
#             }
#         """
#         variables = {
#             "name": name,
#             "description": description,
#             "metadata": metadata,
#             "type": type,
#         }
#         result = await self.make_api_call("create dataset", query, variables)

#         return Dataset.from_dict(self, result["data"]["createDataset"])

#     def create_dataset_sync(
#         self,
#         name: str,
#         description: Optional[str] = None,
#         metadata: Optional[Dict] = None,
#         type: DatasetType = "key_value",
#     ) -> Dataset:
#         query = """
#             mutation createDataset(
#                 $name: String!
#                 $description: String
#                 $metadata: Json
#                 $type: DatasetType
#             ) {
#                 createDataset(
#                     name: $name
#                     description: $description
#                     metadata: $metadata
#                     type: $type
#                 ) {
#                     id
#                     createdAt
#                     name
#                     description
#                     metadata
#                     type
#                 }
#             }
#         """
#         variables = {
#             "name": name,
#             "description": description,
#             "metadata": metadata,
#             "type": type,
#         }
#         result = self.make_api_call_sync("create dataset", query, variables)

#         return Dataset.from_dict(self, result["data"]["createDataset"])

#     async def get_dataset(self, id: str) -> Optional[Dataset]:
#         result = await self.make_rest_api_call(
#             subpath="/export/dataset", body={"id": id}
#         )

#         dataset_dict = result.get("data")

#         if dataset_dict is None:
#             return None

#         return Dataset.from_dict(self, dataset_dict)

#     def get_dataset_sync(self, id: str) -> Optional[Dataset]:
#         result = self.make_rest_api_call_sync(
#             subpath="/export/dataset", body={"id": id}
#         )

#         dataset_dict = result.get("data")

#         if dataset_dict is None:
#             return None

#         return Dataset.from_dict(self, dataset_dict)

#     async def update_dataset(
#         self,
#         id: str,
#         name: Optional[str] = None,
#         description: Optional[str] = None,
#         metadata: Optional[Dict] = None,
#     ) -> Dataset:
#         query = """
#             mutation UpdateDataset(
#                 $id: String!
#                 $name: String
#                 $description: String
#                 $metadata: Json
#             ) {
#                 updateDataset(
#                     id: $id
#                     name: $name
#                     description: $description
#                     metadata: $metadata
#                 ) {
#                     id
#                     createdAt
#                     name
#                     description
#                     metadata
#                     type
#                 }
#             }
#         """
#         variables: Dict = {
#             "id": id,
#         }
#         if name is not None:
#             variables["name"] = name
#         if description is not None:
#             variables["description"] = description
#         if metadata is not None:
#             variables["metadata"] = metadata

#         result = await self.make_api_call("update dataset", query, variables)

#         return Dataset.from_dict(self, result["data"]["updateDataset"])

#     def update_dataset_sync(
#         self,
#         id: str,
#         name: Optional[str] = None,
#         description: Optional[str] = None,
#         metadata: Optional[Dict] = None,
#     ) -> Dataset:
#         query = """
#             mutation UpdateDataset(
#                 $id: String!
#                 $name: String
#                 $description: String
#                 $metadata: Json
#             ) {
#                 updateDataset(
#                     id: $id
#                     name: $name
#                     description: $description
#                     metadata: $metadata
#                 ) {
#                     id
#                     createdAt
#                     name
#                     description
#                     metadata
#                     type
#                 }
#             }
#         """
#         variables: Dict = {
#             "id": id,
#         }
#         if name is not None:
#             variables["name"] = name
#         if description is not None:
#             variables["description"] = description
#         if metadata is not None:
#             variables["metadata"] = metadata

#         result = self.make_api_call_sync("update dataset", query, variables)

#         return Dataset.from_dict(self, result["data"]["updateDataset"])

#     async def delete_dataset(self, id: str):
#         query = """
#             mutation DeleteDataset(
#                 $id: String!
#             ) {
#                 deleteDataset(
#                     id: $id
#                 ) {
#                     id
#                     createdAt
#                     name
#                     description
#                     metadata
#                     type
#                 }
#             }
#         """
#         variables = {"id": id}
#         result = await self.make_api_call("delete dataset", query, variables)

#         return Dataset.from_dict(self, result["data"]["deleteDataset"])

#     def delete_dataset_sync(self, id: str):
#         query = """
#             mutation DeleteDataset(
#                 $id: String!
#             ) {
#                 deleteDataset(
#                     id: $id
#                 ) {
#                     id
#                     createdAt
#                     name
#                     description
#                     metadata
#                     type
#                 }
#             }
#         """
#         variables = {"id": id}
#         result = self.make_api_call_sync("delete dataset", query, variables)

#         return Dataset.from_dict(self, result["data"]["deleteDataset"])

#     # DatasetItem API

#     async def create_dataset_item(
#         self,
#         dataset_id: str,
#         input: Dict,
#         expected_output: Optional[Dict] = None,
#         metadata: Optional[Dict] = None,
#     ) -> DatasetItem:
#         query = """
#             mutation CreateDatasetItem(
#                 $datasetId: String!
#                 $input: Json!
#                 $expectedOutput: Json
#                 $metadata: Json
#             ) {
#                 createDatasetItem(
#                     datasetId: $datasetId
#                     input: $input
#                     expectedOutput: $expectedOutput
#                     metadata: $metadata
#                 ) {
#                     id
#                     createdAt
#                     datasetId
#                     metadata
#                     input
#                     expectedOutput
#                     intermediarySteps
#                 }
#             }
#         """
#         variables = {
#             "datasetId": dataset_id,
#             "input": input,
#             "expectedOutput": expected_output,
#             "metadata": metadata,
#         }
#         result = await self.make_api_call("create dataset item", query, variables)

#         return DatasetItem.from_dict(result["data"]["createDatasetItem"])

#     def create_dataset_item_sync(
#         self,
#         dataset_id: str,
#         input: Dict,
#         expected_output: Optional[Dict] = None,
#         metadata: Optional[Dict] = None,
#     ) -> DatasetItem:
#         query = """
#             mutation CreateDatasetItem(
#                 $datasetId: String!
#                 $input: Json!
#                 $expectedOutput: Json
#                 $metadata: Json
#             ) {
#                 createDatasetItem(
#                     datasetId: $datasetId
#                     input: $input
#                     expectedOutput: $expectedOutput
#                     metadata: $metadata
#                 ) {
#                     id
#                     createdAt
#                     datasetId
#                     metadata
#                     input
#                     expectedOutput
#                     intermediarySteps
#                 }
#             }
#         """
#         variables = {
#             "datasetId": dataset_id,
#             "input": input,
#             "expectedOutput": expected_output,
#             "metadata": metadata,
#         }
#         result = self.make_api_call_sync("create dataset item", query, variables)

#         return DatasetItem.from_dict(result["data"]["createDatasetItem"])

#     async def get_dataset_item(self, id: str) -> DatasetItem:
#         query = """
#             query GetDataItem($id: String!) {
#                 datasetItem(id: $id) {
#                     id
#                     createdAt
#                     datasetId
#                     metadata
#                     input
#                     expectedOutput
#                     intermediarySteps
#                 }
#             }
#         """
#         variables = {"id": id}
#         result = await self.make_api_call("get dataset item", query, variables)

#         return DatasetItem.from_dict(result["data"]["datasetItem"])

#     def get_dataset_item_sync(self, id: str) -> DatasetItem:
#         query = """
#             query GetDataItem($id: String!) {
#                 datasetItem(id: $id) {
#                     id
#                     createdAt
#                     datasetId
#                     metadata
#                     input
#                     expectedOutput
#                     intermediarySteps
#                 }
#             }
#         """
#         variables = {"id": id}
#         result = self.make_api_call_sync("get dataset item", query, variables)

#         return DatasetItem.from_dict(result["data"]["datasetItem"])

#     async def delete_dataset_item(self, id: str) -> DatasetItem:
#         query = """
#             mutation DeleteDatasetItem($id: String!) {
#                 deleteDatasetItem(id: $id) {
#                     id
#                     createdAt
#                     datasetId
#                     metadata
#                     input
#                     expectedOutput
#                     intermediarySteps
#                 }
#             }
#         """
#         variables = {"id": id}
#         result = await self.make_api_call("delete dataset item", query, variables)

#         return DatasetItem.from_dict(result["data"]["deleteDatasetItem"])

#     def delete_dataset_item_sync(self, id: str) -> DatasetItem:
#         query = """
#             mutation DeleteDatasetItem($id: String!) {
#                 deleteDatasetItem(id: $id) {
#                     id
#                     createdAt
#                     datasetId
#                     metadata
#                     input
#                     expectedOutput
#                     intermediarySteps
#                 }
#             }
#         """
#         variables = {"id": id}
#         result = self.make_api_call_sync("delete dataset item", query, variables)

#         return DatasetItem.from_dict(result["data"]["deleteDatasetItem"])

#     async def add_step_to_dataset(
#         self, dataset_id: str, step_id: str, metadata: Optional[Dict] = None
#     ) -> DatasetItem:
#         query = """
#             mutation AddStepToDataset(
#                 $datasetId: String!
#                 $stepId: String!
#                 $metadata: Json
#             ) {
#                 addStepToDataset(
#                     datasetId: $datasetId
#                     stepId: $stepId
#                     metadata: $metadata
#                 ) {
#                     id
#                     createdAt
#                     datasetId
#                     metadata
#                     input
#                     expectedOutput
#                     intermediarySteps
#                 }
#             }
#         """
#         variables = {
#             "datasetId": dataset_id,
#             "stepId": step_id,
#             "metadata": metadata,
#         }
#         result = await self.make_api_call("add step to dataset", query, variables)

#         return DatasetItem.from_dict(result["data"]["addStepToDataset"])

#     def add_step_to_dataset_sync(
#         self,
#         dataset_id: str,
#         step_id: str,
#         metadata: Optional[Dict] = None,
#     ) -> DatasetItem:
#         query = """
#             mutation AddStepToDataset(
#                 $datasetId: String!
#                 $stepId: String!
#                 $metadata: Json
#             ) {
#                 addStepToDataset(
#                     datasetId: $datasetId
#                     stepId: $stepId
#                     metadata: $metadata
#                 ) {
#                     id
#                     createdAt
#                     datasetId
#                     metadata
#                     input
#                     expectedOutput
#                     intermediarySteps
#                 }
#             }
#         """
#         variables = {
#             "datasetId": dataset_id,
#             "stepId": step_id,
#             "metadata": metadata,
#         }
#         result = self.make_api_call_sync("add step to dataset", query, variables)

#         return DatasetItem.from_dict(result["data"]["addStepToDataset"])

#     async def add_generation_to_dataset(
#         self, dataset_id: str, generation_id: str, metadata: Optional[Dict] = None
#     ) -> DatasetItem:
#         query = """
#             mutation AddGenerationToDataset(
#                 $datasetId: String!
#                 $generationId: String!
#                 $metadata: Json
#             ) {
#                 addGenerationToDataset(
#                     datasetId: $datasetId
#                     generationId: $generationId
#                     metadata: $metadata
#                 ) {
#                     id
#                     createdAt
#                     datasetId
#                     metadata
#                     input
#                     expectedOutput
#                     intermediarySteps
#                 }
#             }
#         """
#         variables = {
#             "datasetId": dataset_id,
#             "generationId": generation_id,
#             "metadata": metadata,
#         }
#         result = await self.make_api_call("add generation to dataset", query, variables)

#         return DatasetItem.from_dict(result["data"]["addGenerationToDataset"])

#     def add_generation_to_dataset_sync(
#         self,
#         dataset_id: str,
#         generation_id: str,
#         metadata: Optional[Dict] = None,
#     ) -> DatasetItem:
#         query = """
#             mutation AddGenerationToDataset(
#                 $datasetId: String!
#                 $generationId: String!
#                 $metadata: Json
#             ) {
#                 addGenerationToDataset(
#                     datasetId: $datasetId
#                     generationId: $generationId
#                     metadata: $metadata
#                 ) {
#                     id
#                     createdAt
#                     datasetId
#                     metadata
#                     input
#                     expectedOutput
#                     intermediarySteps
#                 }
#             }
#         """
#         variables = {
#             "datasetId": dataset_id,
#             "generationId": generation_id,
#             "metadata": metadata,
#         }
#         result = self.make_api_call_sync("add generation to dataset", query, variables)

#         return DatasetItem.from_dict(result["data"]["addGenerationToDataset"])

#     # Prompt API

#     async def get_prompt(
#         self, name: str, version: Optional[int] = None
#     ) -> Optional[Prompt]:
#         query = """
#             query GetPrompt($name: String!, $version: Int) {
#                 promptVersion(name: $name, version: $version) {
#                     id
#                     createdAt
#                     updatedAt
#                     type
#                     templateMessages
#                     tools
#                     settings
#                     variables
#                     variablesDefaultValues
#                     version
#                     lineage {
#                         name
#                     }
#                 }
#             }
#         """
#         variables = {"name": name, "version": version}
#         result = await self.make_api_call("get prompt", query, variables)

#         prompt = result["data"]["promptVersion"]

#         return Prompt.from_dict(self, prompt) if prompt else None

#     def get_prompt_sync(
#         self, name: str, version: Optional[int] = None
#     ) -> Optional[Prompt]:
#         query = """
#             query GetPrompt($name: String!, $version: Int) {
#                 promptVersion(name: $name, version: $version) {
#                     id
#                     createdAt
#                     updatedAt
#                     type
#                     templateMessages
#                     tools
#                     settings
#                     variables
#                     variablesDefaultValues
#                     version
#                     lineage {
#                         name
#                     }
#                 }
#             }
#         """
#         variables = {"name": name, "version": version}
#         result = self.make_api_call_sync("get prompt", query, variables)

#         prompt = result["data"]["promptVersion"]

#         return Prompt.from_dict(self, prompt) if prompt else None
