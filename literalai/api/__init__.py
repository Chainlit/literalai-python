import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

from literalai.dataset import DatasetType
from literalai.filter import (
    generations_filters,
    generations_order_by,
    scores_filters,
    scores_order_by,
    threads_filters,
    threads_order_by,
    users_filters,
)

from .attachment_helpers import (
    AttachmentUpload,
    create_attachment_helper,
    delete_attachment_helper,
    get_attachment_helper,
    update_attachment_helper,
)
from .dataset_helpers import (
    add_generation_to_dataset_helper,
    add_step_to_dataset_helper,
    create_dataset_helper,
    create_dataset_item_helper,
    delete_dataset_helper,
    delete_dataset_item_helper,
    get_dataset_helper,
    get_dataset_item_helper,
    update_dataset_helper,
)
from .generation_helpers import create_generation_helper, get_generations_helper
from .prompt_helpers import (
    create_prompt_helper,
    create_prompt_lineage_helper,
    get_prompt_helper,
)
from .score_helpers import (
    ScoreUpdate,
    create_score_helper,
    delete_score_helper,
    get_scores_helper,
    update_score_helper,
)
from .step_helpers import (
    create_step_helper,
    delete_step_helper,
    get_step_helper,
    send_steps_helper,
    update_step_helper,
)
from .thread_helpers import (
    create_thread_helper,
    delete_thread_helper,
    get_thread_helper,
    get_threads_helper,
    list_threads_helper,
    update_thread_helper,
    upsert_thread_helper,
)
from .user_helpers import (
    create_user_helper,
    delete_user_helper,
    get_user_helper,
    get_users_helper,
    update_user_helper,
)

if TYPE_CHECKING:
    from typing import Tuple  # noqa: F401

import httpx

from literalai.my_types import (
    Attachment,
    ChatGeneration,
    CompletionGeneration,
    GenerationMessage,
    ScoreType,
)
from literalai.step import Step, StepDict, StepType

logger = logging.getLogger(__name__)


class BaseLiteralAPI:
    def __init__(self, api_key: Optional[str] = None, url: Optional[str] = None):
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
    R = TypeVar("R")

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

    def gql_helper(
        self,
        query: str,
        description: str,
        variables: Dict,
        process_response: Callable[..., R],
    ) -> R:
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

    def get_user(self, id: Optional[str] = None, identifier: Optional[str] = None):
        return self.gql_helper(*get_user_helper(id, identifier))

    def create_user(self, identifier: str, metadata: Optional[Dict] = None):
        return self.gql_helper(*create_user_helper(identifier, metadata))

    def update_user(
        self, id: str, identifier: Optional[str] = None, metadata: Optional[Dict] = None
    ):
        return self.gql_helper(*update_user_helper(id, identifier, metadata))

    def delete_user(self, id: str):
        return self.gql_helper(*delete_user_helper(id))

    def get_or_create_user(self, identifier: str, metadata: Optional[Dict] = None):
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
        return self.gql_helper(
            *get_threads_helper(first, after, before, filters, order_by)
        )

    def list_threads(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[threads_filters] = None,
        order_by: Optional[threads_order_by] = None,
    ):
        return self.gql_helper(
            *list_threads_helper(first, after, before, filters, order_by)
        )

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
        return self.gql_helper(
            *create_thread_helper(name, metadata, participant_id, environment, tags)
        )

    def upsert_thread(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return self.gql_helper(
            *upsert_thread_helper(id, name, metadata, participant_id, environment, tags)
        )

    def update_thread(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return self.gql_helper(
            *update_thread_helper(id, name, metadata, participant_id, environment, tags)
        )

    def delete_thread(self, id: str):
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
        return self.gql_helper(
            *get_scores_helper(first, after, before, filters, order_by)
        )

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
        return self.gql_helper(
            *create_score_helper(
                name,
                value,
                type,
                step_id,
                generation_id,
                dataset_experiment_item_id,
                comment,
                tags,
            )
        )

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
        (
            query,
            description,
            variables,
            content,
            process_response,
        ) = create_attachment_helper(
            thread_id=thread_id,
            step_id=step_id,
            id=id,
            metadata=metadata,
            mime=mime,
            name=name,
            object_key=object_key,
            url=url,
            content=content,
            path=path,
        )

        if content:
            uploaded = self.upload_file(content=content, thread_id=thread_id, mime=mime)

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
        return self.gql_helper(
            *create_step_helper(
                thread_id=thread_id,
                type=type,
                start_time=start_time,
                end_time=end_time,
                input=input,
                output=output,
                metadata=metadata,
                parent_id=parent_id,
                name=name,
                tags=tags,
            )
        )

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
        return self.gql_helper(
            *update_step_helper(
                id=id,
                type=type,
                input=input,
                output=output,
                metadata=metadata,
                name=name,
                tags=tags,
                start_time=start_time,
                end_time=end_time,
                parent_id=parent_id,
            )
        )

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

    def send_steps(self, steps: List[Union[StepDict, "Step"]]):
        return self.gql_helper(*send_steps_helper(steps=steps))

    # Generation API

    def get_generations(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[generations_filters] = None,
        order_by: Optional[generations_order_by] = None,
    ):
        return self.gql_helper(
            *get_generations_helper(first, after, before, filters, order_by)
        )

    def create_generation(
        self, generation: Union[ChatGeneration, CompletionGeneration]
    ):
        return self.gql_helper(*create_generation_helper(generation))

    # Dataset API

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        type: DatasetType = "key_value",
    ):
        return self.gql_helper(
            *create_dataset_helper(self, name, description, metadata, type)
        )

    def get_dataset(self, id: str):
        subpath, _, variables, process_response = get_dataset_helper(self, id)
        response = self.make_rest_call(subpath, variables)
        return process_response(response)

    def update_dataset(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        return self.gql_helper(
            *update_dataset_helper(self, id, name, description, metadata)
        )

    def delete_dataset(self, id: str):
        return self.gql_helper(*delete_dataset_helper(self, id))

    # Dataset Item API

    def create_dataset_item(
        self,
        dataset_id: str,
        input: Dict,
        expected_output: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        return self.gql_helper(
            *create_dataset_item_helper(dataset_id, input, expected_output, metadata)
        )

    def get_dataset_item(self, id: str):
        return self.gql_helper(*get_dataset_item_helper(id))

    def delete_dataset_item(self, id: str):
        return self.gql_helper(*delete_dataset_item_helper(id))

    def add_step_to_dataset(
        self, dataset_id: str, step_id: str, metadata: Optional[Dict] = None
    ):
        return self.gql_helper(
            *add_step_to_dataset_helper(dataset_id, step_id, metadata)
        )

    def add_generation_to_dataset(
        self, dataset_id: str, generation_id: str, metadata: Optional[Dict] = None
    ):
        return self.gql_helper(
            *add_generation_to_dataset_helper(dataset_id, generation_id, metadata)
        )

    # Prompt API

    def create_prompt_lineage(self, name: str, description: Optional[str] = None):
        return self.gql_helper(*create_prompt_lineage_helper(name, description))

    def create_prompt(
        self,
        name: str,
        template_messages: List[GenerationMessage],
        settings: Optional[Dict] = None,
    ):
        lineage = self.create_prompt_lineage(name)
        lineage_id = lineage["id"]
        return self.gql_helper(
            *create_prompt_helper(self, lineage_id, template_messages, settings)
        )

    def get_prompt(self, name: str, version: Optional[int] = None):
        return self.gql_helper(*get_prompt_helper(self, name, version))


class AsyncLiteralAPI(BaseLiteralAPI):
    R = TypeVar("R")

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

    async def gql_helper(
        self,
        query: str,
        description: str,
        variables: Dict,
        process_response: Callable[..., R],
    ) -> R:
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
        self, id: Optional[str] = None, identifier: Optional[str] = None
    ):
        return await self.gql_helper(*get_user_helper(id, identifier))

    async def create_user(self, identifier: str, metadata: Optional[Dict] = None):
        return await self.gql_helper(*create_user_helper(identifier, metadata))

    async def update_user(
        self, id: str, identifier: Optional[str] = None, metadata: Optional[Dict] = None
    ):
        return await self.gql_helper(*update_user_helper(id, identifier, metadata))

    async def delete_user(self, id: str):
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
        return await self.gql_helper(
            *get_threads_helper(first, after, before, filters, order_by)
        )

    async def list_threads(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[threads_filters] = None,
        order_by: Optional[threads_order_by] = None,
    ):
        return await self.gql_helper(
            *list_threads_helper(first, after, before, filters, order_by)
        )

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
        return await self.gql_helper(
            *create_thread_helper(name, metadata, participant_id, environment, tags)
        )

    async def upsert_thread(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return await self.gql_helper(
            *upsert_thread_helper(id, name, metadata, participant_id, environment, tags)
        )

    async def update_thread(
        self,
        id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        return await self.gql_helper(
            *update_thread_helper(id, name, metadata, participant_id, environment, tags)
        )

    async def delete_thread(self, id: str):
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
        return await self.gql_helper(
            *get_scores_helper(first, after, before, filters, order_by)
        )

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
        return await self.gql_helper(
            *create_score_helper(
                name,
                value,
                type,
                step_id,
                generation_id,
                dataset_experiment_item_id,
                comment,
                tags,
            )
        )

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
        (
            query,
            description,
            variables,
            content,
            process_response,
        ) = create_attachment_helper(
            thread_id=thread_id,
            step_id=step_id,
            id=id,
            metadata=metadata,
            mime=mime,
            name=name,
            object_key=object_key,
            url=url,
            content=content,
            path=path,
        )

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
        return await self.gql_helper(
            *create_step_helper(
                thread_id=thread_id,
                type=type,
                start_time=start_time,
                end_time=end_time,
                input=input,
                output=output,
                metadata=metadata,
                parent_id=parent_id,
                name=name,
                tags=tags,
            )
        )

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
        return await self.gql_helper(
            *update_step_helper(
                id=id,
                type=type,
                input=input,
                output=output,
                metadata=metadata,
                name=name,
                tags=tags,
                start_time=start_time,
                end_time=end_time,
                parent_id=parent_id,
            )
        )

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

    async def send_steps(self, steps: List[Union[StepDict, "Step"]]):
        return await self.gql_helper(*send_steps_helper(steps=steps))

    # Generation API

    async def get_generations(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[generations_filters] = None,
        order_by: Optional[generations_order_by] = None,
    ):
        return await self.gql_helper(
            *get_generations_helper(first, after, before, filters, order_by)
        )

    async def create_generation(
        self, generation: Union[ChatGeneration, CompletionGeneration]
    ):
        return await self.gql_helper(*create_generation_helper(generation))

    # Dataset API

    async def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        type: DatasetType = "key_value",
    ):
        sync_api = LiteralAPI(self.api_key, self.url)
        return await self.gql_helper(
            *create_dataset_helper(sync_api, name, description, metadata, type)
        )

    async def get_dataset(self, id: str):
        sync_api = LiteralAPI(self.api_key, self.url)
        subpath, _, variables, process_response = get_dataset_helper(sync_api, id)
        response = await self.make_rest_call(subpath, variables)
        return process_response(response)

    async def update_dataset(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        sync_api = LiteralAPI(self.api_key, self.url)
        return await self.gql_helper(
            *update_dataset_helper(sync_api, id, name, description, metadata)
        )

    async def delete_dataset(self, id: str):
        sync_api = LiteralAPI(self.api_key, self.url)
        return await self.gql_helper(*delete_dataset_helper(sync_api, id))

    # DatasetItem API

    async def create_dataset_item(
        self,
        dataset_id: str,
        input: Dict,
        expected_output: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        return await self.gql_helper(
            *create_dataset_item_helper(dataset_id, input, expected_output, metadata)
        )

    async def get_dataset_item(self, id: str):
        return await self.gql_helper(*get_dataset_item_helper(id))

    async def delete_dataset_item(self, id: str):
        return await self.gql_helper(*delete_dataset_item_helper(id))

    async def add_step_to_dataset(
        self, dataset_id: str, step_id: str, metadata: Optional[Dict] = None
    ):
        return await self.gql_helper(
            *add_step_to_dataset_helper(dataset_id, step_id, metadata)
        )

    async def add_generation_to_dataset(
        self, dataset_id: str, generation_id: str, metadata: Optional[Dict] = None
    ):
        return await self.gql_helper(
            *add_generation_to_dataset_helper(dataset_id, generation_id, metadata)
        )

    # Prompt API

    async def create_prompt_lineage(self, name: str, description: Optional[str] = None):
        return await self.gql_helper(*create_prompt_lineage_helper(name, description))

    async def create_prompt(
        self,
        name: str,
        template_messages: List[GenerationMessage],
        settings: Optional[Dict] = None,
    ):
        lineage = await self.create_prompt_lineage(name)
        lineage_id = lineage["id"]
        sync_api = LiteralAPI(self.api_key, self.url)
        return await self.gql_helper(
            *create_prompt_helper(sync_api, lineage_id, template_messages, settings)
        )

    async def get_prompt(self, name: str, version: Optional[int] = None):
        sync_api = LiteralAPI(self.api_key, self.url)
        return await self.gql_helper(*get_prompt_helper(sync_api, name, version))
