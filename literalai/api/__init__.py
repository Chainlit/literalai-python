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

from typing_extensions import deprecated

from literalai.dataset import DatasetType
from literalai.dataset_experiment import DatasetExperiment, DatasetExperimentItem
from literalai.filter import (
    generations_filters,
    generations_order_by,
    scores_filters,
    scores_order_by,
    steps_filters,
    steps_order_by,
    threads_filters,
    threads_order_by,
    users_filters,
)
from literalai.prompt import Prompt, ProviderSettings

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
    create_experiment_helper,
    create_experiment_item_helper,
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
    promote_prompt_helper,
)
from .score_helpers import (
    ScoreUpdate,
    check_scores_finite,
    create_score_helper,
    create_scores_query_builder,
    delete_score_helper,
    get_scores_helper,
    update_score_helper,
)
from .step_helpers import (
    create_step_helper,
    delete_step_helper,
    get_step_helper,
    get_steps_helper,
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
    PaginatedResponse,
    Score,
    ScoreDict,
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

    def _prepare_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively checks and converts bytes objects in variables.
        """

        def handle_bytes(item):
            if isinstance(item, bytes):
                return "STRIPPED_BINARY_DATA"
            elif isinstance(item, dict):
                return {k: handle_bytes(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [handle_bytes(i) for i in item]
            elif isinstance(item, tuple):
                return tuple(handle_bytes(i) for i in item)
            return item

        return handle_bytes(variables)


class LiteralAPI(BaseLiteralAPI):
    R = TypeVar("R")

    def make_gql_call(
        self, description: str, query: str, variables: Dict[str, Any]
    ) -> Dict:
        """
        Executes a GraphQL call with the provided query and variables.

        Args:
            description (str): Description of the GraphQL operation for logging purposes.
            query (str): The GraphQL query to be executed.
            variables (Dict[str, Any]): Variables required for the GraphQL query.

        Returns:
            Dict: The JSON response from the GraphQL endpoint.

        Raises:
            Exception: If the GraphQL call fails or returns errors.
        """

        def raise_error(error):
            logger.error(f"Failed to {description}: {error}")
            raise Exception(error)

        variables = self._prepare_variables(variables)
        with httpx.Client(follow_redirects=True) as client:
            response = client.post(
                self.graphql_endpoint,
                json={"query": query, "variables": variables},
                headers=self.headers,
                timeout=10,
            )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                raise_error(f"Failed to {description}: {response.text}")

            try:
                json = response.json()
            except ValueError as e:
                raise_error(
                    f"Failed to parse JSON response: {e}, content: {response.content!r}"
                )

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
        """
        Executes a REST API call to the specified subpath with the given body.

        Args:
            subpath (str): The subpath of the REST API endpoint.
            body (Dict[str, Any]): The JSON body to send with the POST request.

        Returns:
            Dict: The JSON response from the REST API endpoint.
        """
        with httpx.Client(follow_redirects=True) as client:
            response = client.post(
                self.rest_endpoint + subpath,
                json=body,
                headers=self.headers,
                timeout=20,
            )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                message = f"Failed to call {subpath}: {response.text}"
                logger.error(message)
                raise Exception(message)

            try:
                return response.json()
            except ValueError as e:
                raise ValueError(
                    f"Failed to parse JSON response: {e}, content: {response.content!r}"
                )

    def gql_helper(
        self,
        query: str,
        description: str,
        variables: Dict,
        process_response: Callable[..., R],
    ) -> R:
        """
        Helper function to make a GraphQL call and process the response.

        Args:
            query (str): The GraphQL query to execute.
            description (str): Description of the GraphQL operation for logging purposes.
            variables (Dict): Variables required for the GraphQL query.
            process_response (Callable[..., R]): A function to process the response.

        Returns:
            R: The result of processing the response.
        """
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
        """
        Retrieves a list of users based on pagination and optional filters.

        Args:
            first (Optional[int]): The number of users to retrieve.
            after (Optional[str]): A cursor for use in pagination, fetching records after this cursor.
            before (Optional[str]): A cursor for use in pagination, fetching records before this cursor.
            filters (Optional[users_filters]): Filters to apply to the user query.

        Returns:
            Dict: A dictionary containing the queried user data.
        """
        return self.gql_helper(*get_users_helper(first, after, before, filters))

    def get_user(self, id: Optional[str] = None, identifier: Optional[str] = None):
        """
        Retrieves a user based on the provided ID or identifier.

        Args:
            id (Optional[str]): The unique ID of the user.
            identifier (Optional[str]): A unique identifier for the user, such as a username or email.

        Returns:
            The user data as returned by the GraphQL helper function.
        """
        return self.gql_helper(*get_user_helper(id, identifier))

    def create_user(self, identifier: str, metadata: Optional[Dict] = None):
        """
        Creates a new user with the specified identifier and optional metadata.

        Args:
            identifier (str): A unique identifier for the user, such as a username or email.
            metadata (Optional[Dict]): Additional data associated with the user.

        Returns:
            The result of the GraphQL call to create a user.
        """
        return self.gql_helper(*create_user_helper(identifier, metadata))

    def update_user(
        self, id: str, identifier: Optional[str] = None, metadata: Optional[Dict] = None
    ):
        """
        Updates an existing user identified by the given ID, with optional new identifier and metadata.

        Args:
            id (str): The unique ID of the user to update.
            identifier (Optional[str]): A new identifier for the user, such as a username or email.
            metadata (Optional[Dict]): New or updated metadata for the user.

        Returns:
            The result of the GraphQL call to update the user.
        """
        return self.gql_helper(*update_user_helper(id, identifier, metadata))

    def delete_user(self, id: str):
        """
        Deletes a user identified by the given ID.

        Args:
            id (str): The unique ID of the user to delete.

        Returns:
            The result of the GraphQL call to delete the user.
        """
        return self.gql_helper(*delete_user_helper(id))

    def get_or_create_user(self, identifier: str, metadata: Optional[Dict] = None):
        """
        Retrieves a user by their identifier, or creates a new user if they do not exist.

        Args:
            identifier (str): The identifier of the user to retrieve or create.
            metadata (Optional[Dict]): Metadata to associate with the user if they are created.

        Returns:
            The existing or newly created user data.
        """
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
        step_types_to_keep: Optional[List[StepType]] = None,
    ):
        """
        Fetches a list of threads based on pagination and optional filters.

        Args:
            first (Optional[int]): Number of threads to fetch.
            after (Optional[str]): Cursor for pagination, fetch threads after this cursor.
            before (Optional[str]): Cursor for pagination, fetch threads before this cursor.
            filters (Optional[threads_filters]): Filters to apply on the threads query.
            order_by (Optional[threads_order_by]): Order by clause for threads.

        Returns:
            A list of threads that match the criteria.
        """
        return self.gql_helper(
            *get_threads_helper(
                first, after, before, filters, order_by, step_types_to_keep
            )
        )

    def list_threads(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[threads_filters] = None,
        order_by: Optional[threads_order_by] = None,
    ):
        """
        Lists threads based on pagination and optional filters, similar to get_threads but may include additional processing.

        Args:
            first (Optional[int]): Number of threads to list.
            after (Optional[str]): Cursor for pagination, list threads after this cursor.
            before (Optional[str]): Cursor for pagination, list threads before this cursor.
            filters (Optional[threads_filters]): Filters to apply on the threads listing.
            order_by (Optional[threads_order_by]): Order by clause for threads.

        Returns:
            A list of threads that match the criteria.
        """
        return self.gql_helper(
            *list_threads_helper(first, after, before, filters, order_by)
        )

    def get_thread(self, id: str):
        """
        Retrieves a single thread by its ID.

        Args:
            id (str): The unique identifier of the thread.

        Returns:
            The thread corresponding to the provided ID.
        """
        return self.gql_helper(*get_thread_helper(id))

    def create_thread(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Creates a new thread with the specified details.

        Args:
            name (Optional[str]): Name of the thread.
            metadata (Optional[Dict]): Metadata associated with the thread.
            participant_id (Optional[str]): Identifier for the participant.
            environment (Optional[str]): Environment in which the thread operates.
            tags (Optional[List[str]]): List of tags associated with the thread.

        Returns:
            The newly created thread.
        """
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
        """
        Updates an existing thread or creates a new one if it does not exist.

        Args:
            id (str): The unique identifier of the thread.
            name (Optional[str]): Name of the thread.
            metadata (Optional[Dict]): Metadata associated with the thread.
            participant_id (Optional[str]): Identifier for the participant.
            environment (Optional[str]): Environment in which the thread operates.
            tags (Optional[List[str]]): List of tags associated with the thread.

        Returns:
            The updated or newly created thread.
        """
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
        """
        Updates the specified details of an existing thread.

        Args:
            id (str): The unique identifier of the thread to update.
            name (Optional[str]): New name of the thread.
            metadata (Optional[Dict]): New metadata for the thread.
            participant_id (Optional[str]): New identifier for the participant.
            environment (Optional[str]): New environment for the thread.
            tags (Optional[List[str]]): New list of tags for the thread.

        Returns:
            The updated thread.
        """
        return self.gql_helper(
            *update_thread_helper(id, name, metadata, participant_id, environment, tags)
        )

    def delete_thread(self, id: str):
        """
        Deletes a thread identified by its ID.

        Args:
            id (str): The unique identifier of the thread to delete.

        Returns:
            The result of the deletion operation.
        """
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

    def create_scores(self, scores: List[ScoreDict]):
        check_scores_finite(scores)

        query = create_scores_query_builder(scores)
        variables = {}
        for id, score in enumerate(scores):
            for k, v in score.items():
                variables[f"{k}_{id}"] = v

        def process_response(response):
            return [Score.from_dict(x) for x in response["data"].values()]

        return self.gql_helper(query, "create scores", variables, process_response)

    def create_score(
        self,
        name: str,
        value: float,
        type: ScoreType,
        step_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        dataset_experiment_item_id: Optional[str] = None,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Creates a single score in the database.

        Args:
            name (str): The name of the score.
            value (float): The numerical value of the score.
            type (ScoreType): The type of the score.
            step_id (Optional[str]): The ID of the step associated with the score.
            generation_id (Optional[str]): The ID of the generation associated with the score.
            dataset_experiment_item_id (Optional[str]): The ID of the dataset experiment item associated with the score.
            comment (Optional[str]): An optional comment about the score.
            tags (Optional[List[str]]): Optional tags associated with the score.

        Returns:
            The created Score object.
        """
        check_scores_finite([{"name": name, "value": value}])

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
        """
        Updates a score identified by its ID with new parameters.

        Args:
            id (str): The unique identifier of the score to update.
            update_params (ScoreUpdate): A dictionary of parameters to update in the score.

        Returns:
            The result of the update operation.
        """
        return self.gql_helper(*update_score_helper(id, update_params))

    def delete_score(self, id: str):
        """
        Deletes a score identified by its ID.

        Args:
            id (str): The unique identifier of the score to delete.

        Returns:
            The result of the deletion operation.
        """
        return self.gql_helper(*delete_score_helper(id))

    # Attachment API

    def upload_file(
        self,
        content: Union[bytes, str],
        thread_id: Optional[str] = None,
        mime: Optional[str] = "application/octet-stream",
    ) -> Dict:
        """
        Uploads a file to the server.

        Args:
            content (Union[bytes, str]): The content of the file to upload.
            thread_id (Optional[str]): The ID of the thread associated with the file.
            mime (Optional[str]): The MIME type of the file. Defaults to 'application/octet-stream'.

        Returns:
            Dict: A dictionary containing the object key and URL of the uploaded file, or None values if the upload fails.
        """
        id = str(uuid.uuid4())
        body = {"fileName": id, "contentType": mime}
        if thread_id:
            body["threadId"] = thread_id

        path = "/api/upload/file"

        with httpx.Client(follow_redirects=True) as client:
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

        with httpx.Client(follow_redirects=True) as client:
            if upload_type == "raw":
                upload_response = client.request(
                    url=url,
                    headers=headers,
                    method=method,
                    data=content,  # type: ignore
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
        """
        Creates an attachment associated with a thread and step, potentially uploading file content.

        Args:
            thread_id (str): The ID of the thread to which the attachment is linked.
            step_id (str): The ID of the step to which the attachment is linked.
            id (Optional[str]): The ID of the attachment, if updating an existing one.
            metadata (Optional[Dict]): Metadata associated with the attachment.
            mime (Optional[str]): MIME type of the file, if content is provided.
            name (Optional[str]): Name of the attachment.
            object_key (Optional[str]): Object key of the uploaded file, if already known.
            url (Optional[str]): URL of the uploaded file, if already known.
            content (Optional[Union[bytes, str]]): File content to upload.
            path (Optional[str]): Path where the file should be stored.

        Returns:
            Attachment: The created or updated attachment object.
        """
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
        """
        Updates an existing attachment with new parameters.

        Args:
            id (str): The unique identifier of the attachment to update.
            update_params (AttachmentUpload): The parameters to update in the attachment.

        Returns:
            The result of the update operation.
        """
        return self.gql_helper(*update_attachment_helper(id, update_params))

    def get_attachment(self, id: str):
        """
        Retrieves an attachment by its ID.

        Args:
            id (str): The unique identifier of the attachment to retrieve.

        Returns:
            The attachment data as returned by the GraphQL helper function.
        """
        return self.gql_helper(*get_attachment_helper(id))

    def delete_attachment(self, id: str):
        """
        Deletes an attachment identified by its ID.

        Args:
            id (str): The unique identifier of the attachment to delete.

        Returns:
            The result of the deletion operation.
        """
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
        """
        Creates a new step with the specified parameters.

        Args:
            thread_id (Optional[str]): The ID of the thread this step is associated with.
            type (Optional[StepType]): The type of the step, defaults to "undefined".
            start_time (Optional[str]): The start time of the step.
            end_time (Optional[str]): The end time of the step.
            input (Optional[Dict]): Input data for the step.
            output (Optional[Dict]): Output data from the step.
            metadata (Optional[Dict]): Metadata associated with the step.
            parent_id (Optional[str]): The ID of the parent step, if any.
            name (Optional[str]): The name of the step.
            tags (Optional[List[str]]): Tags associated with the step.

        Returns:
            The result of the GraphQL helper function for creating a step.
        """
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
        """
        Updates an existing step identified by its ID with new parameters.

        Args:
            id (str): The unique identifier of the step to update.
            type (Optional[StepType]): The type of the step.
            input (Optional[str]): Input data for the step.
            output (Optional[str]): Output data from the step.
            metadata (Optional[Dict]): Metadata associated with the step.
            name (Optional[str]): The name of the step.
            tags (Optional[List[str]]): Tags associated with the step.
            start_time (Optional[str]): The start time of the step.
            end_time (Optional[str]): The end time of the step.
            parent_id (Optional[str]): The ID of the parent step, if any.

        Returns:
            The result of the GraphQL helper function for updating a step.
        """
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

    def get_steps(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[steps_filters] = None,
        order_by: Optional[steps_order_by] = None,
    ) -> PaginatedResponse[Step]:
        """
        Fetches a list of steps based on pagination and optional filters.

        Args:
            first (Optional[int]): Number of steps to fetch.
            after (Optional[str]): Cursor for pagination, fetch steps after this cursor.
            before (Optional[str]): Cursor for pagination, fetch steps before this cursor.
            filters (Optional[steps_filters]): Filters to apply on the steps query.
            order_by (Optional[steps_order_by]): Order by clause for steps.

        Returns:
            A list of steps that match the criteria.
        """
        return self.gql_helper(
            *get_steps_helper(first, after, before, filters, order_by)
        )

    def get_step(
        self,
        id: str,
    ):
        """
        Retrieves a step by its ID.

        Args:
            id (str): The unique identifier of the step to retrieve.

        Returns:
            The step data as returned by the GraphQL helper function.
        """
        return self.gql_helper(*get_step_helper(id=id))

    def delete_step(
        self,
        id: str,
    ):
        """
        Deletes a step identified by its ID.

        Args:
            id (str): The unique identifier of the step to delete.

        Returns:
            The result of the deletion operation.
        """
        return self.gql_helper(*delete_step_helper(id=id))

    def send_steps(self, steps: List[Union[StepDict, "Step"]]):
        """
        Sends a list of steps to be processed.

        Args:
            steps (List[Union[StepDict, "Step"]]): A list of steps or step dictionaries to send.

        Returns:
            The result of the GraphQL helper function for sending steps.
        """
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
        """
        Fetches a list of generations based on pagination and optional filters.

        Args:
            first (Optional[int]): The number of generations to retrieve.
            after (Optional[str]): A cursor for use in pagination, fetching records after this cursor.
            before (Optional[str]): A cursor for use in pagination, fetching records before this cursor.
            filters (Optional[generations_filters]): Filters to apply to the generations query.
            order_by (Optional[generations_order_by]): Order by clause for generations.

        Returns:
            A list of generations that match the criteria.
        """
        return self.gql_helper(
            *get_generations_helper(first, after, before, filters, order_by)
        )

    def create_generation(
        self, generation: Union[ChatGeneration, CompletionGeneration]
    ):
        """
        Creates a new generation, either a chat or completion type.

        Args:
            generation (Union[ChatGeneration, CompletionGeneration]): The generation data to create.

        Returns:
            The result of the creation operation.
        """
        return self.gql_helper(*create_generation_helper(generation))

    # Dataset API

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        type: DatasetType = "key_value",
    ):
        """
        Creates a new dataset with the specified properties.

        Args:
            name (str): The name of the dataset.
            description (Optional[str]): A description of the dataset.
            metadata (Optional[Dict]): Additional metadata for the dataset.
            type (DatasetType): The type of the dataset, defaults to "key_value".

        Returns:
            The result of the dataset creation operation.
        """
        return self.gql_helper(
            *create_dataset_helper(self, name, description, metadata, type)
        )

    def get_dataset(self, id: Optional[str] = None, name: Optional[str] = None):
        """
        Retrieves a dataset by its ID or name.

        Args:
            id (Optional[str]): The unique identifier of the dataset.
            name (Optional[str]): The name of the dataset.

        Returns:
            The dataset data as returned by the REST helper function.
        """
        subpath, _, variables, process_response = get_dataset_helper(
            self, id=id, name=name
        )
        response = self.make_rest_call(subpath, variables)
        return process_response(response)

    def update_dataset(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Updates an existing dataset identified by its ID with new properties.

        Args:
            id (str): The unique identifier of the dataset to update.
            name (Optional[str]): A new name for the dataset.
            description (Optional[str]): A new description for the dataset.
            metadata (Optional[Dict]): New or updated metadata for the dataset.

        Returns:
            The result of the dataset update operation.
        """
        return self.gql_helper(
            *update_dataset_helper(self, id, name, description, metadata)
        )

    def delete_dataset(self, id: str):
        """
        Deletes a dataset identified by its ID.

        Args:
            id (str): The unique identifier of the dataset to delete.

        Returns:
            The result of the deletion operation.
        """
        return self.gql_helper(*delete_dataset_helper(self, id))

    # Dataset Experiment APIs

    def create_experiment(
        self,
        dataset_id: str,
        name: str,
        prompt_id: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> "DatasetExperiment":
        """
        Creates a new experiment associated with a specific dataset.

        Args:
            dataset_id (str): The unique identifier of the dataset.
            name (str): The name of the experiment.
            prompt_id (Optional[str]): The identifier of the prompt associated with the experiment.
            params (Optional[Dict]): Additional parameters for the experiment.

        Returns:
            DatasetExperiment: The newly created experiment object.
        """
        return self.gql_helper(
            *create_experiment_helper(self, dataset_id, name, prompt_id, params)
        )

    def create_experiment_item(
        self, experiment_item: DatasetExperimentItem
    ) -> DatasetExperimentItem:
        """
        Creates an experiment item within an existing experiment.

        Args:
            experiment_item (DatasetExperimentItem): The experiment item to be created, containing all necessary data.

        Returns:
            DatasetExperimentItem: The newly created experiment item with scores attached.
        """
        # Create the dataset experiment item
        result = self.gql_helper(
            *create_experiment_item_helper(
                dataset_experiment_id=experiment_item.dataset_experiment_id,
                dataset_item_id=experiment_item.dataset_item_id,
                input=experiment_item.input,
                output=experiment_item.output,
            )
        )

        for score in experiment_item.scores:
            score["datasetExperimentItemId"] = result.id

        # Create the scores and add to experiment item.
        result.scores = self.create_scores(experiment_item.scores)

        return result

    # Dataset Item API

    def create_dataset_item(
        self,
        dataset_id: str,
        input: Dict,
        expected_output: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Creates a new dataset item with the specified properties.

        Args:
            dataset_id (str): The unique identifier of the dataset.
            input (Dict): The input data for the dataset item.
            expected_output (Optional[Dict]): The expected output data for the dataset item.
            metadata (Optional[Dict]): Additional metadata for the dataset item.

        Returns:
            Dict: The result of the dataset item creation operation.
        """
        return self.gql_helper(
            *create_dataset_item_helper(dataset_id, input, expected_output, metadata)
        )

    def get_dataset_item(self, id: str):
        """
        Retrieves a dataset item by its unique identifier.

        Args:
            id (str): The unique identifier of the dataset item to retrieve.

        Returns:
            Dict: The dataset item data.
        """
        return self.gql_helper(*get_dataset_item_helper(id))

    def delete_dataset_item(self, id: str):
        """
        Deletes a dataset item by its unique identifier.

        Args:
            id (str): The unique identifier of the dataset item to delete.

        Returns:
            Dict: The result of the dataset item deletion operation.
        """
        return self.gql_helper(*delete_dataset_item_helper(id))

    def add_step_to_dataset(
        self, dataset_id: str, step_id: str, metadata: Optional[Dict] = None
    ):
        """
        Adds a step to a dataset.

        Args:
            dataset_id (str): The unique identifier of the dataset.
            step_id (str): The unique identifier of the step to add.
            metadata (Optional[Dict]): Additional metadata for the step being added.

        Returns:
            Dict: The result of adding the step to the dataset.
        """
        return self.gql_helper(
            *add_step_to_dataset_helper(dataset_id, step_id, metadata)
        )

    def add_generation_to_dataset(
        self, dataset_id: str, generation_id: str, metadata: Optional[Dict] = None
    ):
        """
        Adds a generation to a dataset.

        Args:
            dataset_id (str): The unique identifier of the dataset.
            generation_id (str): The unique identifier of the generation to add.
            metadata (Optional[Dict]): Additional metadata for the generation being added.

        Returns:
            Dict: The result of adding the generation to the dataset.
        """
        return self.gql_helper(
            *add_generation_to_dataset_helper(dataset_id, generation_id, metadata)
        )

    # Prompt API

    def get_or_create_prompt_lineage(
        self, name: str, description: Optional[str] = None
    ):
        """
        Creates a prompt lineage with the specified name and optional description.
        If the prompt lineage with that name already exists, it is returned.

        Args:
            name (str): The name of the prompt lineage.
            description (Optional[str]): An optional description of the prompt lineage.

        Returns:
            Dict: The result of the prompt lineage creation operation.
        """
        return self.gql_helper(*create_prompt_lineage_helper(name, description))

    @deprecated('Please use "get_or_create_prompt_lineage" instead.')
    def create_prompt_lineage(self, name: str, description: Optional[str] = None):
        return self.get_or_create_prompt_lineage(name, description)

    def get_or_create_prompt(
        self,
        name: str,
        template_messages: List[GenerationMessage],
        settings: Optional[ProviderSettings] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Prompt:
        """
        A `Prompt` is fully defined by its `name`, `template_messages`, `settings` and tools.
        If a prompt already exists for the given arguments, it is returned.
        Otherwise, a new prompt is created.

        Args:
            name (str): The name of the prompt to retrieve or create.
            template_messages (List[GenerationMessage]): A list of template messages for the prompt.
            settings (Optional[Dict]): Optional settings for the prompt.

        Returns:
            Prompt: The prompt that was retrieved or created.
        """
        lineage = self.get_or_create_prompt_lineage(name)
        lineage_id = lineage["id"]
        return self.gql_helper(
            *create_prompt_helper(self, lineage_id, template_messages, settings, tools)
        )

    @deprecated('Please use "get_or_create_prompt" instead.')
    def create_prompt(
        self,
        name: str,
        template_messages: List[GenerationMessage],
        settings: Optional[ProviderSettings] = None,
    ) -> Prompt:
        return self.get_or_create_prompt(name, template_messages, settings)

    def get_prompt(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Prompt:
        """
        Gets a prompt either by:
        - `id`
        - or `name` and (optional) `version`

        Either the `id` or the `name` must be provided.
        If both are provided, the `id` is used.

        Args:
            id (str): The unique identifier of the prompt to retrieve.
            name (str): The name of the prompt to retrieve.
            version (Optional[int]): The version number of the prompt to retrieve.

        Returns:
            Prompt: The prompt with the given identifier or name.
        """
        if id:
            return self.gql_helper(*get_prompt_helper(self, id=id))
        elif name:
            return self.gql_helper(*get_prompt_helper(self, name=name, version=version))
        else:
            raise ValueError("Either the `id` or the `name` must be provided.")

    def promote_prompt(self, name: str, version: int) -> str:
        """
        Promotes the prompt with name to target version.

        Args:
            name (str): The name of the prompt lineage.
            version (int): The version number to promote.

        Returns:
            str: The champion prompt ID.
        """
        lineage = self.get_or_create_prompt_lineage(name)
        lineage_id = lineage["id"]

        return self.gql_helper(*promote_prompt_helper(lineage_id, version))

    # Misc API

    def get_my_project_id(self):
        """
        Retrieves the projectId associated to the API key.

        Returns:
            The projectId associated to the API key.
        """
        response = self.make_rest_call("/my-project", {})
        return response["projectId"]


class AsyncLiteralAPI(BaseLiteralAPI):
    R = TypeVar("R")

    async def make_gql_call(
        self, description: str, query: str, variables: Dict[str, Any]
    ) -> Dict:
        """
        Asynchronously makes a GraphQL call using the provided query and variables.

        Args:
            description (str): Description of the GraphQL operation for logging purposes.
            query (str): The GraphQL query to be executed.
            variables (Dict[str, Any]): Variables required for the GraphQL query.

        Returns:
            Dict: The JSON response from the GraphQL endpoint.

        Raises:
            Exception: If the GraphQL call fails or returns errors.
        """

        def raise_error(error):
            logger.error(f"Failed to {description}: {error}")
            raise Exception(error)

        variables = self._prepare_variables(variables)

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(
                self.graphql_endpoint,
                json={"query": query, "variables": variables},
                headers=self.headers,
                timeout=10,
            )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                raise_error(f"Failed to {description}: {response.text}")

            try:
                json = response.json()
            except ValueError as e:
                raise_error(
                    f"Failed to parse JSON response: {e}, content: {response.content!r}"
                )

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
        """
        Asynchronously makes a REST API call to a specified subpath with the provided body.

        Args:
            subpath (str): The endpoint subpath to which the POST request is made.
            body (Dict[str, Any]): The JSON body of the POST request.

        Returns:
            Dict: The JSON response from the REST API endpoint.
        """
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(
                self.rest_endpoint + subpath,
                json=body,
                headers=self.headers,
                timeout=20,
            )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                message = f"Failed to call {subpath}: {response.text}"
                logger.error(message)
                raise Exception(message)

            try:
                return response.json()
            except ValueError as e:
                raise ValueError(
                    f"Failed to parse JSON response: {e}, content: {response.content!r}"
                )

    async def gql_helper(
        self,
        query: str,
        description: str,
        variables: Dict,
        process_response: Callable[..., R],
    ) -> R:
        """
        Helper function to process a GraphQL query by making an asynchronous call and processing the response.

        Args:
            query (str): The GraphQL query to be executed.
            description (str): Description of the GraphQL operation for logging purposes.
            variables (Dict): Variables required for the GraphQL query.
            process_response (Callable[..., R]): The function to process the response.

        Returns:
            R: The result of processing the response.
        """
        response = await self.make_gql_call(description, query, variables)
        return process_response(response)

    async def get_users(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[users_filters] = None,
    ):
        """
        Asynchronously fetches a list of users based on pagination and optional filters.

        Args:
            first (Optional[int]): The number of users to retrieve.
            after (Optional[str]): A cursor for use in pagination, fetching records after this cursor.
            before (Optional[str]): A cursor for use in pagination, fetching records before this cursor.
            filters (Optional[users_filters]): Filters to apply to the user query.

        Returns:
            The result of the GraphQL helper function for fetching users.
        """
        return await self.gql_helper(*get_users_helper(first, after, before, filters))

    # User API

    async def get_user(
        self, id: Optional[str] = None, identifier: Optional[str] = None
    ):
        """
        Asynchronously retrieves a user by ID or identifier.

        Args:
            id (Optional[str]): The unique identifier of the user to retrieve.
            identifier (Optional[str]): An alternative identifier for the user.

        Returns:
            The result of the GraphQL helper function for fetching a user.
        """
        return await self.gql_helper(*get_user_helper(id, identifier))

    async def create_user(self, identifier: str, metadata: Optional[Dict] = None):
        """
        Asynchronously creates a new user with the specified identifier and optional metadata.

        Args:
            identifier (str): The identifier for the new user.
            metadata (Optional[Dict]): Additional metadata for the user.

        Returns:
            The result of the GraphQL helper function for creating a user.
        """
        return await self.gql_helper(*create_user_helper(identifier, metadata))

    async def update_user(
        self, id: str, identifier: Optional[str] = None, metadata: Optional[Dict] = None
    ):
        """
        Asynchronously updates an existing user identified by ID with new identifier and/or metadata.

        Args:
            id (str): The unique identifier of the user to update.
            identifier (Optional[str]): New identifier for the user.
            metadata (Optional[Dict]): New metadata for the user.

        Returns:
            The result of the GraphQL helper function for updating a user.
        """
        return await self.gql_helper(*update_user_helper(id, identifier, metadata))

    async def delete_user(self, id: str):
        """
        Asynchronously deletes a user identified by ID.

        Args:
            id (str): The unique identifier of the user to delete.

        Returns:
            The result of the GraphQL helper function for deleting a user.
        """
        return await self.gql_helper(*delete_user_helper(id))

    async def get_or_create_user(
        self, identifier: str, metadata: Optional[Dict] = None
    ):
        """
        Asynchronously retrieves a user by identifier or creates a new one if it does not exist.

        Args:
            identifier (str): The identifier of the user to retrieve or create.
            metadata (Optional[Dict]): Metadata for the user if creation is necessary.

        Returns:
            The existing or newly created user.
        """
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
        step_types_to_keep: Optional[List[StepType]] = None,
    ):
        """
        Asynchronously fetches a list of threads based on pagination and optional filters and ordering.

        Args:
            first (Optional[int]): The number of threads to retrieve.
            after (Optional[str]): A cursor for use in pagination, fetching records after this cursor.
            before (Optional[str]): A cursor for use in pagination, fetching records before this cursor.
            filters (Optional[threads_filters]): Filters to apply to the thread query.
            order_by (Optional[threads_order_by]): Ordering criteria for the threads.

        Returns:
            The result of the GraphQL helper function for fetching threads.
        """
        return await self.gql_helper(
            *get_threads_helper(
                first, after, before, filters, order_by, step_types_to_keep
            )
        )

    async def list_threads(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[threads_filters] = None,
        order_by: Optional[threads_order_by] = None,
    ):
        """
        Asynchronously lists threads based on pagination and optional filters and ordering, similar to `get_threads`.

        Args:
            first (Optional[int]): The number of threads to list.
            after (Optional[str]): A cursor for use in pagination, fetching records after this cursor.
            before (Optional[str]): A cursor for use in pagination, fetching records before this cursor.
            filters (Optional[threads_filters]): Filters to apply to the thread query.
            order_by (Optional[threads_order_by]): Ordering criteria for the threads.

        Returns:
            The result of the GraphQL helper function for listing threads.
        """
        return await self.gql_helper(
            *list_threads_helper(first, after, before, filters, order_by)
        )

    async def get_thread(self, id: str):
        """
        Asynchronously retrieves a thread by its ID.

        Args:
            id (str): The unique identifier of the thread to retrieve.

        Returns:
            The result of the GraphQL helper function for fetching a thread.
        """
        return await self.gql_helper(*get_thread_helper(id))

    async def create_thread(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Asynchronously creates a new thread with specified details.

        Args:
            name (Optional[str]): The name of the thread.
            metadata (Optional[Dict]): Metadata associated with the thread.
            participant_id (Optional[str]): Identifier for the participant associated with the thread.
            environment (Optional[str]): The environment in which the thread operates.
            tags (Optional[List[str]]): Tags associated with the thread.

        Returns:
            The result of the GraphQL helper function for creating a thread.
        """
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
        """
        Asynchronously updates or inserts a thread based on the provided ID.

        Args:
            id (str): The unique identifier of the thread to upsert.
            name (Optional[str]): The name of the thread.
            metadata (Optional[Dict]): Metadata associated with the thread.
            participant_id (Optional[str]): Identifier for the participant associated with the thread.
            environment (Optional[str]): The environment in which the thread operates.
            tags (Optional[List[str]]): Tags associated with the thread.

        Returns:
            The result of the GraphQL helper function for upserting a thread.
        """
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
        """
        Asynchronously updates an existing thread identified by ID with new details.

        Args:
            id (str): The unique identifier of the thread to update.
            name (Optional[str]): New name of the thread.
            metadata (Optional[Dict]): New metadata for the thread.
            participant_id (Optional[str]): New identifier for the participant.
            environment (Optional[str]): New environment for the thread.
            tags (Optional[List[str]]): New list of tags for the thread.

        Returns:
            The result of the GraphQL helper function for updating a thread.
        """
        return await self.gql_helper(
            *update_thread_helper(id, name, metadata, participant_id, environment, tags)
        )

    async def delete_thread(self, id: str):
        """
        Asynchronously deletes a thread identified by its ID.

        Args:
            id (str): The unique identifier of the thread to delete.

        Returns:
            The result of the GraphQL helper function for deleting a thread.
        """
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
        """
        Asynchronously fetches scores based on pagination and optional filters.

        Args:
            first (Optional[int]): The number of scores to retrieve.
            after (Optional[str]): A cursor for use in pagination, fetching records after this cursor.
            before (Optional[str]): A cursor for use in pagination, fetching records before this cursor.
            filters (Optional[scores_filters]): Filters to apply to the scores query.
            order_by (Optional[scores_order_by]): Ordering options for the scores.

        Returns:
            The result of the GraphQL helper function for fetching scores.
        """
        return await self.gql_helper(
            *get_scores_helper(first, after, before, filters, order_by)
        )

    async def create_scores(self, scores: List[ScoreDict]):
        """
        Asynchronously creates multiple scores.

        Args:
            scores (List[ScoreDict]): A list of dictionaries representing the scores to be created.

        Returns:
            The result of the GraphQL helper function for creating scores.
        """
        check_scores_finite(scores)

        query = create_scores_query_builder(scores)
        variables = {}

        for id, score in enumerate(scores):
            for k, v in score.items():
                variables[f"{k}_{id}"] = v

        def process_response(response):
            return [Score.from_dict(x) for x in response["data"].values()]

        return await self.gql_helper(
            query, "create scores", variables, process_response
        )

    async def create_score(
        self,
        name: str,
        value: float,
        type: ScoreType,
        step_id: Optional[str] = None,
        generation_id: Optional[str] = None,
        dataset_experiment_item_id: Optional[str] = None,
        comment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Asynchronously creates a single score.

        Args:
            name (str): The name of the score.
            value (float): The numerical value of the score.
            type (ScoreType): The type of the score.
            step_id (Optional[str]): The ID of the step associated with the score.
            generation_id (Optional[str]): The ID of the generation associated with the score.
            dataset_experiment_item_id (Optional[str]): The ID of the dataset experiment item associated with the score.
            comment (Optional[str]): A comment associated with the score.
            tags (Optional[List[str]]): A list of tags associated with the score.

        Returns:
            The result of the GraphQL helper function for creating a score.
        """
        check_scores_finite([{"name": name, "value": value}])

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
        """
        Asynchronously updates a score identified by its ID.

        Args:
            id (str): The unique identifier of the score to update.
            update_params (ScoreUpdate): A dictionary of parameters to update.

        Returns:
            The result of the GraphQL helper function for updating a score.
        """
        return await self.gql_helper(*update_score_helper(id, update_params))

    async def delete_score(self, id: str):
        """
        Asynchronously deletes a score identified by its ID.

        Args:
            id (str): The unique identifier of the score to delete.

        Returns:
            The result of the GraphQL helper function for deleting a score.
        """
        return await self.gql_helper(*delete_score_helper(id))

    # Attachment API

    async def upload_file(
        self,
        content: Union[bytes, str],
        thread_id: str,
        mime: Optional[str] = "application/octet-stream",
    ) -> Dict:
        """
        Asynchronously uploads a file to the server.

        Args:
            content (Union[bytes, str]): The content of the file to upload.
            thread_id (str): The ID of the thread associated with the file.
            mime (Optional[str]): The MIME type of the file.

        Returns:
            A dictionary containing the object key and URL of the uploaded file.
        """
        id = str(uuid.uuid4())
        body = {"fileName": id, "contentType": mime, "threadId": thread_id}

        path = "/api/upload/file"

        async with httpx.AsyncClient(follow_redirects=True) as client:
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

        async with httpx.AsyncClient(follow_redirects=True) as client:
            if upload_type == "raw":
                upload_response = await client.request(
                    url=url,
                    headers=headers,
                    method=method,
                    data=content,  # type: ignore
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
        """
        Asynchronously creates an attachment and uploads it if content is provided.

        Args:
            thread_id (str): The ID of the thread associated with the attachment.
            step_id (str): The ID of the step associated with the attachment.
            id (Optional[str]): An optional unique identifier for the attachment.
            metadata (Optional[Dict]): Optional metadata for the attachment.
            mime (Optional[str]): The MIME type of the attachment.
            name (Optional[str]): The name of the attachment.
            object_key (Optional[str]): The object key for the attachment if already uploaded.
            url (Optional[str]): The URL of the attachment if already uploaded.
            content (Optional[Union[bytes, str]]): The content of the attachment to upload.
            path (Optional[str]): The file path of the attachment if it is to be uploaded from a local file.

        Returns:
            The attachment object created after the upload and creation process.
        """
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
        """
        Asynchronously updates an attachment identified by its ID.

        Args:
            id (str): The unique identifier of the attachment to update.
            update_params (AttachmentUpload): A dictionary of parameters to update the attachment.

        Returns:
            The result of the GraphQL helper function for updating an attachment.
        """
        return await self.gql_helper(*update_attachment_helper(id, update_params))

    async def get_attachment(self, id: str):
        """
        Asynchronously retrieves an attachment by its ID.

        Args:
            id (str): The unique identifier of the attachment to retrieve.

        Returns:
            The result of the GraphQL helper function for fetching an attachment.
        """
        return await self.gql_helper(*get_attachment_helper(id))

    async def delete_attachment(self, id: str):
        """
        Asynchronously deletes an attachment identified by its ID.

        Args:
            id (str): The unique identifier of the attachment to delete.

        Returns:
            The result of the GraphQL helper function for deleting an attachment.
        """
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
        """
        Asynchronously creates a new step with the specified parameters.

        Args:
            thread_id (Optional[str]): The ID of the thread associated with the step.
            type (Optional[StepType]): The type of the step, defaults to "undefined".
            start_time (Optional[str]): The start time of the step.
            end_time (Optional[str]): The end time of the step.
            input (Optional[Dict]): Input data for the step.
            output (Optional[Dict]): Output data from the step.
            metadata (Optional[Dict]): Metadata associated with the step.
            parent_id (Optional[str]): The ID of the parent step, if any.
            name (Optional[str]): The name of the step.
            tags (Optional[List[str]]): Tags associated with the step.

        Returns:
            The result of the GraphQL helper function for creating a step.
        """
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
        """
        Asynchronously updates an existing step identified by its ID with new parameters.

        Args:
            id (str): The unique identifier of the step to update.
            type (Optional[StepType]): The type of the step.
            input (Optional[str]): Input data for the step.
            output (Optional[str]): Output data from the step.
            metadata (Optional[Dict]): Metadata associated with the step.
            name (Optional[str]): The name of the step.
            tags (Optional[List[str]]): Tags associated with the step.
            start_time (Optional[str]): The start time of the step.
            end_time (Optional[str]): The end time of the step.
            parent_id (Optional[str]): The ID of the parent step, if any.

        Returns:
            The result of the GraphQL helper function for updating a step.
        """
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

    async def get_steps(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[steps_filters] = None,
        order_by: Optional[steps_order_by] = None,
    ) -> PaginatedResponse[Step]:
        return await self.gql_helper(
            *get_steps_helper(first, after, before, filters, order_by)
        )

    get_steps.__doc__ = LiteralAPI.get_steps.__doc__

    async def get_step(
        self,
        id: str,
    ):
        """
        Asynchronously retrieves a step by its ID.

        Args:
            id (str): The unique identifier of the step to retrieve.

        Returns:
            The result of the GraphQL helper function for fetching a step.
        """
        return await self.gql_helper(*get_step_helper(id=id))

    async def delete_step(
        self,
        id: str,
    ):
        """
        Asynchronously deletes a step identified by its ID.

        Args:
            id (str): The unique identifier of the step to delete.

        Returns:
            The result of the GraphQL helper function for deleting a step.
        """
        return await self.gql_helper(*delete_step_helper(id=id))

    async def send_steps(self, steps: List[Union[StepDict, "Step"]]):
        """
        Asynchronously sends a list of steps to be processed.

        Args:
            steps (List[Union[StepDict, "Step"]]): A list of steps or step dictionaries to send.

        Returns:
            The result of the GraphQL helper function for sending steps.
        """
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
        """
        Asynchronously fetches a list of generations based on pagination and optional filters.

        Args:
            first (Optional[int]): The number of generations to retrieve.
            after (Optional[str]): A cursor for use in pagination, fetching records after this cursor.
            before (Optional[str]): A cursor for use in pagination, fetching records before this cursor.
            filters (Optional[generations_filters]): Filters to apply to the generations query.
            order_by (Optional[generations_order_by]): Ordering options for the generations.

        Returns:
            The result of the GraphQL helper function for fetching generations.
        """
        return await self.gql_helper(
            *get_generations_helper(first, after, before, filters, order_by)
        )

    async def create_generation(
        self, generation: Union[ChatGeneration, CompletionGeneration]
    ):
        """
        Asynchronously creates a new generation with the specified details.

        Args:
            generation (Union[ChatGeneration, CompletionGeneration]): The generation data to create.

        Returns:
            The result of the GraphQL helper function for creating a generation.
        """
        return await self.gql_helper(*create_generation_helper(generation))

    # Dataset API

    async def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        type: DatasetType = "key_value",
    ):
        """
        Asynchronously creates a new dataset with the specified details.

        Args:
            name (str): The name of the dataset.
            description (Optional[str]): A description of the dataset.
            metadata (Optional[Dict]): Metadata associated with the dataset.
            type (DatasetType): The type of the dataset, defaults to "key_value".

        Returns:
            The result of the GraphQL helper function for creating a dataset.
        """
        sync_api = LiteralAPI(self.api_key, self.url)
        return await self.gql_helper(
            *create_dataset_helper(sync_api, name, description, metadata, type)
        )

    async def get_dataset(self, id: Optional[str] = None, name: Optional[str] = None):
        """
        Asynchronously retrieves a dataset by its ID or name.

        Args:
            id (Optional[str]): The unique identifier of the dataset to retrieve.
            name (Optional[str]): The name of the dataset to retrieve.

        Returns:
            The processed response from the REST API call.
        """
        sync_api = LiteralAPI(self.api_key, self.url)
        subpath, _, variables, process_response = get_dataset_helper(
            sync_api, id=id, name=name
        )
        response = await self.make_rest_call(subpath, variables)
        return process_response(response)

    async def update_dataset(
        self,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Asynchronously updates an existing dataset identified by its ID with new details.

        Args:
            id (str): The unique identifier of the dataset to update.
            name (Optional[str]): The new name of the dataset.
            description (Optional[str]): A new description for the dataset.
            metadata (Optional[Dict]): New metadata for the dataset.

        Returns:
            The result of the GraphQL helper function for updating a dataset.
        """
        sync_api = LiteralAPI(self.api_key, self.url)
        return await self.gql_helper(
            *update_dataset_helper(sync_api, id, name, description, metadata)
        )

    async def delete_dataset(self, id: str):
        """
        Asynchronously deletes a dataset identified by its ID.

        Args:
            id (str): The unique identifier of the dataset to delete.

        Returns:
            The result of the GraphQL helper function for deleting a dataset.
        """
        sync_api = LiteralAPI(self.api_key, self.url)
        return await self.gql_helper(*delete_dataset_helper(sync_api, id))

    # Dataset Experiment APIs

    async def create_experiment(
        self,
        dataset_id: str,
        name: str,
        prompt_id: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> "DatasetExperiment":
        sync_api = LiteralAPI(self.api_key, self.url)

        return await self.gql_helper(
            *create_experiment_helper(sync_api, dataset_id, name, prompt_id, params)
        )

    create_experiment.__doc__ = LiteralAPI.create_experiment.__doc__

    async def create_experiment_item(
        self, experiment_item: DatasetExperimentItem
    ) -> DatasetExperimentItem:
        """
        Asynchronously creates an item within an experiment.

        Args:
            experiment_item (DatasetExperimentItem): The experiment item to be created.

        Returns:
            DatasetExperimentItem: The created experiment item with updated scores.
        """
        check_scores_finite(experiment_item.scores)

        # Create the dataset experiment item
        result = await self.gql_helper(
            *create_experiment_item_helper(
                dataset_experiment_id=experiment_item.dataset_experiment_id,
                dataset_item_id=experiment_item.dataset_item_id,
                input=experiment_item.input,
                output=experiment_item.output,
            )
        )

        for score in experiment_item.scores:
            score["datasetExperimentItemId"] = result.id

        # Create the scores and add to experiment item.
        result.scores = await self.create_scores(experiment_item.scores)

        return result

    # DatasetItem API

    async def create_dataset_item(
        self,
        dataset_id: str,
        input: Dict,
        expected_output: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Asynchronously creates a dataset item.

        Args:
            dataset_id (str): The unique identifier of the dataset.
            input (Dict): The input data for the dataset item.
            expected_output (Optional[Dict]): The expected output data for the dataset item.
            metadata (Optional[Dict]): Additional metadata for the dataset item.

        Returns:
            The result of the GraphQL helper function for creating a dataset item.
        """
        return await self.gql_helper(
            *create_dataset_item_helper(dataset_id, input, expected_output, metadata)
        )

    async def get_dataset_item(self, id: str):
        """
        Asynchronously retrieves a dataset item by its ID.

        Args:
            id (str): The unique identifier of the dataset item.

        Returns:
            The result of the GraphQL helper function for fetching a dataset item.
        """
        return await self.gql_helper(*get_dataset_item_helper(id))

    async def delete_dataset_item(self, id: str):
        """
        Asynchronously deletes a dataset item by its ID.

        Args:
            id (str): The unique identifier of the dataset item to delete.

        Returns:
            The result of the GraphQL helper function for deleting a dataset item.
        """
        return await self.gql_helper(*delete_dataset_item_helper(id))

    async def add_step_to_dataset(
        self, dataset_id: str, step_id: str, metadata: Optional[Dict] = None
    ):
        """
        Asynchronously adds a step to a dataset.

        Args:
            dataset_id (str): The unique identifier of the dataset.
            step_id (str): The unique identifier of the step to add.
            metadata (Optional[Dict]): Additional metadata for the step being added.

        Returns:
            The result of the GraphQL helper function for adding a step to a dataset.
        """
        return await self.gql_helper(
            *add_step_to_dataset_helper(dataset_id, step_id, metadata)
        )

    async def add_generation_to_dataset(
        self, dataset_id: str, generation_id: str, metadata: Optional[Dict] = None
    ):
        """
        Asynchronously adds a generation to a dataset.

        Args:
            dataset_id (str): The unique identifier of the dataset.
            generation_id (str): The unique identifier of the generation to add.
            metadata (Optional[Dict]): Additional metadata for the generation being added.

        Returns:
            The result of the GraphQL helper function for adding a generation to a dataset.
        """
        return await self.gql_helper(
            *add_generation_to_dataset_helper(dataset_id, generation_id, metadata)
        )

    # Prompt API

    async def get_or_create_prompt_lineage(
        self, name: str, description: Optional[str] = None
    ):
        return await self.gql_helper(*create_prompt_lineage_helper(name, description))

    get_or_create_prompt_lineage.__doc__ = (
        LiteralAPI.get_or_create_prompt_lineage.__doc__
    )

    @deprecated('Please use "get_or_create_prompt_lineage" instead.')
    async def create_prompt_lineage(self, name: str, description: Optional[str] = None):
        return await self.get_or_create_prompt_lineage(name, description)

    async def get_or_create_prompt(
        self,
        name: str,
        template_messages: List[GenerationMessage],
        settings: Optional[ProviderSettings] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Prompt:
        lineage = await self.get_or_create_prompt_lineage(name)
        lineage_id = lineage["id"]

        sync_api = LiteralAPI(self.api_key, self.url)
        return await self.gql_helper(
            *create_prompt_helper(
                sync_api, lineage_id, template_messages, settings, tools
            )
        )

    get_or_create_prompt.__doc__ = LiteralAPI.get_or_create_prompt.__doc__

    @deprecated('Please use "get_or_create_prompt" instead.')
    async def create_prompt(
        self,
        name: str,
        template_messages: List[GenerationMessage],
        settings: Optional[ProviderSettings] = None,
    ):
        return await self.get_or_create_prompt(name, template_messages, settings)

    async def get_prompt(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Prompt:
        sync_api = LiteralAPI(self.api_key, self.url)
        if id:
            return await self.gql_helper(*get_prompt_helper(sync_api, id=id))
        elif name:
            return await self.gql_helper(
                *get_prompt_helper(sync_api, name=name, version=version)
            )
        else:
            raise ValueError("Either the `id` or the `name` must be provided.")

    get_prompt.__doc__ = LiteralAPI.get_prompt.__doc__

    async def promote_prompt(self, name: str, version: int) -> str:
        lineage = await self.get_or_create_prompt_lineage(name)
        lineage_id = lineage["id"]

        return await self.gql_helper(*promote_prompt_helper(lineage_id, version))

    promote_prompt.__doc__ = LiteralAPI.promote_prompt.__doc__

    # Misc API

    async def get_my_project_id(self):
        response = await self.make_rest_call("/my-project", {})
        return response["projectId"]

    get_my_project_id.__doc__ = LiteralAPI.get_my_project_id.__doc__
