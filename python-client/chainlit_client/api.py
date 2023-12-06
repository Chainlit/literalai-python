import mimetypes
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from .step import Step, StepDict, StepType
from .thread import Thread
from .types import (
    Attachment,
    BaseGeneration,
    Feedback,
    FeedbackStrategy,
    PaginatedResponse,
    User,
)

step_fields = """
        id
        threadId
        parentId
        startTime
        endTime
        createdAt
        type
        input
        output
        metadata
        feedback {
            value
            comment
        }
        tags
        generation {
            type
            provider
            settings
        }
        name
        attachments {
            id
            mime
            name
            objectKey
            url
        }"""

thread_fields = (
    """
        id
        metadata
        tags
        participant {
            id
            identifier
            metadata
        }
        steps {
"""
    + step_fields
    + """
        }"""
)


def serialize_step(event, id):
    result = {}

    for key, value in event.items():
        # Only keep the keys that are not None to avoid overriding existing values
        if value is not None:
            result[f"{key}_{id}"] = value

    return result


def variables_builder(steps: List[Union[StepDict, "Step"]]):
    variables = {}
    for i in range(len(steps)):
        step = steps[i]
        if isinstance(step, Step):
            variables.update(serialize_step(step.to_dict(), i))
        else:
            variables.update(serialize_step(step, i))
    return variables


def query_variables_builder(steps):
    generated = ""
    for id in range(len(steps)):
        generated += f"""$id_{id}: String!
        $threadId_{id}: String!
        $type_{id}: StepType
        $startTime_{id}: DateTime
        $endTime_{id}: DateTime
        $input_{id}: String
        $output_{id}:String
        $metadata_{id}: Json
        $parentId_{id}: String
        $name_{id}: String
        $generation_{id}: GenerationPayloadInput
        $feedback_{id}: FeedbackPayloadInput
        $attachments_{id}: [AttachmentPayloadInput!]
        """
    return generated


def ingest_steps_builder(steps):
    generated = ""
    for id in range(len(steps)):
        generated += f"""
      step{id}: ingestStep(
        id: $id_{id}
        threadId: $threadId_{id}
        startTime: $startTime_{id}
        endTime: $endTime_{id}
        type: $type_{id}
        input: $input_{id}
        output: $output_{id}
        metadata: $metadata_{id}
        parentId: $parentId_{id}
        name: $name_{id}
        generation: $generation_{id}
        feedback: $feedback_{id}
        attachments: $attachments_{id}
      ) {{
        ok
        message
      }}
"""
    return generated


def query_builder(steps):
    return f"""
    mutation AddStep({query_variables_builder(steps)}) {{
      {ingest_steps_builder(steps)}
    }}
    """


class API:
    def __init__(self, api_key=None, url=None):
        self.api_key = api_key
        self.url = url
        self.graphql_endpoint = url + "/api/graphql"

        if self.api_key is None:
            raise Exception("CHAINLIT_API_KEY not set")
        if self.url is None:
            raise Exception("CHAINLIT_API_URL not set")

    @property
    def headers(self):
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

    async def make_api_call(
        self, description: str, query: str, variables: Dict[str, Any]
    ) -> Dict:
        def raise_error(error):
            print(f"Failed to {description}: {error}")
            raise error

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.graphql_endpoint,
                    json={"query": query, "variables": variables},
                    headers=self.headers,
                    timeout=10,
                )

                json = response.json()

                if response.status_code != 200:
                    raise_error(response.text)

                if json.get("errors"):
                    raise_error(json["errors"])

                return response.json()
            except Exception as error:
                raise_error(error)

        # This should not be reached, exceptions should be thrown beforehands
        # Added because of mypy
        raise Exception("Unkown error")

    # User API

    async def create_user(
        self, identifier: str, metadata: Optional[Dict] = None
    ) -> User:
        query = """
        mutation CreateUser($identifier: String!, $metadata: Json) {
            createParticipant(identifier: $identifier, metadata: $metadata) {
                id
                identifier
                metadata
            }
        }
        """

        variables = {"identifier": identifier, "metadata": metadata}

        user = await self.make_api_call("create user", query, variables)

        return User.from_dict(user["data"]["createParticipant"])

    async def update_user(
        self, id: str, identifier: Optional[str] = None, metadata: Optional[Dict] = None
    ) -> User:
        query = """
        mutation UpdateUser(
            $id: String!,
            $identifier: String,
            $metadata: Json,
        ) {
            updateParticipant(
                id: $id,
                identifier: $identifier,
                metadata: $metadata
            ) {
                id
                identifier
                metadata
            }
        }
"""
        variables = {"id": id, "identifier": identifier, "metadata": metadata}

        # remove None values to prevent the API from removing existing values
        variables = {k: v for k, v in variables.items() if v is not None}

        user = await self.make_api_call("update user", query, variables)

        return User.from_dict(user["data"]["updateParticipant"])

    async def get_user(
        self, id: Optional[str] = None, identifier: Optional[str] = None
    ) -> Optional[User]:
        if id is None and identifier is None:
            raise Exception("Either id or identifier must be provided")

        if id is not None and identifier is not None:
            raise Exception("Only one of id or identifier must be provided")

        query = """
        query GetUser($id: String, $identifier: String) {
            participant(id: $id, identifier: $identifier) {
                id
                identifier
                metadata
                createdAt
            }
        }"""

        variables = {"id": id, "identifier": identifier}

        result = await self.make_api_call("get user", query, variables)

        user = result["data"]["participant"]

        return User.from_dict(user) if user else None

    async def delete_user(self, id: str) -> str:
        query = """
        mutation DeleteUser($id: String!) {
            deleteParticipant(id: $id) {
                id
            }
        }
        """

        variables = {"id": id}

        result = await self.make_api_call("delete user", query, variables)

        return result["data"]["deleteParticipant"]["id"]

    # Thread API

    async def list_threads(
        self,
        first: Optional[int] = None,
        after: Optional[str] = None,
        filters: Optional[str] = None,
    ) -> PaginatedResponse:
        query = """
        query GetThreads(
            $after: ID,
            $before: ID,
            $cursorAnchor: DateTime,
            $filters: ThreadFiltersInput,
            $first: Int,
            $last: Int,
            $projectId: String,
            $skip: Int
            ) {
            threads(
                after: $after,
                before: $before,
                cursorAnchor: $cursorAnchor,
                filters: $filters,
                first: $first,
                last: $last,
                projectId: $projectId,
                skip: $skip
                ) {
                pageInfo {
                    startCursor
                    endCursor
                    hasNextPage
                    hasPreviousPage
                }
                totalCount
                edges {
                    cursor
                    node {
                        id
                        metadata
                        projectId
                        startTime
                        endTime
                        tags
                        tokenCount
                        environment
                    }
                }
            }
        }
    """
        variables: Dict[str, Any] = {}

        if first:
            variables["first"] = first
        if after:
            variables["after"] = after

        result = await self.make_api_call("list threads", query, variables)

        response = result["data"]["threads"]

        response["data"] = list(map(lambda x: x["node"], response["edges"]))
        del response["edges"]

        return PaginatedResponse[Thread].from_dict(response, Thread)

    async def create_thread(
        self,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Thread:
        query = (
            """
        mutation CreateThread(
            $metadata: Json,
            $participantId: String,
            $environment: String,
            $tags: [String!],
            $startTime: DateTime,
            $endTime: DateTime,
        ) {
            createThread(
                metadata: $metadata
                participantId: $participantId
                environment: $environment
                tags: $tags
                startTime: $startTime
                endTime: $endTime
            ) {
"""
            + thread_fields
            + """
            }
        }
"""
        )
        variables = {
            "metadata": metadata,
            "participantId": participant_id,
            "environment": environment,
            "tags": tags,
            "startTime": start_time,
            "endTime": end_time,
        }

        thread = await self.make_api_call("create thread", query, variables)

        return Thread.from_dict(thread["data"]["createThread"])

    async def update_thread(
        self,
        id: str,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Thread:
        query = (
            """
        mutation UpdateThread(
            $id: String!,
            $metadata: Json,
            $participantId: String,
            $environment: String,
            $tags: [String!],
            $startTime: DateTime,
            $endTime: DateTime,
        ) {
            updateThread(
                id: $id
                metadata: $metadata
                participantId: $participantId
                environment: $environment
                tags: $tags
                startTime: $startTime
                endTime: $endTime
            ) {
"""
            + thread_fields
            + """
            }
        }
"""
        )
        variables = {
            "id": id,
            "metadata": metadata,
            "participantId": participant_id,
            "environment": environment,
            "tags": tags,
            "startTime": start_time,
            "endTime": end_time,
        }

        # remove None values to prevent the API from removing existing values
        variables = {k: v for k, v in variables.items() if v is not None}

        thread = await self.make_api_call("update thread", query, variables)

        return Thread.from_dict(thread["data"]["updateThread"])

    async def get_thread(self, id: str) -> Optional[Thread]:
        query = (
            """
        query GetThread($id: String!) {
            thread(id: $id) {
"""
            + thread_fields
            + """
            }
        }
    """
        )

        variables = {"id": id}

        result = await self.make_api_call("get thread", query, variables)

        thread = result["data"]["thread"]

        return Thread.from_dict(thread) if thread else None

    async def delete_thread(self, id: str) -> str:
        query = """
        mutation DeleteThread($thread_id: String!) {
            deleteThread(id: $thread_id) {
                id
            }
        }
        """

        variables = {"thread_id": id}

        result = await self.make_api_call("delete thread", query, variables)

        return result["data"]["deleteThread"]["id"]

    # Feedback API

    async def create_feedback(
        self,
        step_id: str,
        value: int,
        comment: Optional[str],
        strategy: Optional[FeedbackStrategy],
    ) -> "Feedback":
        query = """
        mutation CreateFeedback(
            $comment: String,
            $stepId: String!,
            $strategy: FeedbackStrategy,
            $value: Int!,
        ) {
            createFeedback(
                comment: $comment,
                stepId: $stepId,
                strategy: $strategy,
                value: $value,
            ) {
                id
                threadId
                stepId
                value
                comment
                strategy
            }
        }
        """

        variables = {
            "comment": comment,
            "stepId": step_id,
            "strategy": strategy,
            "value": value,
        }

        result = await self.make_api_call("create feedback", query, variables)

        return Feedback.from_dict(result["data"]["createFeedback"])

    async def update_feedback(
        self,
        id: str,
        value: int,
        comment: Optional[str],
        strategy: Optional["FeedbackStrategy"],
    ) -> "Feedback":
        query = """
        mutation UpdateFeedback(
            $id: String!,
            $comment: String,
            $value: Int,
            $strategy: FeedbackStrategy,
        ) {
            updateFeedback(
                id: $id,
                comment: $comment,
                value: $value,
                strategy: $strategy,
            ) {
                id
                threadId
                stepId
                value
                comment
                strategy
            }
        }
        """

        variables = {"id": id, "comment": comment, "value": value, "strategy": strategy}

        # remove None values to prevent the API from removing existing values
        variables = {k: v for k, v in variables.items() if v is not None}

        result = await self.make_api_call("update feedback", query, variables)

        return Feedback.from_dict(result["data"]["updateFeedback"])

    # Attachment API

    async def create_attachment(
        self,
        attachment: "Attachment",
        content: Optional[Union[bytes, str]] = None,
        path: Optional[str] = None,
    ):
        if not content and not attachment.url and not path:
            raise Exception("Either content, path or attachment url must be provided")

        if content and path:
            raise Exception("Only one of content and path must be provided")

        if (content and attachment.url) or (path and attachment.url):
            raise Exception(
                "Only one of content, path and attachment url must be provided"
            )

        if path:
            # TODO: if attachment.mime is text, we could read as text?
            with open(path, "rb") as f:
                content = f.read()
            if not attachment.name:
                attachment.name = path.split("/")[-1]
            if not attachment.mime:
                mime, _ = mimetypes.guess_type(path)
                attachment.mime = mime or "application/octet-stream"

        if content:
            uploaded = await self.upload_file(
                content=content, thread_id=attachment.thread_id, mime=attachment.mime
            )

            if uploaded["object_key"] is None or uploaded["url"] is None:
                raise Exception("Failed to upload file")

            attachment.object_key = uploaded["object_key"]
            attachment.url = uploaded["url"]

        query = """
        mutation CreateAttachment(
            $metadata: Json,
            $mime: String,
            $name: String!,
            $objectKey: String,
            $stepId: String!,
            $threadId: String!,
            $url: String,
        ) {
            createAttachment(
                metadata: $metadata,
                mime: $mime,
                name: $name,
                objectKey: $objectKey,
                stepId: $stepId,
                threadId: $threadId,
                url: $url,
            ) {
                id
                threadId
                stepId
                metadata
                mime
                name
                objectKey
                url
            }
        }
        
        """

        variables = attachment.to_dict()

        result = await self.make_api_call("create attachment", query, variables)

        return Attachment.from_dict(result["data"]["createAttachment"])

    async def update_attachment(
        self,
        id: str,
        metadata: Optional[Dict] = None,
        name: Optional[str] = None,
        mime: Optional[str] = None,
        objectKey: Optional[str] = None,
        url: Optional[str] = None,
    ):
        query = """
        mutation UpdateAttachment(
            $id: String!,
            $metadata: Json,
            $mime: String,
            $name: String,
            $objectKey: String,
            $projectId: String,
            $url: String,
        ) {
            updateAttachment(
                id: $id,
                metadata: $metadata,
                mime: $mime,
                name: $name,
                objectKey: $objectKey,
                projectId: $projectId,
                url: $url,
            ) {
                id
                threadId
                stepId
                metadata
                mime
                name
                objectKey
                url
            }
        }
        """

        variables = {
            "id": id,
            "metadata": metadata,
            "mime": mime,
            "name": name,
            "objectKey": objectKey,
            "url": url,
        }

        result = await self.make_api_call("update attachment", query, variables)

        return Attachment.from_dict(result["data"]["updateAttachment"])

    async def get_attachment(self, id: str) -> Optional[Attachment]:
        query = """
        query GetAttachment($id: String!) {
            attachment(id: $id) {
                id
                threadId
                stepId
                metadata
                mime
                name
                objectKey
                url
            }
        }
        """

        variables = {"id": id}

        result = await self.make_api_call("get attachment", query, variables)

        attachment = result["data"]["attachment"]

        return Attachment.from_dict(attachment) if attachment else None

    async def delete_attachment(self, id: str):
        query = """
        mutation DeleteAttachment($id: String!) {
            deleteAttachment(id: $id) {
                id
            }
        }
        """

        variables = {"id": id}

        result = await self.make_api_call("delete attachment", query, variables)

        return result["data"]["deleteAttachment"]

    # Step API

    async def create_step(
        self,
        thread_id: str,
        type: Optional[StepType] = "UNDEFINED",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        input: Optional[str] = None,
        output: Optional[str] = None,
        metadata: Optional[Dict] = None,
        parent_id: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Step:
        query = (
            """
        mutation CreateStep(
            $threadId: String!,
            $type: StepType,
            $startTime: DateTime,
            $endTime: DateTime,
            $input: String,
            $output: String,
            $metadata: Json,
            $parentId: String,
            $name: String,
        ) {
            createStep(
                threadId: $threadId,
                type: $type,
                startTime: $startTime,
                endTime: $endTime,
                input: $input,
                output: $output,
                metadata: $metadata,
                parentId: $parentId,
                name: $name,
            ) {
"""
            + step_fields
            + """
            }
        }
        """
        )

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

        result = await self.make_api_call("create step", query, variables)

        return Step.from_dict(result["data"]["createStep"])

    async def update_step(
        self,
        id: str,
        type: Optional[StepType] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        input: Optional[str] = None,
        output: Optional[str] = None,
        metadata: Optional[Dict] = None,
        parent_id: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        generation: Optional[Dict] = None,
        feedback: Optional[Feedback] = None,
        attachments: Optional[List[Attachment]] = None,
    ) -> Step:
        query = (
            """
        mutation UpdateStep(
            $id: String!,
            $type: StepType,
            $startTime: DateTime,
            $endTime: DateTime,
            $input: String,
            $output: String,
            $metadata: Json,
            $parentId: String,
            $name: String,
            $generation: GenerationPayloadInput,
            $feedback: FeedbackPayloadInput,
            $attachments: [AttachmentPayloadInput!],
        ) {
            updateStep(
                id: $id,
                type: $type,
                startTime: $startTime,
                endTime: $endTime,
                input: $input,
                output: $output,
                metadata: $metadata,
                parentId: $parentId,
                name: $name,
                generation: $generation,
                feedback: $feedback,
                attachments: $attachments,
            ) {    
"""
            + step_fields
            + """
            }
        }
        """
        )

        variables = {
            "id": id,
            "type": type,
            "startTime": start_time,
            "endTime": end_time,
            "input": input,
            "output": output,
            "metadata": metadata,
            "parentId": parent_id,
            "name": name,
            "tags": tags,
            "generation": generation,
            "feedback": feedback,
            "attachments": attachments,
        }

        result = await self.make_api_call("update step", query, variables)

        return Step.from_dict(result["data"]["updateStep"])

    async def get_step(self, id: str) -> Optional[Step]:
        query = (
            """
        query GetStep($id: String!) {
            step(id: $id) {"""
            + step_fields
            + """
            }
        }
    """
        )
        variables = {"id": id}

        result = await self.make_api_call("get step", query, variables)

        step = result["data"]["step"]

        return Step.from_dict(step) if step else None

    async def delete_step(self, id: str) -> str:
        query = """
        mutation DeleteStep($id: String!) {
            deleteStep(id: $id) {
                id
            }
        }
        """

        variables = {"id": id}

        result = await self.make_api_call("delete step", query, variables)

        return result["data"]["deleteStep"]["id"]

    async def send_steps(self, steps: List[Union[StepDict, "Step"]]) -> Dict:
        query = query_builder(steps)
        variables = variables_builder(steps)

        return await self.make_api_call("send steps", query, variables)

    # Upload API

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
            if response.status_code != 200:
                reason = response.text
                print(f"Failed to sign upload url: {reason}")
                return {"object_key": None, "url": None}
            json_res = response.json()

        upload_details = json_res["post"]
        object_key = upload_details["fields"]["key"]
        signed_url = json_res["signedUrl"]

        # Prepare form data
        form_data = {}  # type: Dict[str, Tuple[Union[str, None], Any]]
        for field_name, field_value in upload_details["fields"].items():
            form_data[field_name] = (None, field_value)

        # Add file to the form_data
        # Note: The content_type parameter is not needed here, as the correct MIME type should be set in the 'Content-Type' field from upload_details
        form_data["file"] = (id, content)

        async with httpx.AsyncClient() as client:
            upload_response = await client.post(
                upload_details["url"],
                files=form_data,
            )
            try:
                upload_response.raise_for_status()
                url = f'{upload_details["url"]}/{object_key}'
                return {"object_key": object_key, "url": signed_url}
            except Exception as e:
                print(f"Failed to upload file: {str(e)}")
                return {"object_key": None, "url": None}
