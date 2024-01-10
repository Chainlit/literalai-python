import datetime
import logging
import mimetypes
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypedDict, Union

if TYPE_CHECKING:
    from typing import Tuple  # noqa: F401

import httpx

from literalai.my_types import (
    Attachment,
    Feedback,
    FeedbackStrategy,
    PaginatedResponse,
    PaginatedRestResponse,
    User,
)
from literalai.step import Step, StepDict, StepType
from literalai.thread import Thread, ThreadDict, ThreadFilter

logger = logging.getLogger(__name__)

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
            id
            value
            comment
        }
        tags
        generation {
            type
            provider
            settings
            inputs
            completion
            templateFormat
            template
            formatted
            messages
            tokenCount
        }
        name
        attachments {
            id
            stepId
            metadata
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
        createdAt
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

shallow_thread_fields = """
        id
        metadata
        tags
        createdAt
        participant {
            id
            identifier
            metadata
        }    
        
"""


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

        if url and url[-1] == "/":
            url = url[:-1]

        self.url = url

        if self.api_key is None:
            raise Exception("LITERAL_API_KEY not set")
        if self.url is None:
            raise Exception("LITERAL_API_URL not set")

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

    async def make_api_call(
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

            return json

        # This should not be reached, exceptions should be thrown beforehands
        # Added because of mypy
        raise Exception("Unkown error")

    async def make_rest_api_call(self, subpath: str, body: Dict[str, Any]) -> Dict:
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
        filters: Optional[ThreadFilter] = None,
    ) -> PaginatedResponse:
        query = (
            """
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
"""
            + shallow_thread_fields
            + """
                    }
                }
            }
        }
    """
        )
        variables: Dict[str, Any] = {}

        if first:
            variables["first"] = first
        if after:
            variables["after"] = after
        if filters:
            variables["filters"] = filters.to_dict()

        result = await self.make_api_call("list threads", query, variables)

        response = result["data"]["threads"]

        response["data"] = list(map(lambda x: x["node"], response["edges"]))
        del response["edges"]

        return PaginatedResponse[Thread].from_dict(response, Thread)

    async def export_threads(
        self,
        page: Optional[int] = None,
        filters: Optional[ThreadFilter] = None,
        cursor_anchor: Optional[str] = None,
    ) -> PaginatedRestResponse[ThreadDict]:
        body: Dict[str, Any] = {}

        if cursor_anchor:
            body["cursorAnchor"] = cursor_anchor

        if page:
            body["page"] = page

        if filters:
            body["filters"] = filters.to_dict()

        result = await self.make_rest_api_call(subpath="/export/threads", body=body)

        return PaginatedRestResponse(**result)

    async def create_thread(
        self,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Thread:
        query = (
            """
        mutation CreateThread(
            $metadata: Json,
            $participantId: String,
            $environment: String,
            $tags: [String!],
        ) {
            createThread(
                metadata: $metadata
                participantId: $participantId
                environment: $environment
                tags: $tags
            ) {
"""
            + shallow_thread_fields
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
        }

        thread = await self.make_api_call("create thread", query, variables)

        return Thread.from_dict(thread["data"]["createThread"])

    async def upsert_thread(
        self,
        thread_id: str,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Thread:
        query = (
            """
        mutation UpsertThread(
            $id: String!,
            $metadata: Json,
            $participantId: String,
            $environment: String,
            $tags: [String!],
        ) {
            upsertThread(
                id: $id
                metadata: $metadata
                participantId: $participantId
                environment: $environment
                tags: $tags
            ) {
"""
            + shallow_thread_fields
            + """
            }
        }
"""
        )
        variables = {
            "id": thread_id,
            "metadata": metadata,
            "participantId": participant_id,
            "environment": environment,
            "tags": tags,
        }

        # remove None values to prevent the API from removing existing values
        variables = {k: v for k, v in variables.items() if v is not None}

        thread = await self.make_api_call("upsert thread", query, variables)

        return Thread.from_dict(thread["data"]["upsertThread"])

    async def update_thread(
        self,
        id: str,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Thread:
        query = (
            """
        mutation UpdateThread(
            $id: String!,
            $metadata: Json,
            $participantId: String,
            $environment: String,
            $tags: [String!],
        ) {
            updateThread(
                id: $id
                metadata: $metadata
                participantId: $participantId
                environment: $environment
                tags: $tags
            ) {
"""
            + shallow_thread_fields
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
        }

        # remove None values to prevent the API from removing existing values
        variables = {k: v for k, v in variables.items() if v is not None}

        thread = await self.make_api_call("update thread", query, variables)

        return Thread.from_dict(thread["data"]["updateThread"])

    async def get_thread(self, id: str) -> Optional[Thread]:
        query = (
            """
        query GetThread($id: String!) {
            threadDetail(id: $id) {
"""
            + thread_fields
            + """
            }
        }
    """
        )

        variables = {"id": id}

        result = await self.make_api_call("get thread", query, variables)

        thread = result["data"]["threadDetail"]

        return Thread.from_dict(thread) if thread else None

    async def delete_thread(self, id: str) -> bool:
        query = """
        mutation DeleteThread($thread_id: String!) {
            deleteThread(id: $thread_id) {
                id
            }
        }
        """

        variables = {"thread_id": id}

        result = await self.make_api_call("delete thread", query, variables)
        deleted = bool(result["data"]["deleteThread"])
        return deleted

    # User Session API

    async def create_user_session(
        self,
        id: Optional[str] = None,
        started_at: Optional[str] = None,
        is_interactive: Optional[bool] = None,
        ended_at: Optional[str] = None,
        anon_participant_identifier: Optional[str] = None,
        participant_identifier: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        query = """
        mutation CreateParticipantSession(
            $id: String, 
            $isInteractive: Boolean, 
            $startedAt: DateTime!, 
            $endedAt: DateTime, 
            $anonParticipantIdentifier: String, 
            $participantIdentifier: String, 
            $metadata: Json, 
        ) {
            createParticipantSession(
                id: $id, 
                isInteractive: $isInteractive, 
                startedAt: $startedAt, 
                endedAt: $endedAt, 
                anonParticipantIdentifier: $anonParticipantIdentifier, 
                participantIdentifier: $participantIdentifier, 
                metadata: $metadata, 
            ) {
                id
                isInteractive
                startedAt
                endedAt
                anonParticipantIdentifier
                participantIdentifier
                metadata
            }
        }
        """

        variables = {
            "id": id,
            "isInteractive": is_interactive,
            "startedAt": started_at or datetime.datetime.utcnow().isoformat(),
            "endedAt": ended_at if ended_at else None,
            "anonParticipantIdentifier": anon_participant_identifier,
            "participantIdentifier": participant_identifier,
            "metadata": metadata,
        }

        participant_session = await self.make_api_call(
            "create participant session", query, variables
        )

        return participant_session["data"]["createParticipantSession"]

    async def update_user_session(
        self,
        id: str,
        is_interactive: Optional[bool] = None,
        ended_at: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        query = """
        mutation UpdateParticipantSession(
            $id: String!,
            $isInteractive: Boolean,
            $endedAt: DateTime,
            $metadata: Json,
        ) {
            updateParticipantSession(
                id: $id,
                isInteractive: $isInteractive,
                endedAt: $endedAt,
                metadata: $metadata,
            ) {
                id
                isInteractive
                endedAt
                metadata
            }
        }
        """
        variables = {
            "id": id,
            "isInteractive": is_interactive,
            "endedAt": ended_at,
            "metadata": metadata,
        }

        # remove None values to prevent the API from removing existing values
        variables = {k: v for k, v in variables.items() if v is not None}

        session = await self.make_api_call(
            "update participant session", query, variables
        )

        return session["data"]["updateParticipantSession"]

    async def get_user_session(self, id: str) -> Optional[Dict]:
        query = """
        query GetParticipantSession($id: String!) {
            participantSession(id: $id) {
                id
                isInteractive
                startedAt
                endedAt
                anonParticipantIdentifier
                participantIdentifier
                metadata
            }
        }"""

        variables = {"id": id}

        result = await self.make_api_call("get user session", query, variables)

        user_session = result["data"]["participantSession"]

        return user_session

    async def delete_user_session(self, id: str) -> str:
        query = """
        mutation DeleteParticipantSession($id: String!) {
            deleteParticipantSession(id: $id) {
                id
            }
        }
        """

        variables = {"id": id}

        result = await self.make_api_call(
            "delete participant session", query, variables
        )

        return result["data"]["deleteParticipantSession"]["id"]

    # Feedback API

    async def create_feedback(
        self,
        step_id: str,
        value: int,
        comment: Optional[str] = None,
        strategy: Optional[FeedbackStrategy] = None,
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

    class FeedbackUpdate(TypedDict, total=False):
        comment: Optional[str]
        value: Optional[int]
        strategy: Optional[FeedbackStrategy]

    async def update_feedback(
        self,
        id: str,
        update_params: FeedbackUpdate,
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
        variables = {"id": id, **update_params}
        result = await self.make_api_call("update feedback", query, variables)

        return Feedback.from_dict(result["data"]["updateFeedback"])

    # Attachment API

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
        if not content and not url and not path:
            raise Exception("Either content, path or attachment url must be provided")

        if content and path:
            raise Exception("Only one of content and path must be provided")

        if (content and url) or (path and url):
            raise Exception(
                "Only one of content, path and attachment url must be provided"
            )

        if path:
            # TODO: if attachment.mime is text, we could read as text?
            with open(path, "rb") as f:
                content = f.read()
            if not name:
                name = path.split("/")[-1]
            if not mime:
                mime, _ = mimetypes.guess_type(path)
                mime = mime or "application/octet-stream"

        if not name:
            raise Exception("Attachment name must be provided")

        if content:
            uploaded = await self.upload_file(
                content=content, thread_id=thread_id, mime=mime
            )

            if uploaded["object_key"] is None or uploaded["url"] is None:
                raise Exception("Failed to upload file")

            object_key = uploaded["object_key"]
            url = None
            if not object_key:
                url = uploaded["url"]

        query = """
        mutation CreateAttachment(
            $metadata: Json,
            $mime: String,
            $name: String!,
            $objectKey: String,
            $stepId: String!,
            $url: String,
        ) {
            createAttachment(
                metadata: $metadata,
                mime: $mime,
                name: $name,
                objectKey: $objectKey,
                stepId: $stepId,
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
            "metadata": metadata,
            "mime": mime,
            "name": name,
            "objectKey": object_key,
            "stepId": step_id,
            "threadId": thread_id,
            "url": url,
            "id": id,
        }

        result = await self.make_api_call("create attachment", query, variables)

        return Attachment.from_dict(result["data"]["createAttachment"])

    class AttachmentUpload(TypedDict, total=False):
        metadata: Optional[Dict]
        name: Optional[str]
        mime: Optional[str]
        objectKey: Optional[str]
        url: Optional[str]

    async def update_attachment(
        self,
        id: str,
        update_params: AttachmentUpload,
    ) -> "Attachment":
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
        variables = {"id": id, **update_params}
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
        type: Optional[StepType] = "undefined",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        input: Optional[str] = None,
        output: Optional[str] = None,
        metadata: Optional[Dict] = None,
        parent_id: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "Step":
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
        input: Optional[str] = None,
        output: Optional[str] = None,
        metadata: Optional[Dict] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> "Step":
        query = (
            """
        mutation UpdateStep(
            $id: String!,
            $type: StepType,
            $input: String,
            $output: String,
            $metadata: Json,
            $name: String,
            $startTime: DateTime,
            $endTime: DateTime,
            $parentId: String,
        ) {
            updateStep(
                id: $id,
                type: $type,
                startTime: $startTime,
                endTime: $endTime,
                input: $input,
                output: $output,
                metadata: $metadata,
                name: $name,
                parentId: $parentId,
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
            "input": input,
            "output": output,
            "metadata": metadata,
            "name": name,
            "tags": tags,
            "startTime": start_time,
            "endTime": end_time,
            "parentId": parent_id,
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

    async def delete_step(self, id: str) -> bool:
        query = """
        mutation DeleteStep($id: String!) {
            deleteStep(id: $id) {
                id
            }
        }
        """

        variables = {"id": id}

        result = await self.make_api_call("delete step", query, variables)

        deleted = bool(result["data"]["deleteStep"])
        return deleted

    async def send_steps(self, steps: List[Union[StepDict, "Step"]]) -> "Dict":
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
