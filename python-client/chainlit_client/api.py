import mimetypes
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from chainlit_client.step import Step
from chainlit_client.thread import Thread
from chainlit_client.types import Attachment, Feedback, FeedbackStrategy, User

step_fields = """
        id
        threadId
        parentId
        startTime
        endTime
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
    result = {
        f"id_{id}": event.get("id"),
        f"threadId_{id}": event.get("thread_id"),
        f"startTime_{id}": event.get("start"),
        f"endTime_{id}": event.get("end"),
        f"type_{id}": event.get("type"),
        f"metadata_{id}": event.get("metadata"),
        f"parentId_{id}": event.get("parent_id"),
        f"name_{id}": event.get("name"),
        f"input_{id}": event.get("input"),
        f"output_{id}": event.get("output"),
        f"generation_{id}": event.get("generation"),
        f"feedback_{id}": event.get("feedback"),
        f"attachments_{id}": event.get("attachments"),
    }

    # Remove the keys that are not set
    # When they are None, the API cleans any preexisting value
    for key in list(result):
        if result[key] is None:
            del result[key]

    return result


def variables_builder(steps: List[Union[Dict, "Step"]]):
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
    def __init__(self, api_key=None, endpoint=None):
        self.api_key = api_key
        self.endpoint = endpoint
        if self.api_key is None:
            raise Exception("CHAINLIT_API_KEY not set")
        if self.endpoint is None:
            raise Exception("CHAINLIT_ENDPOINT not set")

    @property
    def headers(self):
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

    async def make_api_call(self, description, query, variables):
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.endpoint,
                    json={"query": query, "variables": variables},
                    headers=self.headers,
                    timeout=10,
                )

                json = response.json()

                if response.status_code != 200:
                    reason = response.text
                    print(f"Failed to {description}: {reason}")
                    return None

                if json.get("errors"):
                    print(f"Failed to {description}: {json['errors']}")
                    return None

                return response.json()
            except Exception as e:
                print(f"Failed to {description}: {e}")
                return None

    # User API

    async def create_user(self, identifier: str, metadata: Optional[Dict] = None):
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
    ):
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
    ):
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
            }
        }"""

        variables = {"id": id, "identifier": identifier}

        result = await self.make_api_call("get user", query, variables)

        return User.from_dict(result["data"]["participant"])

    async def delete_user(self, id: str):
        query = """
        mutation DeleteUser($id: String!) {
            deleteParticipant(id: $id) {
                id
                identifier
                metadata
            }
        }
        """

        variables = {"id": id}

        return await self.make_api_call("delete user", query, variables)

    # Thread API

    async def list_threads(self, first: int = 10, skip: int = 0):
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
        variables = {"first": first, "skip": skip}

        return await self.make_api_call("list threads", query, variables)

    async def create_thread(
        self,
        metadata: Optional[Dict] = None,
        participant_id: Optional[str] = None,
        environment: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ):
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
    ):
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

    async def get_thread(self, id):
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

        return Thread.from_dict(result["data"]["thread"])

    async def delete_thread(self, id: str):
        query = """
        mutation DeleteThread($thread_id: String!) {
            deleteThread(id: $thread_id) {
                id
            }
        }
        """

        variables = {"thread_id": id}

        return await self.make_api_call("delete thread", query, variables)

    # Feedback API

    async def set_feedback(
        self,
        step_id: str,
        value: Optional[int] = None,
        comment: Optional[str] = None,
        feedback_strategy: Optional[FeedbackStrategy] = None,
    ):
        step = await self.get_step(step_id=step_id)

        if step is None:
            raise Exception(f"Step {step_id} not found")

        if step.feedback is None:
            step.feedback = Feedback()

        if value is not None:
            step.feedback.value = value

        if comment is not None:
            step.feedback.comment = comment

        if feedback_strategy is not None:
            step.feedback.strategy = feedback_strategy

        return await self.send_steps([step])

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
                content, attachment.mime, attachment.thread_id
            )

            if uploaded["object_key"] is None or uploaded["url"] is None:
                raise Exception("Failed to upload file")

            attachment.objectKey = uploaded["object_key"]
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

    async def get_attachment(self, id: str):
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

        return Attachment.from_dict(result["data"]["attachment"])

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

    async def create_step(self, step: Union[Dict, "Step"]):
        return await self.send_steps([step])

    async def update_step(self, step: Union[Dict, "Step"]):
        return await self.send_steps([step])

    async def get_step(self, step_id: str):
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
        variables = {"id": step_id}

        result = await self.make_api_call("get step", query, variables)

        return Step.from_dict(result["data"]["step"])

    async def delete_step(self, id: str):
        query = """
        mutation DeleteStep($id: String!) {
            deleteStep(id: $id) {
                id
            }
        }
        """

        variables = {"id": id}

        result = await self.make_api_call("delete step", query, variables)

        return result["data"]["deleteStep"]

    async def send_steps(self, steps: List[Union[Dict, "Step"]]):
        query = query_builder(steps)
        variables = variables_builder(steps)

        return await self.make_api_call("send steps", query, variables)

    # Upload API

    async def upload_file(
        self, content: Union[bytes, str], mime: str, thread_id: str
    ) -> Dict:
        id = str(uuid.uuid4())
        body = {"fileName": id, "contentType": mime, "threadId": thread_id}

        path = "/api/upload/file"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}{path}",
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
