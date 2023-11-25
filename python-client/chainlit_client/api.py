import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from chainlit_client.step import Step
from chainlit_client.thread import Thread


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

    async def send_steps(self, steps: List[Union[Dict, "Step"]]):
        query = query_builder(steps)
        variables = variables_builder(steps)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.endpoint,
                    json={"query": query, "variables": variables},
                    headers=self.headers,
                    timeout=10,
                )

                return response.json()
            except Exception as e:
                print(f"Failed to send steps: {e}")
                return None

    async def get_thread(self, id):
        query = """
        query GetSteps($id: String!) {
            thread(id: $id) {
                id
                steps {
                    id
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
                    }
                }
            }
        }
    """
        variables = {"id": id}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.endpoint,
                    json={"query": query, "variables": variables},
                    headers=self.headers,
                    timeout=10,
                )

                return Thread.from_dict(response.json()["data"]["thread"])
            except Exception as e:
                print(f"Failed to get thread: {e}")
                return None

    async def upload_file(
        self, content: Union[bytes, str], mime: str, thread_id: Optional[str]
    ) -> Dict:
        id = str(uuid.uuid4())
        body = {"fileName": id, "contentType": mime}

        if thread_id:
            body["threadId"] = thread_id

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