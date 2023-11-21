import httpx


def serialize_step(event, id):
    return {
        f"id_{id}": event.get("id"),
        f"threadId_{id}": event.get("thread_id"),
        f"startTime_{id}": event.get("start"),
        f"endTime_{id}": event.get("end"),
        f"type_{id}": event.get("type"),
        f"metadata_{id}": event.get("metadata"),
        f"parentId_{id}": event.get("parent_id"),
        f"operatorName_{id}": event.get("name"),
        f"input_{id}": event.get("input"),
        f"output_{id}": event.get("output"),
        f"generation_{id}": event.get("generation"),
        f"operatorRole_{id}": event.get("operatorRole"),
    }


def variables_builder(steps):
    variables = {}
    for i in range(len(steps)):
        variables.update(serialize_step(steps[i], i))
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
        $operatorName_{id}: String
        $generation_{id}: GenerationPayloadInput
        $operatorRole_{id}: OperatorRole
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
        operatorName: $operatorName_{id}
        generation: $generation_{id}
        operatorRole: $operatorRole_{id}
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

    async def send_steps(self, steps):
        query = query_builder(steps)
        variables = variables_builder(steps)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.endpoint,
                    json={"query": query, "variables": variables},
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key,
                    },
                    timeout=10,
                )

                return response.json()
            except Exception as e:
                print(e)
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
                    operatorName
                    operatorRole

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
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key,
                    },
                    timeout=10,
                )

                return response.json()
            except Exception as e:
                print(e)
                return None
