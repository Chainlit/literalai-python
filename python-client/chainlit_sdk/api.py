import httpx
import os


def step_type(type):
    if type == "message":
        return "MESSAGE"
    if type == "run":
        return "RUN"
    if type == "llm":
        return "LLM"
    # default is run type
    return "RUN"


def serialize_step(event, id):
    metadata = {}
    # return {
    #     f"id_{id}": event.get("id"),
    #     f"traceId_{id}": event.get("thread_id"),
    #     f"startTime_{id}": event.get("start"),
    #     f"endTime_{id}": event.get("end"),
    #     f"operation_{id}": step_type(event.get("type")),
    #     f"metadata_{id}": metadata,
    # }
    return {
        f"id": event.get("id"),
        f"traceId": event.get("thread_id"),
        f"startTime": event.get("start"),
        f"endTime": event.get("end"),
        f"operation": step_type(event.get("type")),
        f"metadata": metadata,
    }


def variables_builder(steps):
    # variables = {}
    # for i in range(len(steps)):
    #     variables.update(serialize_step(steps[i], i))
    # return variables
    variables = []
    for i in range(len(steps)):
        variables.append(serialize_step(steps[i], i))
    return variables


def query_variables_builder(steps):
    generated = ""
    for id in range(len(steps)):
        generated += f"""$id_{id}: String!
        $traceId_{id}: String!
        $operation_{id}: StepOperation
        $startTime_{id}: DateTime
        $endTime_{id}: DateTime
        $input_{id}: String
        $output_{id}:String
        $metadata_{id}: Json
        """
    return generated


def ingest_steps_builder(steps):
    generated = ""
    for id in range(len(steps)):
        generated += f"""
      step{id}: ingestStep(
        id: $id_{id}
        traceId: $traceId_{id}
        startTime: $startTime_{id}
        endTime: $endTime_{id}
        operation: $operation_{id}
        input: $input_{id}
        output: $output_{id}
        metadata: $metadata_{id}
      ) {{
        ok
        message
      }}
"""
    return generated


def query_builder(steps):
    # return f"""
    # mutation AddStep({query_variables_builder(steps)}) {{
    #   {ingest_steps_builder(steps)}
    # }}
    # """
    return """
mutation addSteps($steps: [StepInput!]!) {
    ingestSteps(steps: $steps) {
        ok
        message
    }
}
"""


class API:
    def __init__(self):
        self.api_key = os.getenv("SDK_API_KEY")
        self.endpoint = os.getenv("SDK_ENDPOINT", "http://localhost:3000/graphql")
        if self.api_key is None:
            raise Exception("SDK_API_KEY not set")

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
                )

                print(response)

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
                )

                return response.json()
            except Exception as e:
                print(e)
                return None
