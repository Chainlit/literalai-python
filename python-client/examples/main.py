import asyncio
import json
import uuid

from chainlit_sdk import Chainlit
from chainlit_sdk.types import StepRole, StepType
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

sdk = Chainlit(batch_size=2)
sdk.instrument_openai()

thread_id = uuid.uuid4()


@sdk.run(thread_id=thread_id)
def get_completion(input):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Tell an inspiring quote to the user, mentioning their name. Be extremely supportive while keeping it short. Write one sentence per line.",
            },
            {
                "role": "assistant",
                "content": welcome_message,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return completion.choices[0].message.content


welcome_message = "What's your name? "
with sdk.step(type=StepType.MESSAGE, thread_id=thread_id, role=StepRole.SYSTEM) as step:
    step.output = welcome_message

text = input(welcome_message)

with sdk.step(type=StepType.MESSAGE, thread_id=thread_id, role=StepRole.USER) as step:
    step.output = text

    completion = get_completion(text)

    with sdk.step(type=StepType.MESSAGE, role=StepRole.ASSISTANT) as step:
        print("")
        print(completion)
        step.output = completion

sdk.wait_until_queue_empty()


# Get the steps from the API for the demo
async def main():
    print("\nSearching for the thread", thread_id, "...")
    steps = await sdk.api.get_thread(id=str(thread_id))

    thread = steps

    print(json.dumps(thread, indent=2))


asyncio.run(main())
