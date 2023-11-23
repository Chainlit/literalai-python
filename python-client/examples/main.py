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

thread_id = None


@sdk.run()
def get_completion(welcome_message, text):
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


@sdk.thread
def run():
    global thread_id
    thread_id = sdk.get_current_thread_id()
    welcome_message = "What's your name? "
    with sdk.step(type=StepType.MESSAGE, role=StepRole.SYSTEM) as step:
        step.output = welcome_message

    text = input(welcome_message)

    with sdk.step(type=StepType.MESSAGE, role=StepRole.USER) as step:
        step.output = text

        completion = get_completion(welcome_message=welcome_message, text=text)

        with sdk.step(type=StepType.MESSAGE, role=StepRole.ASSISTANT) as step:
            print("")
            print(completion)
            step.output = completion


run()
sdk.wait_until_queue_empty()


# Get the steps from the API for the demo
async def main():
    print("\nSearching for the thread", thread_id, "...")
    steps = await sdk.api.get_thread(id=thread_id)

    thread = steps

    print(json.dumps(thread, indent=2))


asyncio.run(main())
