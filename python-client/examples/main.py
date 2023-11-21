import asyncio
import json
import uuid

from chainlit_sdk import Chainlit
from chainlit_sdk.observability_agent import OperatorRole
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

sdk = Chainlit(batch_size=2)
sdk.instrument_openai()

thread_id = uuid.uuid4()

welcome_message = "What's your name? "
with sdk.observer.step(type="message", thread_id=thread_id) as step:
    step.output = welcome_message
    step.operatorRole = OperatorRole.SYSTEM

text = input(welcome_message)

with sdk.observer.step(type="message", thread_id=thread_id) as step:
    step.output = text
    step.operatorRole = OperatorRole.USER
    with sdk.observer.step(type="run") as step:
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
        with sdk.observer.step(type="message") as step:
            print("")
            print(completion.choices[0].message.content)
            step.output = completion.choices[0].message.content
            step.operatorRole = OperatorRole.ASSISTANT

sdk.wait_until_queue_empty()


# Get the steps from the API for the demo
async def main():
    print("\nSearching for the thread", thread_id, "...")
    steps = await sdk.api.get_thread(id=str(thread_id))

    thread = steps

    print(json.dumps(thread, indent=2))


asyncio.run(main())
