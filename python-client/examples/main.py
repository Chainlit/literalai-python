import asyncio
import json
import uuid

import chainlit_sdk
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

project_id = "test-Sj7iBawHzKSA"

sdk = chainlit_sdk.ChainlitSDK(project_id=project_id, batch_size=2)
chainlit_sdk.instrument_openai()

thread_id = uuid.uuid4()

welcome_message = "What's your name? "
with sdk.observer.step(type="message", thread_id=thread_id) as step:
    step.set_parameter("content", welcome_message)
    step.set_parameter("role", "assistant")

text = input(welcome_message)

with sdk.observer.step(type="message", thread_id=thread_id) as step:
    step.set_parameter("content", text)
    step.set_parameter("role", "user")
    with sdk.observer.step(type="run", thread_id=thread_id) as step:
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
        with sdk.observer.step(type="message", thread_id=thread_id) as step:
            print("")
            print(completion.choices[0].message.content)
            step.set_parameter("content", completion.choices[0].message.content)
            step.set_parameter("role", "assistant")

sdk.wait_until_queue_empty()


# Get the steps from the API for the demo
async def main():
    print("\nSearching for the conversation", thread_id, "...")
    steps = await sdk.api.get_steps(thread_id=str(thread_id), project_id=project_id)

    thread = steps

    print(json.dumps(thread, indent=2))


asyncio.run(main())
