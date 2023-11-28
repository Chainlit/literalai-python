import asyncio
import json

from chainlit_client import ChainlitClient
from chainlit_client.types import Attachment
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

sdk = ChainlitClient(batch_size=2)
sdk.instrument_openai()

thread_id = None


@sdk.step(type="RUN")
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
    sdk.message(message=welcome_message, role="SYSTEM")
    text = input(welcome_message)
    sdk.message(message=text, role="USER")

    completion = get_completion(welcome_message=welcome_message, text=text)

    print("")
    print(completion)
    sdk.message(message=completion, role="ASSISTANT")


run()
sdk.wait_until_queue_empty()


# Get the steps from the API for the demo
async def main():
    threads = await sdk.api.list_threads()
    print(threads["data"]["threads"]["totalCount"], "threads")

    print("\nSearching for the thread", thread_id, "...")
    thread = await sdk.api.get_thread(id=thread_id)

    print(json.dumps(thread.to_dict(), indent=2))

    # get the LLM step
    llm_step = [step for step in thread.steps if step.type == "LLM"][0]

    if not llm_step:
        print("Error: No LLM step found")
        return

    # attach a feedback
    await sdk.api.set_human_feedback(
        thread_id=thread_id, step_id=llm_step.id, value=1, comment="this is a comment"
    )

    # get the updated steps
    thread = await sdk.api.get_thread(id=thread_id)

    print(
        json.dumps(
            [step.to_dict() for step in thread.steps if step.type == "LLM"],
            indent=2,
        )
    )


asyncio.run(main())
