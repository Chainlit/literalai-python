import asyncio
import json

from dotenv import load_dotenv
from openai import OpenAI

from literalai import LiteralClient

load_dotenv()

client = OpenAI()

sdk = LiteralClient(batch_size=2)
sdk.instrument_openai()

thread_id = None


@sdk.step(type="run")
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
    thread_id = sdk.get_current_thread().id

    welcome_message = "What's your name? "
    sdk.message(content=welcome_message, type="system_message")
    text = input(welcome_message)
    sdk.message(content=text, type="user_message")

    completion = get_completion(welcome_message=welcome_message, text=text)

    print("")
    print(completion)
    sdk.message(content=completion, type="assistant_message")


run()
sdk.flush_and_stop()


# Get the steps from the API for the demo
async def main():
    print("\nSearching for the thread", thread_id, "...")
    thread = sdk.api.get_thread(id=thread_id)

    print(json.dumps(thread.to_dict(), indent=2))

    # get the LLM step
    llm_step = [step for step in thread.steps if step.type == "llm"][0]

    if not llm_step:
        print("Error: No LLM step found")
        return

    # attach a score
    sdk.api.create_score(
        step_id=llm_step.id,
        name="user-feedback",
        type="HUMAN",
        value=1,
        comment="this is a comment",
    )

    # get the updated steps
    thread = sdk.api.get_thread(id=thread_id)

    print(
        json.dumps(
            [step.to_dict() for step in thread.steps if step.type == "llm"],
            indent=2,
        )
    )


asyncio.run(main())
