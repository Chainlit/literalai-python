import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from literalai import LiteralClient

load_dotenv()

async_client = AsyncOpenAI()
client = OpenAI()


sdk = LiteralClient(batch_size=2)
sdk.instrument_openai()


@sdk.thread
async def async_run():
    with sdk.step(type="run", name="async_get_chat_completion"):
        stream = await async_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Tell an inspiring quote to the user, mentioning their name. Be extremely supportive while keeping it short. Write one sentence per line.",
                },
                {
                    "role": "assistant",
                    "content": "What's your name? ",
                },
                {
                    "role": "user",
                    "content": "Joe",
                },
            ],
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print("")

    with sdk.step(type="run", name="async_get_completion"):
        stream = await async_client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt="Tell an inspiring quote to the user, mentioning their name. Be extremely supportive while keeping it short. Write one sentence per line.\n\nAssistant: What's your name?\n\nUser: Joe\n\nAssistant: ",
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].text is not None:
                print(chunk.choices[0].text, end="")
        print("")


asyncio.run(async_run())


@sdk.thread
def run():
    with sdk.step(type="run", name="get_chat_completion"):
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Tell an inspiring quote to the user, mentioning their name. Be extremely supportive while keeping it short. Write one sentence per line.",
                },
                {
                    "role": "assistant",
                    "content": "What's your name? ",
                },
                {
                    "role": "user",
                    "content": "Joe",
                },
            ],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print("")

    with sdk.step(type="run", name="get_completion"):
        stream = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt="Tell an inspiring quote to the user, mentioning their name. Be extremely supportive while keeping it short. Write one sentence per line.\n\nAssistant: What's your name?\n\nUser: Joe\n\nAssistant: ",
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].text is not None:
                print(chunk.choices[0].text, end="")
        print("")


run()
sdk.flush_and_stop()
