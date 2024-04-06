import asyncio

from dotenv import load_dotenv

from literalai import LiteralClient

load_dotenv()

client = LiteralClient()


@client.thread
def request(query):
    client.message(query)
    return query


@client.thread
async def async_request(query, sleepy):
    client.message(query)
    await asyncio.sleep(sleepy)
    return query


@client.thread(thread_id="e3fcf535-2555-4f75-bc10-fc1499baeff4")
def precise_request(query):
    client.message(query)
    return query


async def main():
    request("hello")
    request("world!")
    precise_request("bonjour!")

    await asyncio.gather(async_request("foo", 5), async_request("bar", 2))


asyncio.run(main())
client.flush_and_stop()
