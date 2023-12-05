import asyncio
import json

from chainlit_client import ChainlitClient
from dotenv import load_dotenv

load_dotenv()


sdk = ChainlitClient(batch_size=2)


async def main():
    thread = await sdk.api.create_thread(metadata={"key": "value"}, tags=["hello"])

    id = thread.id
    print(id, thread.to_dict())

    thread = await sdk.api.update_thread(
        id=id,
        metadata={"test": "test"},
        tags=["hello:world"],
    )

    print(thread.to_dict())

    thread = await sdk.api.get_thread(id=id)

    print(thread.to_dict())

    await sdk.api.delete_thread(id=id)

    try:
        thread = await sdk.api.get_thread(id=id)
    except Exception as e:
        print(e)

    after = None
    max_calls = 5
    while len((result := await sdk.api.list_threads(first=2, after=after)).data) > 0:
        print(result.to_dict())
        after = result.pageInfo.endCursor
        max_calls -= 1
        if max_calls == 0:
            break


asyncio.run(main())
