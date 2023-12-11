import asyncio

from chainlit_client import ChainlitClient
from dotenv import load_dotenv

load_dotenv()


sdk = ChainlitClient(batch_size=2)


async def main():
    thread = await sdk.api.create_thread(metadata={"key": "value"}, tags=["hello"])

    id = thread.id

    res = await sdk.api.upload_file(
        content="Hello World", mime="text/plain", thread_id=id
    )

    print(res)


asyncio.run(main())
