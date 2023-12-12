import asyncio

from dotenv import load_dotenv
import mimetypes
from pathlib import Path

from chainlit_client import ChainlitClient

load_dotenv()


sdk = ChainlitClient(batch_size=2)


async def main():
    thread = await sdk.api.create_thread(metadata={"key": "value"}, tags=["hello"])

    id = thread.id
    path = Path(__file__).parent / "./samplefile.txt"
    mime, _ = mimetypes.guess_type(path)

    with open(path, "rb") as file:
        data = file.read()

    res = await sdk.api.upload_file(content=data, mime=mime, thread_id=id)

    print(res)


asyncio.run(main())
