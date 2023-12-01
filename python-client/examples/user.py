import asyncio
import json

from chainlit_client import ChainlitClient
from dotenv import load_dotenv

load_dotenv()


sdk = ChainlitClient(batch_size=2)


async def main():
    user = await sdk.api.create_user(identifier="3-test-user", metadata={"name": "123"})

    id = user.id
    print(id, user.to_dict())

    user = await sdk.api.update_user(
        id=id,
        identifier="3-user",
        metadata={"test": "test"},
    )

    print(user.to_dict())

    await sdk.api.delete_user(id=id)

    try:
        user = await sdk.api.update_user(
            id=id,
            identifier="3-users",
            metadata={"test": "test"},
        )
    except Exception as e:
        print(e)


asyncio.run(main())
