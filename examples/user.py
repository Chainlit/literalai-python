import asyncio

from dotenv import load_dotenv

from literalai import LiteralClient

load_dotenv()


sdk = LiteralClient(batch_size=2)


async def main():
    user = sdk.api.create_user(identifier="test-user-example", metadata={"name": "123"})

    id = user.id
    print(id, user.to_dict())

    user = sdk.api.update_user(
        id=id,
        identifier="user",
        metadata={"test": "test"},
    )

    print(user.to_dict())

    user = sdk.api.get_user(id=id)

    print(user.to_dict())

    user = sdk.api.get_user(identifier="user")

    print(user.to_dict())

    sdk.api.delete_user(id=id)

    try:
        user = sdk.api.update_user(
            id=id,
            identifier="user",
            metadata={"test": "test"},
        )
    except Exception as e:
        print(e)


asyncio.run(main())
