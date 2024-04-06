import asyncio
import os

from literalai.api import AsyncLiteralAPI, LiteralAPI

api = LiteralAPI(os.getenv("LITERAL_API_KEY"), os.getenv("LITERAL_API_URL"))

users = api.get_users()
user = api.get_user(identifier="Arthur")

threads = api.get_threads(first=2)

print(threads)


async def main():
    async_api = AsyncLiteralAPI(
        os.getenv("LITERAL_API_KEY"), os.getenv("LITERAL_API_URL")
    )
    print(await async_api.get_threads(first=2))


asyncio.run(main())
