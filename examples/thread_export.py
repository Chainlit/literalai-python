import asyncio

from dotenv import load_dotenv

from literalai import LiteralClient
from literalai.thread import DateTimeFilter, ThreadFilter

load_dotenv()


sdk = LiteralClient()


async def main():
    result = await sdk.api.export_threads()
    print(result)

    print("threads fetched", len(result.data))
    # save the fetched threads somewhere

    filters = ThreadFilter(
        createdAt=DateTimeFilter(operator="gt", value="2023-12-14"),
    )

    result = await sdk.api.export_threads(filters=filters)

    print("threads fetched", len(result.data))
    # save the fetched threads somewhere


asyncio.run(main())
