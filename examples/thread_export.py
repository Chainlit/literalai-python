import asyncio

from dotenv import load_dotenv

from literalai import LiteralClient
from literalai.thread import DateTimeFilter, NumberListFilter, ThreadFilter

load_dotenv()


sdk = LiteralClient()


async def main():
    after = None
    has_next_page = True
    while has_next_page:
        result = await sdk.api.export_threads(after=after)
        after = result.pageInfo.endCursor
        has_next_page = result.pageInfo.hasNextPage

        print("threads fetched", len(result.data))
        # save the fetched threads somewhere

    filters = ThreadFilter(
        createdAt=DateTimeFilter(operator="gt", value="2023-12-14"),
        feedbacksValue=NumberListFilter(operator="in", value=[1]),
    )

    after = None
    has_next_page = True
    while has_next_page:
        result = await sdk.api.export_threads(after=after, filters=filters)
        after = result.pageInfo.endCursor
        has_next_page = result.pageInfo.hasNextPage

        print("threads fetched", len(result.data))
        # save the fetched threads somewhere


asyncio.run(main())
