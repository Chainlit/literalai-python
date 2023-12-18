import asyncio

from dotenv import load_dotenv

from chainlit_client import ChainlitClient
from chainlit_client.thread import DateTimeFilter, ThreadFilter

load_dotenv()


sdk = ChainlitClient()


async def main():
    after = None
    has_next_page = True
    while has_next_page:
        result = await sdk.api.export_threads(after=after)
        after = result.pageInfo.endCursor
        has_next_page = result.pageInfo.hasNextPage

        print("threads fetched", len(result.data))
        # save the fetched threads somewhere

    print("filtered")

    filters = ThreadFilter(createdAt=DateTimeFilter(operator="gt", value="2023-12-14"))

    after = None
    has_next_page = True
    while has_next_page:
        result = await sdk.api.export_threads(after=after, filters=filters)
        after = result.pageInfo.endCursor
        has_next_page = result.pageInfo.hasNextPage

        print("threads fetched", len(result.data))
        # save the fetched threads somewhere


asyncio.run(main())
