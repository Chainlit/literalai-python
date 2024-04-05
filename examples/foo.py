import asyncio
import os

from dotenv import load_dotenv

from literalai import LiteralClient

load_dotenv()

client = LiteralClient(api_key=os.getenv("LITERAL_API_KEY"))


filters = [
    {"field": "tags", "operator": "in", "value": ["Test"]},
    {"field": "name", "operator": "nis", "value": None},
]
order_by = {"column": "tokenCount", "direction": "DESC"}


async def main():
    has_next_page = True
    after = None

    while has_next_page:
        response = await client.api.get_threads(
            filters=filters, order_by=order_by, after=after
        )
        print(response.to_dict())
        has_next_page = response.pageInfo.hasNextPage
        after = response.pageInfo.endCursor


asyncio.run(main())
