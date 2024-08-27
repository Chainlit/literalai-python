import os
import urllib.parse

import pytest

from literalai import LiteralClient

from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def non_mocked_hosts() -> list:
    non_mocked_hosts = []

    # Always skip mocking API
    url = os.getenv("LITERAL_API_URL", None)
    if url is not None:
        parsed = urllib.parse.urlparse(url)
        non_mocked_hosts.append(parsed.hostname)

    return non_mocked_hosts


@pytest.mark.e2e
class TestLlamaIndex:
    @pytest.fixture(
        scope="class"
    )  # Feel free to move this fixture up for further testing
    def client(self):
        url = os.getenv("LITERAL_API_URL", None)
        api_key = os.getenv("LITERAL_API_KEY", None)
        assert url is not None and api_key is not None, "Missing environment variables"

        client = LiteralClient(batch_size=5, url=url, api_key=api_key)
        client.instrument_llamaindex()

        return client

    async def test_instrument_llamaindex(self, client: "LiteralClient"):
        assert client is not None
