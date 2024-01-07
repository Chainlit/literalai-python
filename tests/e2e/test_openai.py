import os
import urllib.parse
from asyncio import sleep

import pytest
from openai import AsyncOpenAI, AzureOpenAI, OpenAI
from pytest_httpx import HTTPXMock

from literalai import LiteralClient
from literalai.my_types import ChatGeneration, CompletionGeneration


@pytest.fixture
def non_mocked_hosts() -> list:
    non_mocked_hosts = []

    # Always skip mocking API
    url = os.getenv("LITERAL_API_URL", None)
    if url is not None:
        parsed = urllib.parse.urlparse(url)
        non_mocked_hosts.append(parsed.hostname)

    return non_mocked_hosts


async def wait_until_queue_empty(client: "LiteralClient"):
    # we can't call client.wait_until_queue_empty() because it will join the event_processor thread,
    # this would work only once and then the thread would be dead
    while not client.event_processor.event_queue.empty():
        await sleep(0.1)


@pytest.mark.e2e
class TestOpenAI:
    @pytest.fixture(
        scope="class"
    )  # Feel free to move this fixture up for further testing
    def client(self):
        url = os.getenv("LITERAL_API_URL", None)
        api_key = os.getenv("LITERAL_API_KEY", None)
        assert url is not None and api_key is not None, "Missing environment variables"

        client = LiteralClient(batch_size=1, url=url, api_key=api_key)
        client.instrument_openai()

        return client

    async def test_chat(self, client: "LiteralClient", httpx_mock: "HTTPXMock"):
        # https://platform.openai.com/docs/api-reference/chat/object
        httpx_mock.add_response(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-0613",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "\n\nHello there, how may I assist you today?",
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
        )

        openai_client = OpenAI(api_key="sk_test_123")
        thread_id = None

        @client.thread
        def main():
            # https://platform.openai.com/docs/api-reference/chat/create
            openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "Tell me a funny joke.",
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return client.get_current_thread()

        thread_id = main().id
        await wait_until_queue_empty(client)

        thread = await client.api.get_thread(id=thread_id)
        assert thread is not None
        assert thread.steps is not None
        assert len(thread.steps) == 1

        step = thread.steps[0]

        assert step.type == "llm"
        assert step.generation is not None
        assert type(step.generation) == ChatGeneration
        assert step.generation.settings is not None
        assert step.generation.settings.get("model") == "gpt-3.5-turbo"

    async def test_completion(self, client: "LiteralClient", httpx_mock: "HTTPXMock"):
        # https://platform.openai.com/docs/api-reference/completions/object
        httpx_mock.add_response(
            json={
                "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
                "object": "text_completion",
                "created": 1589478378,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "text": "\n\nThis is indeed a test",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                },
            },
        )

        openai_client = OpenAI(api_key="sk_test_123")
        thread_id = None

        @client.thread
        def main():
            # https://platform.openai.com/docs/api-reference/completions/create
            openai_client.completions.create(
                model="gpt-3.5-turbo",
                prompt="Tell me a funny joke.",
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return client.get_current_thread()

        thread_id = main().id
        await wait_until_queue_empty(client)

        thread = await client.api.get_thread(id=thread_id)
        assert thread is not None
        assert thread.steps is not None
        assert len(thread.steps) == 1

        step = thread.steps[0]

        assert step.type == "llm"
        assert step.generation is not None
        assert type(step.generation) == CompletionGeneration
        assert step.generation.settings is not None
        assert step.generation.settings.get("model") == "gpt-3.5-turbo"
        assert step.generation.completion == "\n\nThis is indeed a test"
        assert step.generation.token_count == 12
        assert step.generation.formatted == "Tell me a funny joke."

    async def test_async_chat(self, client: "LiteralClient", httpx_mock: "HTTPXMock"):
        # https://platform.openai.com/docs/api-reference/chat/object
        httpx_mock.add_response(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-0613",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "\n\nHello there, how may I assist you today?",
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
        )

        openai_client = AsyncOpenAI(api_key="sk_test_123")
        thread_id = None

        @client.thread
        async def main():
            # https://platform.openai.com/docs/api-reference/chat/create
            await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "Tell me a funny joke.",
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return client.get_current_thread()

        thread_id = (await main()).id
        await wait_until_queue_empty(client)

        thread = await client.api.get_thread(id=thread_id)
        assert thread is not None
        assert thread.steps is not None
        assert len(thread.steps) == 1

        step = thread.steps[0]

        assert step.type == "llm"
        assert step.generation is not None
        assert type(step.generation) == ChatGeneration
        assert step.generation.settings is not None
        assert step.generation.settings.get("model") == "gpt-3.5-turbo"

    async def test_async_completion(
        self, client: "LiteralClient", httpx_mock: "HTTPXMock"
    ):
        # https://platform.openai.com/docs/api-reference/completions/object
        httpx_mock.add_response(
            json={
                "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
                "object": "text_completion",
                "created": 1589478378,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "text": "\n\nThis is indeed a test",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                },
            },
        )

        openai_client = AsyncOpenAI(api_key="sk_test_123")
        thread_id = None

        @client.thread
        async def main():
            # https://platform.openai.com/docs/api-reference/completions/create
            await openai_client.completions.create(
                model="gpt-3.5-turbo",
                prompt="Tell me a funny joke.",
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return client.get_current_thread()

        thread_id = (await main()).id
        await wait_until_queue_empty(client)

        thread = await client.api.get_thread(id=thread_id)
        assert thread is not None
        assert thread.steps is not None
        assert len(thread.steps) == 1

        step = thread.steps[0]

        assert step.type == "llm"
        assert step.generation is not None
        assert type(step.generation) == CompletionGeneration
        assert step.generation.settings is not None
        assert step.generation.settings.get("model") == "gpt-3.5-turbo"
        assert step.generation.completion == "\n\nThis is indeed a test"
        assert step.generation.token_count == 12
        assert step.generation.formatted == "Tell me a funny joke."

    async def test_azure_completion(
        self, client: "LiteralClient", httpx_mock: "HTTPXMock"
    ):
        # https://platform.openai.com/docs/api-reference/completions/object
        httpx_mock.add_response(
            json={
                "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
                "object": "text_completion",
                "created": 1589478378,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "text": "\n\nThis is indeed a test",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                },
            },
        )

        openai_client = AzureOpenAI(
            api_key="sk_test_123",
            api_version="2023-05-15",
            azure_endpoint="https://example.org",
        )
        thread_id = None

        @client.thread
        def main():
            # https://platform.openai.com/docs/api-reference/completions/create
            openai_client.completions.create(
                model="gpt-3.5-turbo",
                prompt="Tell me a funny joke.",
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return client.get_current_thread()

        thread_id = main().id
        await wait_until_queue_empty(client)

        thread = await client.api.get_thread(id=thread_id)
        assert thread is not None
        assert thread.steps is not None
        assert len(thread.steps) == 1

        step = thread.steps[0]

        assert step.type == "llm"
        assert step.generation is not None
        assert type(step.generation) == CompletionGeneration
        assert step.generation.settings is not None
        assert step.generation.settings.get("model") == "gpt-3.5-turbo"
        assert step.generation.completion == "\n\nThis is indeed a test"
        assert step.generation.token_count == 12
        assert step.generation.formatted == "Tell me a funny joke."
