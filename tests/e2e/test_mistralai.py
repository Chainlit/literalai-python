from asyncio import sleep
from literalai.client import LiteralClient
import pytest
import os
from pytest_httpx import HTTPXMock
import urllib.parse
from mistralai.client import MistralClient
from mistralai.async_client import MistralAsyncClient

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


@pytest.mark.e2e
class TestMistralAI:
    @pytest.fixture(scope="class")
    def client(self):
        url = os.getenv("LITERAL_API_URL", None)
        api_key = os.getenv("LITERAL_API_KEY", None)
        assert url is not None and api_key is not None, "Missing environment variables"

        client = LiteralClient(batch_size=1, url=url, api_key=api_key)
        client.instrument_mistralai()

        return client

    async def test_chat(self, client: "LiteralClient", httpx_mock: "HTTPXMock"):
        httpx_mock.add_response(
            json={
                "id": "afc02d747c9b47d3b21e6a1f9fd7e2e6",
                "object": "chat.completion",
                "created": 1718881182,
                "model": "open-mistral-7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '1+1=2\n\nHere\'s a fun fact: The sum of 1 and 1 is called the "successor" in mathematical logic and set theory. It represents the next number after 1. This concept is fundamental in the development of mathematics and forms the basis for the understanding of numbers, arithmetic, and more complex mathematical structures.',
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "total_tokens": 85,
                    "completion_tokens": 76,
                },
            }
        )
        mai_client = MistralClient(api_key="j3s4V1z4")
        thread_id = None

        @client.thread
        def main():
            # https://docs.mistral.ai/api/#operation/createChatCompletion
            mai_client.chat(
                model="open-mistral-7b",
                messages=[
                    {
                        "role": "user",
                        "content": "1+1=?",
                    }
                ],
                temperature=0,
                max_tokens=256,
            )
            return client.get_current_thread()

        thread_id = main().id
        await sleep(2)
        thread = client.api.get_thread(id=thread_id)
        assert thread is not None
        assert thread.steps is not None
        assert len(thread.steps) == 1

        step = thread.steps[0]

        assert step.type == "llm"
        assert step.generation is not None
        assert type(step.generation) == ChatGeneration
        assert step.generation.settings is not None
        assert step.generation.model == "open-mistral-7b"

    async def test_completion(self, client: "LiteralClient", httpx_mock: "HTTPXMock"):
        httpx_mock.add_response(
            json={
                "id": "8103af31e335493da136a79f2e64b59c",
                "object": "chat.completion",
                "created": 1718884349,
                "model": "codestral-2405",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "2\n\n",
                            "tool_calls": None,
                        },
                        "finish_reason": "length",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "total_tokens": 11,
                    "completion_tokens": 3,
                },
            },
        )

        mai_client = MistralClient(api_key="j3s4V1z4")
        thread_id = None

        @client.thread
        def main():
            # https://docs.mistral.ai/api/#operation/createFIMCompletion
            mai_client.completion(
                model="codestral-2405",
                prompt="1+1=",
                temperature=0,
                max_tokens=3,
            )
            return client.get_current_thread()

        thread_id = main().id
        await sleep(2)
        thread = client.api.get_thread(id=thread_id)
        assert thread is not None
        assert thread.steps is not None
        assert len(thread.steps) == 1

        step = thread.steps[0]

        assert step.type == "llm"
        assert step.generation is not None
        assert type(step.generation) == CompletionGeneration
        assert step.generation.settings is not None
        assert step.generation.model == "codestral-2405"
        assert step.generation.completion == "2\n\n"
        assert step.generation.token_count == 11
        assert step.generation.prompt == "1+1="

    async def test_async_chat(self, client: "LiteralClient", httpx_mock: "HTTPXMock"):
        httpx_mock.add_response(
            json={
                "id": "afc02d747c9b47d3b21e6a1f9fd7e2e6",
                "object": "chat.completion",
                "created": 1718881182,
                "model": "open-mistral-7b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '1+1=2\n\nHere\'s a fun fact: The sum of 1 and 1 is called the "successor" in mathematical logic and set theory. It represents the next number after 1. This concept is fundamental in the development of mathematics and forms the basis for the understanding of numbers, arithmetic, and more complex mathematical structures.',
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "total_tokens": 85,
                    "completion_tokens": 76,
                },
            },
        )

        mai_client = MistralAsyncClient(api_key="j3s4V1z4")
        thread_id = None

        @client.thread
        async def main():

            # https://docs.mistral.ai/api/#operation/createChatCompletion
            await mai_client.chat(
                model="open-mistral-7b",
                messages=[
                    {
                        "role": "user",
                        "content": "1+1=?",
                    }
                ],
                temperature=0,
                max_tokens=256,
            )
            return client.get_current_thread()

        thread_id = (await main()).id
        await sleep(2)
        thread = client.api.get_thread(id=thread_id)
        assert thread is not None
        assert thread.steps is not None
        assert len(thread.steps) == 1

        step = thread.steps[0]

        assert step.type == "llm"
        assert step.generation is not None
        assert type(step.generation) == ChatGeneration
        assert step.generation.settings is not None
        assert step.generation.model == "open-mistral-7b"

    async def test_async_completion(
        self, client: "LiteralClient", httpx_mock: "HTTPXMock"
    ):
        httpx_mock.add_response(
            json={
                "id": "8103af31e335493da136a79f2e64b59c",
                "object": "chat.completion",
                "created": 1718884349,
                "model": "codestral-2405",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "2\n\n",
                            "tool_calls": None,
                        },
                        "finish_reason": "length",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "total_tokens": 11,
                    "completion_tokens": 3,
                },
            },
        )

        mai_client = MistralAsyncClient(api_key="j3s4V1z4")
        thread_id = None

        @client.thread
        async def main():
            # https://docs.mistral.ai/api/#operation/createFIMCompletion
            await mai_client.completion(
                model="codestral-2405",
                prompt="1+1=",
                temperature=0,
                max_tokens=3,
            )
            return client.get_current_thread()

        thread_id = (await main()).id
        await sleep(2)
        thread = client.api.get_thread(id=thread_id)
        assert thread is not None
        assert thread.steps is not None
        assert len(thread.steps) == 1

        step = thread.steps[0]

        assert step.type == "llm"
        assert step.generation is not None
        assert type(step.generation) == CompletionGeneration
        assert step.generation.settings is not None
        assert step.generation.model == "codestral-2405"
        assert step.generation.completion == "2\n\n"
        assert step.generation.token_count == 11
        assert step.generation.prompt == "1+1="
