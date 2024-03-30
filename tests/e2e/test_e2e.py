import asyncio
import os
import secrets
import uuid

import pytest

import literalai
from literalai import LiteralClient
from literalai.my_types import ChatGeneration

"""
End to end tests for the SDK
By default this test suite won't run, you need to explicitly run it with the following command:
    pytest -m e2e

Those test require a $LITERAL_API_URL and $LITERAL_API_KEY environment variables to be set.
"""


@pytest.mark.e2e
class Teste2e:
    """
    Don't change the order of the tests, they are dependend on each other
    Those tests are not meant to be parallelized
    """

    @pytest.fixture(
        scope="session"
    )  # Feel free to move this fixture up for further testing
    def client(self):
        url = os.getenv("LITERAL_API_URL", None)
        api_key = os.getenv("LITERAL_API_KEY", None)
        assert url is not None and api_key is not None, "Missing environment variables"

        client = LiteralClient(batch_size=1, url=url, api_key=api_key)
        yield client
        client.event_processor.wait_until_queue_empty()

    async def test_user(self, client: LiteralClient):
        user = await client.api.create_user(
            identifier=f"test_user_{secrets.token_hex()}", metadata={"foo": "bar"}
        )
        assert user.id is not None
        assert user.metadata == {"foo": "bar"}

        updated_user = await client.api.update_user(id=user.id, metadata={"foo": "baz"})
        assert updated_user.metadata == {"foo": "baz"}

        fetched_user = await client.api.get_user(id=user.id)
        assert fetched_user and fetched_user.id == user.id

        users = await client.api.get_users(first=1)
        assert len(users.data) == 1

        await client.api.delete_user(id=user.id)

        deleted_user = await client.api.get_user(id=user.id)
        assert deleted_user is None

    async def test_generation(self, client: LiteralClient):
        chat_generation = ChatGeneration(
            provider="test",
            model="test",
            messages=[
                {"content": "Hello", "role": "user"},
                {"content": "Hi", "role": "assistant"},
            ],
            tags=["test"],
        )
        generation = await client.api.create_generation(chat_generation)
        assert generation.id is not None

        generations = await client.api.get_generations(
            first=1, order_by={"column": "createdAt", "direction": "DESC"}
        )
        assert len(generations.data) == 1
        assert generations.data[0].id == generation.id

    async def test_thread(self, client: LiteralClient):
        user = await client.api.create_user(
            identifier=f"test_user_{secrets.token_hex()}"
        )
        thread = await client.api.create_thread(
            metadata={"foo": "bar"}, participant_id=user.id, tags=["hello"]
        )
        assert thread.id is not None
        assert thread.metadata == {"foo": "bar"}
        assert user.id
        assert thread.user
        assert thread.user.id == user.id

        fetched_thread = await client.api.get_thread(id=thread.id)
        assert fetched_thread and fetched_thread.id == thread.id

        updated_thread = await client.api.update_thread(
            id=fetched_thread.id, tags=["hello:world"]
        )
        assert updated_thread.tags == ["hello:world"]

        threads = await client.api.get_threads(
            first=1, order_by={"column": "createdAt", "direction": "DESC"}
        )
        assert len(threads.data) == 1
        assert threads.data[0].id == thread.id

        threads = await client.api.list_threads(
            first=1, order_by={"column": "createdAt", "direction": "DESC"}
        )
        assert len(threads.data) == 1
        assert threads.data[0].id == thread.id

        await client.api.delete_thread(id=thread.id)
        await client.api.delete_user(id=user.id)

        deleted_thread = await client.api.get_thread(id=thread.id)
        assert deleted_thread is None

    async def test_step(self, client: LiteralClient):
        thread = await client.api.create_thread()
        step = await client.api.create_step(
            thread_id=thread.id, metadata={"foo": "bar"}
        )
        assert step.id is not None
        assert step.thread_id == thread.id

        updated_step = await client.api.update_step(id=step.id, metadata={"foo": "baz"})
        assert updated_step.metadata == {"foo": "baz"}

        fetched_step = await client.api.get_step(id=step.id)
        assert fetched_step and fetched_step.id == step.id

        sent_step = await client.api.send_steps(steps=[step.to_dict()])
        assert len(sent_step["data"].keys()) == 1

        await client.api.delete_step(id=step.id)
        deleted_step = await client.api.get_step(id=step.id)

        assert deleted_step is None

    async def test_score(self, client: LiteralClient):
        thread = await client.api.create_thread()
        step = await client.api.create_step(
            thread_id=thread.id, metadata={"foo": "bar"}
        )

        assert step.id is not None

        score = await client.api.create_score(
            step_id=step.id,
            name="user-feedback",
            type="HUMAN",
            comment="hello",
            value=1,
        )
        assert score.id is not None
        assert score.comment == "hello"

        updated_score = await client.api.update_score(
            id=score.id, update_params={"value": 0}
        )
        assert updated_score.value == 0
        assert updated_score.comment == "hello"

        scores = await client.api.get_scores(
            first=1, order_by={"column": "createdAt", "direction": "DESC"}
        )
        assert len(scores.data) == 1
        assert scores.data[0].id == score.id

    async def test_attachment(self, client: LiteralClient):
        attachment_url = (
            "https://upload.wikimedia.org/wikipedia/commons/8/8f/Example_image.svg"
        )
        thread = await client.api.create_thread()
        step = await client.api.create_step(
            thread_id=thread.id, metadata={"foo": "bar"}
        )

        assert step.id is not None

        attachment = await client.api.create_attachment(
            url=attachment_url,
            step_id=step.id,
            thread_id=thread.id,
            name="foo",
            metadata={"foo": "bar"},
        )
        assert attachment.id is not None
        assert attachment.name == "foo"
        assert attachment.metadata == {"foo": "bar"}

        fetched_attachment = await client.api.get_attachment(id=attachment.id)
        assert fetched_attachment is not None
        assert fetched_attachment.id == attachment.id

        updated_attachment = await client.api.update_attachment(
            id=attachment.id, update_params={"name": "bar"}
        )
        assert updated_attachment.name == "bar"
        assert updated_attachment.metadata == {"foo": "bar"}

        await client.api.delete_attachment(id=attachment.id)
        deleted_attachment = await client.api.get_attachment(id=attachment.id)
        assert deleted_attachment is None

    async def test_ingestion(self, client: LiteralClient):
        with client.thread():
            with client.step(name="test_ingestion") as step:
                step.metadata = {"foo": "bar"}
                assert client.event_processor.event_queue._qsize() == 0
                stack = literalai.context.active_steps_var.get()
                assert len(stack) == 1

        assert client.event_processor.event_queue._qsize() == 1
        stack = literalai.context.active_steps_var.get()
        assert len(stack) == 0

    @pytest.mark.timeout(5)
    async def test_thread_decorator(self, client: LiteralClient):
        async def assert_delete(thread_id: str):
            await asyncio.sleep(1)
            assert await client.api.delete_thread(thread_id) is True

        @client.thread(tags=["foo"], metadata={"async": "False"})
        def thread_decorated():
            t = client.get_current_thread()
            assert t is not None
            assert t.tags == ["foo"]
            assert t.metadata == {"async": "False"}
            return t.id

        id = thread_decorated()
        await assert_delete(id)

        @client.thread(tags=["foo"], name="test", metadata={"async": True})
        async def a_thread_decorated():
            t = client.get_current_thread()
            assert t is not None
            assert t.tags == ["foo"]
            assert t.metadata == {"async": True}
            assert t.name == "test"
            return t.id

        id = await a_thread_decorated()
        await assert_delete(id)

    @pytest.mark.timeout(5)
    async def test_step_decorator(self, client: LiteralClient):
        async def assert_delete(thread_id: str, step_id: str):
            await asyncio.sleep(1)
            assert await client.api.delete_step(step_id) is True
            assert await client.api.delete_thread(thread_id) is True

        @client.thread
        def thread_decorated():
            @client.step(name="foo", type="llm")
            def step_decorated():
                t = client.get_current_thread()
                s = client.get_current_step()
                assert s is not None
                assert s.name == "foo"
                assert s.type == "llm"
                return t.id, s.id

            return step_decorated()

        thread_id, step_id = thread_decorated()
        await assert_delete(thread_id, step_id)

        @client.thread
        async def a_thread_decorated():
            @client.step(name="foo", type="llm")
            async def a_step_decorated():
                t = client.get_current_thread()
                s = client.get_current_step()
                assert s is not None
                assert s.name == "foo"
                assert s.type == "llm"
                return t.id, s.id

            return await a_step_decorated()

        thread_id, step_id = await a_thread_decorated()
        await assert_delete(thread_id, step_id)

    @pytest.mark.timeout(5)
    async def test_run_decorator(self, client: LiteralClient):
        async def assert_delete(step_id: str):
            await asyncio.sleep(1)
            step = await client.api.get_step(step_id)
            assert step and step.output is not None
            assert await client.api.delete_step(step_id) is True

        @client.run(name="foo")
        def step_decorated():
            s = client.get_current_step()
            assert s is not None
            assert s.name == "foo"
            assert s.type == "run"
            return s.id

        step_id = step_decorated()
        await assert_delete(step_id)

    async def test_parallel_requests(self, client: LiteralClient):
        ids = []

        @client.thread
        def request():
            t = client.get_current_thread()
            ids.append(t.id)

        @client.thread
        async def async_request():
            t = client.get_current_thread()
            ids.append(t.id)

        request()
        request()
        await async_request()
        await async_request()

        assert len(ids) == len(set(ids))

    async def create_test_step(self, client: LiteralClient):
        thread = await client.api.create_thread()
        return await client.api.create_step(
            thread_id=thread.id,
            input={"content": "hello world!"},
            output={"content": "hello back!"},
        )

    @pytest.mark.timeout(5)
    async def test_dataset(self, client: LiteralClient):
        dataset_name = str(uuid.uuid4())
        step = await self.create_test_step(client)
        dataset = await client.api.create_dataset(
            name=dataset_name, description="bar", metadata={"demo": True}
        )
        assert dataset.name == dataset_name
        assert dataset.description == "bar"
        assert dataset.metadata == {"demo": True}

        # Update a dataset
        next_name = str(uuid.uuid4())
        await dataset.update(name=next_name)
        assert dataset.name == next_name
        assert dataset.description == "bar"

        # Create dataset items
        inputs = [
            {
                "input": {"content": "What is literal?"},
                "expected_output": {"content": "Literal is an observability solution."},
            },
            {
                "input": {"content": "How can I install the sdk?"},
                "expected_output": {"content": "pip install literalai"},
            },
        ]

        [await dataset.create_item(**input) for input in inputs]

        for item in dataset.items:
            assert item["datasetId"] == dataset.id
            assert item["input"] is not None
            assert item["expectedOutput"] is not None

        # Get a dataset with items
        fetched_dataset = await client.api.get_dataset(id=dataset.id)

        assert fetched_dataset is not None

        assert len(fetched_dataset.items) == 2

        # Add step to dataset
        assert step.id is not None
        step_item = await fetched_dataset.add_step(step.id)

        assert step_item["input"] == {"content": "hello world!"}
        assert step_item["expectedOutput"] == {"content": "hello back!"}

        # Delete a dataset item
        item_id = fetched_dataset.items[0]["id"]
        await fetched_dataset.delete_item(item_id=item_id)

        # Delete a dataset
        await fetched_dataset.delete()

        deleted_dataset = await client.api.get_dataset(id=dataset.id)

        assert deleted_dataset is None

    @pytest.mark.timeout(5)
    async def test_generation_dataset(self, client: LiteralClient):
        chat_generation = ChatGeneration(
            provider="test",
            model="test",
            messages=[
                {"content": "Hello", "role": "user"},
                {"content": "Hi", "role": "assistant"},
            ],
            message_completion={"content": "Hello back!", "role": "assistant"},
            tags=["test"],
        )
        generation = await client.api.create_generation(chat_generation)
        assert generation.id is not None
        dataset_name = str(uuid.uuid4())
        dataset = await client.api.create_dataset(name=dataset_name, type="generation")
        assert dataset.name == dataset_name

        # Add generation to dataset
        generation_item = await dataset.add_generation(generation.id)

        assert generation_item["input"] == {
            "messages": [
                {"content": "Hello", "role": "user"},
                {"content": "Hi", "role": "assistant"},
            ]
        }
        assert generation_item["expectedOutput"] == {
            "content": "Hello back!",
            "role": "assistant",
        }

        # Delete a dataset
        await dataset.delete()

    @pytest.mark.timeout(5)
    async def test_dataset_sync(self, client: LiteralClient):
        step = await self.create_test_step(client)
        dataset_name = str(uuid.uuid4())
        dataset = client.api.create_dataset_sync(
            name=dataset_name, description="bar", metadata={"demo": True}
        )
        assert dataset.name == dataset_name
        assert dataset.description == "bar"
        assert dataset.metadata == {"demo": True}

        # Update a dataset
        next_name = str(uuid.uuid4())
        dataset.update_sync(name=next_name)
        assert dataset.name == next_name
        assert dataset.description == "bar"

        # Create dataset items
        inputs = [
            {
                "input": {"content": "What is literal?"},
                "expected_output": {"content": "Literal is an observability solution."},
            },
            {
                "input": {"content": "How can I install the sdk?"},
                "expected_output": {"content": "pip install literalai"},
            },
        ]
        [dataset.create_item_sync(**input) for input in inputs]

        for item in dataset.items:
            assert item["datasetId"] == dataset.id
            assert item["input"] is not None
            assert item["expectedOutput"] is not None

        # Get a dataset with items
        fetched_dataset = client.api.get_dataset_sync(id=dataset.id)

        assert fetched_dataset is not None
        assert len(fetched_dataset.items) == 2

        # Add step to dataset
        assert step.id is not None
        step_item = fetched_dataset.add_step_sync(step.id)

        assert step_item["input"] == {"content": "hello world!"}
        assert step_item["expectedOutput"] == {"content": "hello back!"}

        # Delete a dataset item
        item_id = fetched_dataset.items[0]["id"]
        fetched_dataset.delete_item_sync(item_id=item_id)

        # Delete a dataset
        fetched_dataset.delete_sync()

        assert client.api.get_dataset_sync(id=fetched_dataset.id) is None

    @pytest.mark.timeout(5)
    async def test_prompt(self, client: LiteralClient):
        prompt = await client.api.get_prompt(name="Default")
        assert prompt is not None
        assert prompt.name == "Default"
        assert prompt.version == 0
        assert prompt.provider == "openai"

        messages = prompt.format()

        expected = """Hello, this is a test value and this

* item 0
* item 1
* item 2

is a templated list."""

        assert messages[0]["content"] == expected

        messages = prompt.format({"test_var": "Edited value"})

        expected = """Hello, this is a Edited value and this

* item 0
* item 1
* item 2

is a templated list."""

        assert messages[0]["content"] == expected
