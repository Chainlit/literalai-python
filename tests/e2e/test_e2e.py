import datetime
import os
import secrets

import pytest

import chainlit_client
from chainlit_client import ChainlitClient

"""
End to end tests for the SDK
By default this test suite won't run, you need to explicitly run it with the following command:
    pytest -m e2e

Those test require a $CHAINLIT_API_URL and $CHAINLIT_API_KEY environment variables to be set.
"""


@pytest.mark.e2e
class Teste2e:
    """
    Don't change the order of the tests, they are dependend on each other
    Those tests are not meant to be parallelized
    """

    @pytest.fixture(
        scope="class"
    )  # Feel free to move this fixture up for further testing
    def client(self):
        url = os.getenv("CHAINLIT_API_URL", None)
        api_key = os.getenv("CHAINLIT_API_KEY", None)
        assert url is not None and api_key is not None, "Missing environment variables"

        client = ChainlitClient(batch_size=1, url=url, api_key=api_key)
        return client

    async def test_user(self, client: ChainlitClient):
        user = await client.api.create_user(
            identifier=f"test_user_{secrets.token_hex()}", metadata={"foo": "bar"}
        )
        assert user.id is not None
        assert user.metadata == {"foo": "bar"}

        updated_user = await client.api.update_user(id=user.id, metadata={"foo": "baz"})
        assert updated_user.metadata == {"foo": "baz"}

        fetched_user = await client.api.get_user(id=user.id)
        assert fetched_user and fetched_user.id == user.id

        await client.api.delete_user(id=user.id)

        deleted_user = await client.api.get_user(id=user.id)
        assert deleted_user is None

    async def test_user_session(self, client: ChainlitClient):
        user_identifier = f"test_user_{secrets.token_hex()}"
        await client.api.create_user(identifier=user_identifier)

        user_session = await client.api.create_user_session(
            participant_identifier=user_identifier,
        )

        anon_user_session = await client.api.create_user_session(
            anon_participant_identifier="foobar",
        )

        assert user_session["startedAt"] is not None
        assert user_session["endedAt"] is None

        assert anon_user_session["startedAt"] is not None
        assert anon_user_session["endedAt"] is None

        ended_at = datetime.datetime.utcnow().isoformat()

        updated_user_session = await client.api.update_user_session(
            id=user_session["id"], is_interactive=True, ended_at=ended_at
        )
        updated_anon_user_session = await client.api.update_user_session(
            id=anon_user_session["id"], ended_at=ended_at
        )

        assert updated_user_session["endedAt"] is not None
        assert updated_user_session["isInteractive"] is True

        assert updated_anon_user_session["endedAt"] is not None
        assert updated_anon_user_session["isInteractive"] is False

        fetched_user_session = await client.api.get_user_session(id=user_session["id"])
        fetched_anon_user_session = await client.api.get_user_session(
            id=anon_user_session["id"]
        )

        assert fetched_user_session and fetched_user_session["id"] is not None
        assert fetched_anon_user_session and fetched_anon_user_session["id"] is not None

        await client.api.delete_user_session(id=user_session["id"])
        await client.api.delete_user_session(id=anon_user_session["id"])

        deleted_user_session = await client.api.get_user_session(id=user_session["id"])
        deleted_anon_user_session = await client.api.get_user_session(
            id=anon_user_session["id"]
        )

        assert deleted_user_session is None
        assert deleted_anon_user_session is None

    async def test_thread(self, client: ChainlitClient):
        """
        Warning: it does not test the list thread pagination
        FIXME update method is broken
        """

        thread = await client.api.create_thread(metadata={"foo": "bar"}, tags=["hello"])
        assert thread.id is not None
        assert thread.metadata == {"foo": "bar"}

        fetched_thread = await client.api.get_thread(id=thread.id)
        assert fetched_thread and fetched_thread.id == thread.id

        updated_thread = await client.api.update_thread(
            id=fetched_thread.id, tags=["hello:world"]
        )
        assert updated_thread.tags == ["hello:world"]

        threads = await client.api.list_threads(first=1)
        assert len(threads.data) == 1

        await client.api.delete_thread(id=thread.id)

        deleted_thread = await client.api.get_thread(id=thread.id)
        assert deleted_thread is None

    async def test_step(self, client: ChainlitClient):
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

    async def test_feedback(self, client: ChainlitClient):
        thread = await client.api.create_thread()
        step = await client.api.create_step(
            thread_id=thread.id, metadata={"foo": "bar"}
        )

        assert step.id is not None

        feedback = await client.api.create_feedback(
            step_id=step.id, comment="hello", value=1
        )
        assert feedback.id is not None
        assert feedback.comment == "hello"

        updated_feedback = await client.api.update_feedback(
            id=feedback.id, update_params={"value": 2}
        )
        assert updated_feedback.value == 2
        assert updated_feedback.comment == "hello"

    async def test_attachment(self, client: ChainlitClient):
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

    async def test_ingestion(self, client: ChainlitClient):
        with client.thread():
            with client.step(name="test_ingestion") as step:
                step.metadata = {"foo": "bar"}
                assert client.event_processor.event_queue._qsize() == 0
                stack = chainlit_client.context.active_steps_var.get()
                assert len(stack) == 1

        assert client.event_processor.event_queue._qsize() == 1
        stack = chainlit_client.context.active_steps_var.get()
        assert len(stack) == 0

        client.event_processor.wait_until_queue_empty()
