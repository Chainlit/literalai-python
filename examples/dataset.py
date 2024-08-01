import asyncio

from dotenv import load_dotenv

from literalai import AsyncLiteralClient, LiteralClient

load_dotenv()

sdk = LiteralClient(batch_size=2)
async_sdk = AsyncLiteralClient(batch_size=2)


async def init():
    thread = await async_sdk.api.create_thread()
    return await async_sdk.api.create_step(
        thread_id=thread.id,
        input={"content": "hello world!"},
        output={"content": "hello back!"},
    )


step = asyncio.run(init())


async def main_async():
    # Create a dataset
    dataset = await async_sdk.api.create_dataset(
        name="foo", description="bar", metadata={"demo": True}
    )
    assert dataset.name == "foo"
    assert dataset.description == "bar"
    assert dataset.metadata == {"demo": True}

    # Update a dataset
    dataset = await async_sdk.api.update_dataset(id=dataset.id, name="baz")
    assert dataset.name == "baz"

    # Create dataset items
    items = [
        {
            "input": {"content": "What is literal?"},
            "expected_output": {"content": "Literal is an observability solution."},
        },
        {
            "input": {"content": "How can I install the sdk?"},
            "expected_output": {"content": "pip install literalai"},
        },
    ]
    items = [
        await async_sdk.api.create_dataset_item(dataset_id=dataset.id, **item)
        for item in items
    ]

    for item in items:
        assert item.dataset_id == dataset.id
        assert item.input is not None
        assert item.expected_output is not None

    # Get a dataset with items
    dataset = await async_sdk.api.get_dataset(id=dataset.id)

    assert dataset.items is not None
    assert len(dataset.items) == 2

    # Add step to dataset
    assert step.id is not None
    step_item = await async_sdk.api.add_step_to_dataset(dataset.id, step.id)

    assert step_item.input == {"content": "hello world!"}
    assert step_item.expected_output == {"content": "hello back!"}

    # Delete a dataset item
    await async_sdk.api.delete_dataset_item(id=items[0].id)

    # Delete a dataset
    await async_sdk.api.delete_dataset(id=dataset.id)


def main_sync():
    # Create a dataset
    dataset = sdk.api.create_dataset(
        name="foo", description="bar", metadata={"demo": True}
    )
    assert dataset.name == "foo"
    assert dataset.description == "bar"
    assert dataset.metadata == {"demo": True}

    # Update a dataset
    dataset = sdk.api.update_dataset(id=dataset.id, name="baz")
    assert dataset.name == "baz"

    # Create dataset items
    items = [
        {
            "input": {"content": "What is literal?"},
            "expected_output": {"content": "Literal is an observability solution."},
        },
        {
            "input": {"content": "How can I install the sdk?"},
            "expected_output": {"content": "pip install literalai"},
        },
    ]
    items = [
        sdk.api.create_dataset_item(dataset_id=dataset.id, **item) for item in items
    ]

    for item in items:
        assert item.dataset_id == dataset.id
        assert item.input is not None
        assert item.expected_output is not None

    # Get a dataset with items
    dataset = sdk.api.get_dataset(id=dataset.id)

    assert dataset.items is not None
    assert len(dataset.items) == 2

    # Add step to dataset
    assert step.id is not None
    step_item = sdk.api.add_step_to_dataset(dataset.id, step.id)

    assert step_item.input == {"content": "hello world!"}
    assert step_item.expected_output == {"content": "hello back!"}

    # Delete a dataset item
    sdk.api.delete_dataset_item(id=items[0].id)

    # Delete a dataset
    sdk.api.delete_dataset(id=dataset.id)


asyncio.run(main_async())

main_sync()
