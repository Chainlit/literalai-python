import asyncio

from dotenv import load_dotenv

from literalai import LiteralClient

load_dotenv()

sdk = LiteralClient(batch_size=2)


async def init():
    thread = await sdk.api.create_thread()
    return await sdk.api.create_step(
        thread_id=thread.id,
        input={"content": "hello world!"},
        output={"content": "hello back!"},
    )


step = asyncio.run(init())


async def main_async():
    # Create a dataset
    dataset = await sdk.api.create_dataset(
        name="foo", description="bar", metadata={"demo": True}
    )
    assert dataset.name == "foo"
    assert dataset.description == "bar"
    assert dataset.metadata == {"demo": True}

    # Update a dataset
    dataset = await sdk.api.update_dataset(id=dataset.id, name="baz")
    assert dataset.name == "baz"

    # Create dataset items
    items = [
        {
            "input": {"content": "What is literal?"},
            "output": {"content": "Literal is an observability solution."},
        },
        {
            "input": {"content": "How can I install the sdk?"},
            "output": {"content": "pip install literalai"},
        },
    ]
    items = [
        await sdk.api.create_dataset_item(dataset_id=dataset.id, **item)
        for item in items
    ]

    for item in items:
        assert item.dataset_id == dataset.id
        assert item.input is not None
        assert item.output is not None

    # Get a dataset with items
    dataset = await sdk.api.get_dataset(id=dataset.id)

    assert dataset.items is not None
    assert len(dataset.items) == 2

    # Add step to dataset
    assert step.id is not None
    step_item = await sdk.api.add_step_to_dataset(dataset.id, step.id)

    assert step_item.input == {"content": "hello world!"}
    assert step_item.output == {"content": "hello back!"}

    # Delete a dataset item
    await sdk.api.delete_dataset_item(id=items[0].id)

    # Delete a dataset
    await sdk.api.delete_dataset(id=dataset.id)


def main_sync():
    # Create a dataset
    dataset = sdk.api.create_dataset_sync(
        name="foo", description="bar", metadata={"demo": True}
    )
    assert dataset.name == "foo"
    assert dataset.description == "bar"
    assert dataset.metadata == {"demo": True}

    # Update a dataset
    dataset = sdk.api.update_dataset_sync(id=dataset.id, name="baz")
    assert dataset.name == "baz"

    # Create dataset items
    items = [
        {
            "input": {"content": "What is literal?"},
            "output": {"content": "Literal is an observability solution."},
        },
        {
            "input": {"content": "How can I install the sdk?"},
            "output": {"content": "pip install literalai"},
        },
    ]
    items = [
        sdk.api.create_dataset_item_sync(dataset_id=dataset.id, **item)
        for item in items
    ]

    for item in items:
        assert item.dataset_id == dataset.id
        assert item.input is not None
        assert item.output is not None

    # Get a dataset with items
    dataset = sdk.api.get_dataset_sync(id=dataset.id)

    assert dataset.items is not None
    assert len(dataset.items) == 2

    # Add step to dataset
    assert step.id is not None
    step_item = sdk.api.add_step_to_dataset_sync(dataset.id, step.id)

    assert step_item.input == {"content": "hello world!"}
    assert step_item.output == {"content": "hello back!"}

    # Delete a dataset item
    sdk.api.delete_dataset_item_sync(id=items[0].id)

    # Delete a dataset
    sdk.api.delete_dataset_sync(id=dataset.id)


asyncio.run(main_async())

main_sync()
