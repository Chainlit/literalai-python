import asyncio

from dotenv import load_dotenv

from literalai import LiteralClient

load_dotenv()


sdk = LiteralClient(batch_size=2)


async def main():
    thread = sdk.api.create_thread()
    step = sdk.start_step(name="test", thread_id=thread.id)
    sdk.api.send_steps([step])

    try:
        attachment = sdk.api.create_attachment(
            name="test",
            url="https://www.perdu.com/",
            mime="text/html",
            thread_id=thread.id,
            step_id=step.id,
        )

        print(attachment.to_dict())

        attachment = sdk.api.update_attachment(
            id=attachment.id,
            update_params={
                "url": "https://api.github.com/repos/chainlit/chainlit",
                "mime": "application/json",
                "metadata": {"test": "test"},
            },
        )

        print(attachment.to_dict())

        attachment = sdk.api.get_attachment(id=attachment.id)

        print(attachment.to_dict())

        sdk.api.delete_attachment(id=attachment.id)

        try:
            attachment = sdk.api.get_attachment(id=attachment.id)
        except Exception as e:
            print(e)

    finally:
        sdk.api.delete_thread(id=thread.id)


asyncio.run(main())
