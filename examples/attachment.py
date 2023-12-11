import asyncio

from dotenv import load_dotenv

from chainlit_client import ChainlitClient
from chainlit_client.types import Attachment

load_dotenv()


sdk = ChainlitClient(batch_size=2)


async def main():
    thread = await sdk.api.create_thread()
    step = sdk.create_step(name="test", thread_id=thread.id)
    await sdk.api.send_steps([step])

    try:
        attachment = await sdk.api.create_attachment(
            attachment=Attachment(
                name="test",
                url="https://www.perdu.com/",
                mime="text/html",
                thread_id=thread.id,
                step_id=step.id,
            ),
        )

        print(attachment.to_dict())

        attachment = await sdk.api.update_attachment(
            id=attachment.id,
            url="https://api.github.com/repos/chainlit/chainlit",
            mime="application/json",
            metadata={"test": "test"},
        )

        print(attachment.to_dict())

        attachment = await sdk.api.get_attachment(id=attachment.id)

        print(attachment.to_dict())

        await sdk.api.delete_attachment(id=attachment.id)

        try:
            attachment = await sdk.api.get_attachment(id=attachment.id)
        except Exception as e:
            print(e)

    finally:
        await sdk.api.delete_thread(id=thread.id)


asyncio.run(main())
