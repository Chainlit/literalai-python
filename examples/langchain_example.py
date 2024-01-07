import asyncio

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI

from literalai import LiteralClient

load_dotenv()

client = LiteralClient()
chat_model = ChatOpenAI()


@client.thread
async def main():
    text = "What would be a good company name for a company that makes colorful socks?"
    messages = [HumanMessage(content=text)]

    cb = client.langchain_callback()
    with client.step(name="chat_model.invoke"):
        print(chat_model.invoke(messages, config={"callbacks": [cb]}).content)

    print(
        (
            await chat_model.ainvoke(
                messages, config={"callbacks": [client.langchain_callback()]}
            )
        ).content
    )

    print(
        chat_model.batch(
            [messages], config={"callbacks": [client.langchain_callback()]}
        )
    )

    print(
        await chat_model.abatch(
            [messages], config={"callbacks": [client.langchain_callback()]}
        )
    )

    for chunk in chat_model.stream(
        messages, config={"callbacks": [client.langchain_callback()]}
    ):
        print(chunk.content, end="", flush=True)
    print("")

    async for chunk in chat_model.astream(
        messages, config={"callbacks": [client.langchain_callback()]}
    ):
        print(chunk.content, end="", flush=True)
    print("")


asyncio.run(main())
client.wait_until_queue_empty()
print("Done")
