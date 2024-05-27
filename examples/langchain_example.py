import asyncio

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI

from literalai import LiteralClient

load_dotenv()

client = LiteralClient()
chat_model = ChatOpenAI()


@client.thread(name="main")
async def main():
    text = "What would be a good company name for a company that makes colorful socks?"
    messages = [HumanMessage(content=text)]

    cb = client.langchain_callback()
    with client.step(name="chat_model.invoke"):
        print(chat_model.invoke(messages, config={"callbacks": [cb]}))

    print(
        (
            await chat_model.ainvoke(
                messages, config={"callbacks": [client.langchain_callback()]}
            )
        )
    )


asyncio.run(main())
client.flush_and_stop()

print("Done")
