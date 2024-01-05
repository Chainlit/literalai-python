import gradio as gr
from openai import AsyncClient
from dotenv import load_dotenv
import os

load_dotenv()

async_client = AsyncClient(api_key=os.environ.get("OPENAI_KEY"))


async def chat_call(message, history):
    history_openai_format = []
    history_openai_format.append(
        {
            "role": "system",
            "content": """
        Your name is Bob, you are an assistant your are a math expert.
        You only answer math question, if a question is not math related you should respond by pointing out that you are a math expert.
        """,
        }
    )
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    print(history_openai_format)
    stream = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=history_openai_format,
        temperature=1.0,
        stream=True,
    )

    partial_message = ""

    async for chunk in stream:
        content = chunk.choices[0].delta.content

        if content:
            partial_message += content
            yield partial_message


gr.ChatInterface(chat_call).launch()
