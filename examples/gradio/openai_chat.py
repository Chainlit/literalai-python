import gradio as gr
from openai import AsyncClient
from dotenv import load_dotenv
import os
from chainlit_client import ChainlitClient

load_dotenv()

async_client = AsyncClient(api_key=os.environ.get("OPENAI_KEY"))
sdk = ChainlitClient()
sdk.instrument_openai()


async def run(message, history):
    with sdk.thread():
        sdk.message(content=message, type="user_message")
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

        response = request_llm(history_openai_format)
        result = ""
        async for chunk in response:
            result = chunk
            yield chunk
        sdk.message(content=result, type="assistant_message", name="Bob")


async def request_llm(messages):
    with sdk.step(name="Ask question", type="run"):
        stream = await async_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=1.0,
            stream=True,
        )
        partial_message = ""

        async for chunk in stream:
            content = chunk.choices[0].delta.content

            if content:
                partial_message += content
                yield partial_message


gr.ChatInterface(run).launch()
