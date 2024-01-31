import gradio as gr
from openai import AsyncClient
from dotenv import load_dotenv
import os
from literalai import LiteralClient

load_dotenv()

async_client = AsyncClient(api_key=os.environ.get("OPENAI_KEY"))
sdk = LiteralClient()
sdk.instrument_openai()

history_map = {}


def run(history, message):
    return history + [(message, "")], gr.Textbox(value="", interactive=False)


async def call_assistant(messages):
    if len(messages) == 0:
        return
    last_message = messages[-1][0]
    history = messages[:-1]

    with sdk.thread():
        async for chunk in request_llm(last_message, history):
            messages[-1][1] = chunk
            yield messages


async def request_llm(message, history):
    sdk.message(content=message, type="user_message")

    with sdk.step(name="Ask question", type="run") as step:
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
        stream = await async_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=history_openai_format,
            temperature=1.0,
            stream=True,
        )
        history_map[len(history_map.keys())] = step.id
        response = ""

        async for chunk in stream:
            content = chunk.choices[0].delta.content

            if content:
                response += content
                yield response
    sdk.message(content=response, type="assistant_message", name="Bob")


async def vote(x: gr.LikeData):
    index = x.index[0] if type(x.index) is list else x.index
    step_id = history_map[index]

    await sdk.api.create_feedback(step_id=step_id, value=1 if x.liked else -1)


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    textbox = gr.Textbox()

    textbox.submit(run, [chatbot, textbox], [chatbot, textbox], queue=False).then(
        call_assistant, chatbot, chatbot, api_name="Bob"
    ).then(lambda: gr.Textbox(interactive=True), None, [textbox], queue=False)
    chatbot.like(vote, None, None)

demo.queue()
demo.launch()
