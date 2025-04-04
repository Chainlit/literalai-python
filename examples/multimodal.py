import base64
import time

import requests  # type: ignore
from dotenv import load_dotenv
from openai import OpenAI

from literalai import LiteralClient

load_dotenv()

openai_client = OpenAI()

literalai_client = LiteralClient()
literalai_client.initialize()


def encode_image(url):
    return base64.b64encode(requests.get(url).content)


@literalai_client.step(type="run")
def generate_answer(user_query, image_url):
    literalai_client.set_properties(
        name="foobar",
        metadata={"foo": "bar"},
        tags=["foo", "bar"],
    )
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_query},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
        max_tokens=300,
    )
    return completion.choices[0].message.content


def main():
    with literalai_client.thread(name="Meal Analyzer") as thread:
        welcome_message = (
            "Welcome to the meal analyzer, please upload an image of your plate!"
        )
        literalai_client.message(
            content=welcome_message, type="assistant_message", name="My Assistant"
        )

        user_query = "Is this a healthy meal?"
        user_image = "https://www.eatthis.com/wp-content/uploads/sites/4/2021/05/healthy-plate.jpg"
        user_step = literalai_client.message(
            content=user_query, type="user_message", name="User"
        )

        time.sleep(1)  # to make sure the user step has arrived at Literal AI

        literalai_client.api.create_attachment(
            thread_id=thread.id,
            step_id=user_step.id,
            name="meal_image",
            content=encode_image(user_image),
        )

        answer = generate_answer(user_query=user_query, image_url=user_image)
        literalai_client.message(
            content=answer, type="assistant_message", name="My Assistant"
        )


main()
# Network requests by the SDK are performed asynchronously.
# Invoke flush_and_stop() to guarantee the completion of all requests prior to the process termination.
# WARNING: If you run a continuous server, you should not use this method.
literalai_client.flush_and_stop()
