from dotenv import load_dotenv
from openai import OpenAI

from literalai import LiteralClient

load_dotenv()

openai = OpenAI()

client = LiteralClient()

client.instrument_openai()

prompt = client.api.create_prompt(
    name="hello",
    template_messages=[{"role": "user", "content": "Hello, how are you {{name}}?"}],
)
messages = prompt.format_messages(name="Alice")


res = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
)

print(res)
