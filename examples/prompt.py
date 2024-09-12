from dotenv import load_dotenv

from literalai import LiteralClient

load_dotenv()

client = LiteralClient()

prompt = client.api.get_prompt(name="Default", version=0)

print(prompt)
