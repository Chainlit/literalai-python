# Literal AI client

## Installation

```bash
pip install literalai
```

## Usage

The full documentation is available [here](https://docs.getliteral.ai/python-client).

Create a `.env` file with the `LITERAL_API_KEY` environment variable set to your API key.

```python
from literalai import LiteralClient
from dotenv import load_dotenv

load_dotenv()

literalai_client = LiteralClient()

@literalai_client.step(type="run")
def my_step(input):
    return f"World"


@literalai_client.thread
def main():
    print(my_step("Hello"))


main()
client.flush_and_stop()
print("Done")
```

## Development setup

```bash
pip install -r requirements-dev.txt
```
