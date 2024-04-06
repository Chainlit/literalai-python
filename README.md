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

client = LiteralClient()

@client.step(type="run")
def my_step(input):
    return f"World"


@client.thread
def main():
    print(my_step("Hello"))


main()
client.flush_and_stop()
print("Done")
```

## Development

### Setup

```bash
pip install -r requirements-dev.txt
```

### Testing

Use the `npx prisma migrate reset` command to reset the database and load the seed data (Warning: this will delete all data in the database).

Start the server on your machine.

You can then run the tests with:

```bash
./run_tests.sh
```
