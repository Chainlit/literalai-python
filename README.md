# Chainlit Observability SDK
This SDK provides tools for LLM observability in Python applications.

## Installation
```bash
pip install chainlit-client
```

## Usage

The full documentation is available at https://docs.plat.chainlit.io/python-client/get-started/introduction.

Create a `.env` file with the `CHAINLIT_API_KEY` environment variable set to your API key.

```python
from chainlit_client import ChainlitClient
from dotenv import load_dotenv

load_dotenv()

client = ChainlitClient()

@client.step(type="run")
def my_step(input):
    return f"World" 


@client.thread
def main():
    print(my_step("Hello"))


main()
client.wait_until_queue_empty()
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
