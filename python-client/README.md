# Chainlit Observability SDK
This SDK provides tools for LLM observability in Python applications.

## Installation
```bash
pip install chainlit_sdk
```

## Usage

```python
import chainlit as cl
import chainlit_sdk as sdk
from openai import AsyncOpenAI

client = AsyncOpenAI()

event_processor = sdk.EventProcessor(batch_size=5)
observer = sdk.ObservabilityAgent(processor=event_processor)
sdk.instrument_openai()


@cl.on_chat_start
@observer.a_agent
async def start():
    await cl.Message(content="""Hello!""").send()


@cl.on_message
@observer.a_agent
async def message(message: cl.Message):
    observer.set_span_parameter("message", message.content)

    completion = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a friendly pragmatic person. Be terse.",
            },
            {
                "role": "user",
                "content": message.content,
            },
        ],
    )
    await cl.Message(content=completion.choices[0].message.content).send()
```

## Development

### Setup
```bash
pip install -r requirements.txt
```

### Testing
```bash
pytest
```
