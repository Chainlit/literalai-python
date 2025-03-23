from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from literalai import LiteralClient

load_dotenv()

lai = LiteralClient()
lai.initialize()

prompt = lai.api.get_or_create_prompt(
    name="user intent",
    template_messages=[
        {"role": "system", "content": "You're a helpful assistant."},
        {"role": "user", "content": "{{user_message}}"},
    ],
    settings={
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 4095,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    },
)
messages = prompt.to_langchain_chat_prompt_template()

input_messages = messages.format_messages(
    user_message="The screen is cracked, there are scratches on the surface, and a component is missing."
)

# Returns a langchain_openai.ChatOpenAI instance.
gpt_4o = init_chat_model(  # type: ignore
    model_provider=prompt.provider,
    **prompt.settings,
)

lai.set_properties(prompt=prompt)
print(gpt_4o.invoke(input_messages))

lai.flush_and_stop()
