from literalai import LiteralClient

from langchain_openai import ChatOpenAI  # type: ignore
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain.agents.agent import BaseSingleActionAgent

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o")
search = TavilySearchResults(max_results=2)
tools = [search]

lai_client = LiteralClient()
lai_prompt = lai_client.api.get_or_create_prompt(
    name="LC Agent",
    settings={
        "model": "gpt-4o-mini",
        "top_p": 1,
        "provider": "openai",
        "max_tokens": 4095,
        "temperature": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    },
    template_messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "assistant", "content": "{{chat_history}}"},
        {"role": "user", "content": "{{input}}"},
        {"role": "assistant", "content": "{{agent_scratchpad}}"},
    ],
)
prompt = lai_prompt.to_langchain_chat_prompt_template()

agent: BaseSingleActionAgent = create_tool_calling_agent(model, tools, prompt)  # type: ignore
agent_executor = AgentExecutor(agent=agent, tools=tools)

lai_client.reset_context()
cb = lai_client.langchain_callback()

agent_executor.invoke(
    {
        "chat_history": [
            # You can specify the intermediary messages as tuples too.
            # ("human", "hi! my name is bob"),
            # ("ai", "Hello Bob! How can I assist you today?")
            HumanMessage(content="hi! my name is bob"),
            AIMessage(content="Hello Bob! How can I assist you today?"),
        ],
        "input": "whats the weather in sf?",
    },
    config=RunnableConfig(callbacks=[cb], run_name="Weather SF"),
)
