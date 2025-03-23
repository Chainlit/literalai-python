import asyncio

from agents import Agent, Runner, set_trace_processors, trace
from dotenv import load_dotenv

from literalai import LiteralClient

load_dotenv()

client = LiteralClient()


async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

    with trace("Joke workflow"):
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(
            agent, f"Rate this joke: {first_result.final_output}"
        )
        print(f"Joke: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")


if __name__ == "__main__":
    set_trace_processors([client.openai_agents_tracing_processor()])
    asyncio.run(main())
