import asyncio

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.llms.openai import OpenAI

from literalai.client import LiteralClient

lai_client = LiteralClient()
lai_client.initialize()


class JokeEvent(Event):
    joke: str


class RewriteJoke(Event):
    joke: str


class JokeFlow(Workflow):
    llm = OpenAI()

    @step()
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic

        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    @step()
    async def return_joke(self, ev: JokeEvent) -> RewriteJoke:
        return RewriteJoke(joke=ev.joke + "What is funny?")

    @step()
    async def critique_joke(self, ev: RewriteJoke) -> StopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))


@lai_client.thread(name="JokeFlow")
async def main():
    w = JokeFlow(timeout=60, verbose=False)
    result = await w.run(topic="pirates")
    print(str(result))


if __name__ == "__main__":
    asyncio.run(main())
