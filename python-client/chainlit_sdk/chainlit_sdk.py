import os

from .api import API
from .event_processor import EventProcessor
from .observability_agent import ObservabilityAgent
from .instrumentation.openai import instrument_openai


class Chainlit:
    def __init__(self, batch_size: int = 1, api_key: str = None, endpoint: str = None):
        if not api_key:
            self.api_key = os.getenv("CHAINLIT_API_KEY", None)
            if not self.api_key:
                raise Exception("CHAINLIT_API_KEY not provided")
        if not endpoint:
            self.endpoint = os.getenv(
                "CHAINLIT_ENDPOINT", "https://cloud.chainlit.io/graphql"
            )

        self.api = API(api_key=self.api_key, endpoint=self.endpoint)
        self.event_processor = EventProcessor(
            api=self.api,
            batch_size=batch_size,
        )
        self.observer = ObservabilityAgent(processor=self.event_processor)

    def instrument_openai(self):
        instrument_openai(self.observer)

    def wait_until_queue_empty(self):
        self.event_processor.wait_until_queue_empty()
