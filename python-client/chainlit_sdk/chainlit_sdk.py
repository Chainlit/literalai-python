from . import API
from . import EventProcessor
from . import ObservabilityAgent


class ChainlitSDK:
    def __init__(self, batch_size: int = 1):
        self.api = API()
        self.event_processor = EventProcessor(
            api=self.api,
            batch_size=batch_size,
        )
        self.observer = ObservabilityAgent(processor=self.event_processor)

    def wait_until_queue_empty(self):
        self.event_processor.wait_until_queue_empty()
