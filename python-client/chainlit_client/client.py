import os
from typing import Dict, List, Optional

from .api import API
from .context import active_steps_var, active_thread_id_var
from .event_processor import EventProcessor
from .instrumentation.openai import instrument_openai
from .message import Message, MessageType
from .step import Step, StepContextManager, StepType, step_decorator
from .thread import ThreadContextManager, thread_decorator
from .types import Attachment


class ChainlitClient:
    def __init__(
        self,
        batch_size: int = 1,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        if not api_key:
            if not (api_key := os.getenv("CHAINLIT_API_KEY", None)):
                raise Exception("CHAINLIT_API_KEY not provided")
        endpoint = endpoint or os.getenv(
            "CHAINLIT_ENDPOINT", "https://cloud.chainlit.io/graphql"
        )

        self.api = API(api_key=api_key, endpoint=endpoint)
        self.event_processor = EventProcessor(
            api=self.api,
            batch_size=batch_size,
        )

    def instrument_openai(self):
        instrument_openai(self)

    def thread(self, original_function=None, *, thread_id: Optional[str] = None):
        if original_function:
            return thread_decorator(self, func=original_function, thread_id=thread_id)

        return ThreadContextManager(self, thread_id=thread_id)

    def step(
        self,
        original_function=None,
        *,
        name: str = "",
        type: StepType = "UNDEFINED",
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        if original_function:
            return step_decorator(
                self,
                func=original_function,
                name=name,
                type=type,
                id=id,
                parent_id=parent_id,
                thread_id=thread_id,
            )

        return StepContextManager(
            self, type=type, id=id, parent_id=parent_id, thread_id=thread_id
        )

    def message(
        self,
        content: str = "",
        id: Optional[str] = None,
        type: Optional[MessageType] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        attachments: List[Attachment] = [],
        metadata: Dict = {},
    ):
        step = Message(
            name=name,
            id=id,
            thread_id=thread_id,
            type=type,
            content=content,
            attachments=attachments,
            metadata=metadata,
            processor=self.event_processor,
        )
        step.finalize()

        return step

    def create_step(
        self,
        name: str = "",
        type: Optional[StepType] = None,
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        return Step(
            name=name,
            type=type,
            id=id,
            parent_id=parent_id,
            thread_id=thread_id,
            processor=self.event_processor,
        )

    def get_current_step(self):
        active_steps = active_steps_var.get()
        if active_steps and len(active_steps) > 0:
            return active_steps[-1]

        return None

    def get_current_thread_id(self):
        return active_thread_id_var.get()

    def wait_until_queue_empty(self):
        self.event_processor.wait_until_queue_empty()
