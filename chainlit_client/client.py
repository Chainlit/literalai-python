import os
from typing import Dict, List, Optional

from chainlit_client.api import API
from chainlit_client.context import active_steps_var, active_thread_id_var
from chainlit_client.event_processor import EventProcessor
from chainlit_client.instrumentation.openai import instrument_openai
from chainlit_client.message import Message
from chainlit_client.my_types import Attachment
from chainlit_client.step import (
    MessageStepType,
    Step,
    StepContextManager,
    TrueStepType,
    step_decorator,
)
from chainlit_client.thread import ThreadContextManager, thread_decorator


class ChainlitClient:
    def __init__(
        self,
        batch_size: int = 1,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ):
        if not api_key:
            api_key = os.getenv("CHAINLIT_API_KEY", None)
            if not api_key:
                raise Exception("CHAINLIT_API_KEY not provided")
        if not url:
            url = os.getenv("CHAINLIT_API_URL", "https://cloud.chainlit.io")

        self.api = API(api_key=api_key, url=url)
        self.event_processor = EventProcessor(
            api=self.api,
            batch_size=batch_size,
        )

    def instrument_openai(self):
        instrument_openai(self)

    def thread(self, original_function=None, *, thread_id: Optional[str] = None):
        if original_function:
            return thread_decorator(self, func=original_function, thread_id=thread_id)
        else:
            return ThreadContextManager(self, thread_id=thread_id)

    def step(
        self,
        original_function=None,
        *,
        name: str = "",
        type: TrueStepType = "undefined",
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
        else:
            return StepContextManager(
                self,
                name=name,
                type=type,
                id=id,
                parent_id=parent_id,
                thread_id=thread_id,
            )

    def message(
        self,
        content: str = "",
        id: Optional[str] = None,
        type: Optional[MessageStepType] = None,
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
        step.end()

        return step

    def start_step(
        self,
        name: str = "",
        type: Optional[TrueStepType] = None,
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        step = Step(
            name=name,
            type=type,
            id=id,
            parent_id=parent_id,
            thread_id=thread_id,
            processor=self.event_processor,
        )
        step.start()
        return step

    def get_current_step(self):
        active_steps = active_steps_var.get()
        if active_steps and len(active_steps) > 0:
            return active_steps[-1]
        else:
            return None

    def get_current_thread_id(self):
        return active_thread_id_var.get()

    def wait_until_queue_empty(self):
        self.event_processor.wait_until_queue_empty()
