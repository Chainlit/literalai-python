import os
from typing import Optional

from .api import API
from .context import active_steps_var, active_thread_id_var
from .event_processor import EventProcessor
from .instrumentation.openai import instrument_openai
from .types import (
    Step,
    StepContextManager,
    StepType,
    ThreadContextManager,
    step_decorator,
    thread_decorator,
)


class ChainlitClient:
    def __init__(
        self,
        batch_size: int = 1,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ):
        if not api_key:
            api_key = os.getenv("CHAINLIT_API_KEY", None)
            if not api_key:
                raise Exception("CHAINLIT_API_KEY not provided")
        if not endpoint:
            endpoint = os.getenv(
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
        else:
            return ThreadContextManager(self, thread_id=thread_id)

    def step(
        self,
        original_function=None,
        *,
        name: str = "",
        type: StepType = StepType.UNDEFINED,
        thread_id: Optional[str] = None,
    ):
        if original_function:
            return step_decorator(
                self, func=original_function, name=name, type=type, thread_id=thread_id
            )
        else:
            return StepContextManager(self, type=StepType.RUN, thread_id=thread_id)

    def run(
        self,
        original_function=None,
        *,
        name: str = "",
        thread_id: Optional[str] = None,
    ):
        if original_function:
            return step_decorator(
                self,
                func=original_function,
                name=name,
                type=StepType.RUN,
                thread_id=thread_id,
            )
        else:
            return StepContextManager(self, type=StepType.RUN, thread_id=thread_id)

    def llm(
        self,
        original_function=None,
        *,
        name: str = "",
        thread_id: Optional[str] = None,
    ):
        if original_function:
            return step_decorator(
                self,
                func=original_function,
                name=name,
                type=StepType.LLM,
                thread_id=thread_id,
            )
        else:
            return StepContextManager(self, type=StepType.RUN, thread_id=thread_id)

    def retrieval(
        self,
        original_function=None,
        *,
        name: str = "",
        thread_id: Optional[str] = None,
    ):
        if original_function:
            return step_decorator(
                self,
                func=original_function,
                type=StepType.RETRIEVAL,
                thread_id=thread_id,
            )
        else:
            return StepContextManager(self, type=StepType.RUN, thread_id=thread_id)

    def rerank(
        self,
        original_function=None,
        *,
        name: str = "",
        thread_id: Optional[str] = None,
    ):
        if original_function:
            return step_decorator(
                self,
                func=original_function,
                name=name,
                type=StepType.RERANK,
                thread_id=thread_id,
            )
        else:
            return StepContextManager(self, type=StepType.RUN, thread_id=thread_id)

    def embedding(
        self,
        original_function=None,
        *,
        name: str = "",
        thread_id: Optional[str] = None,
    ):
        if original_function:
            return step_decorator(
                self,
                func=original_function,
                type=StepType.EMBEDDING,
                thread_id=thread_id,
            )
        else:
            return StepContextManager(self, type=StepType.RUN, thread_id=thread_id)

    def tool(
        self,
        original_function=None,
        *,
        name: str = "",
        thread_id: Optional[str] = None,
    ):
        if original_function:
            return step_decorator(
                self,
                func=original_function,
                name=name,
                type=StepType.TOOL,
                thread_id=thread_id,
            )
        else:
            return StepContextManager(self, type=StepType.RUN, thread_id=thread_id)

    def create_step(
        self,
        name: str = "",
        type: Optional[StepType] = None,
        thread_id: Optional[str] = None,
    ):
        step = Step(
            name=name, type=type, thread_id=thread_id, processor=self.event_processor
        )
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
