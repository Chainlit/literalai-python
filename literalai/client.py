import os
from typing import Any, Dict, List, Optional

from literalai.api import API
from literalai.callback.langchain_callback import get_langchain_callback
from literalai.context import active_steps_var, active_thread_var
from literalai.event_processor import EventProcessor
from literalai.instrumentation.openai import instrument_openai
from literalai.message import Message
from literalai.my_types import Attachment
from literalai.step import (
    MessageStepType,
    Step,
    StepContextManager,
    TrueStepType,
    step_decorator,
)
from literalai.thread import ThreadContextManager, thread_decorator


class LiteralClient:
    def __init__(
        self,
        batch_size: int = 1,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ):
        if not api_key:
            api_key = os.getenv("LITERAL_API_KEY", None)
            if not api_key:
                raise Exception("LITERAL_API_KEY not provided")
        if not url:
            url = os.getenv("LITERAL_API_URL", "https://cloud.getliteral.ai")

        self.api = API(api_key=api_key, url=url)
        self.event_processor = EventProcessor(
            api=self.api,
            batch_size=batch_size,
        )

    def instrument_openai(self):
        instrument_openai(self)

    def langchain_callback(
        self,
        to_ignore: Optional[List[str]] = None,
        to_keep: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        LangchainTracer = get_langchain_callback()
        return LangchainTracer(
            self,
            to_ignore=to_ignore,
            to_keep=to_keep,
            **kwargs,
        )

    def thread(
        self, original_function=None, *, thread_id: Optional[str] = None, **kwargs
    ):
        if original_function:
            return thread_decorator(
                self, func=original_function, thread_id=thread_id, **kwargs
            )
        else:
            return ThreadContextManager(self, thread_id=thread_id, **kwargs)

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

    def get_current_thread(self):
        return active_thread_var.get()

    def reset_context(self):
        active_steps_var.set([])
        active_thread_var.set(None)

    def wait_until_queue_empty(self):
        self.event_processor.wait_until_queue_empty()
