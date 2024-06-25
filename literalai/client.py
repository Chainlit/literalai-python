import os
from typing import Any, Dict, List, Optional, Union

from literalai.api import AsyncLiteralAPI, LiteralAPI
from literalai.callback.langchain_callback import get_langchain_callback
from literalai.callback.llama_index_callback import get_llama_index_callback
from literalai.context import active_steps_var, active_thread_var
from literalai.event_processor import EventProcessor
from literalai.instrumentation.mistralai import instrument_mistralai
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


class BaseLiteralClient:
    api: Union[LiteralAPI, AsyncLiteralAPI]

    def __init__(
        self,
        batch_size: int = 5,
        is_async: bool = False,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        disabled: bool = False,
    ):
        if not api_key:
            api_key = os.getenv("LITERAL_API_KEY", None)
            if not api_key:
                raise Exception("LITERAL_API_KEY not provided")
        if not url:
            url = os.getenv("LITERAL_API_URL", "https://cloud.getliteral.ai")
        if is_async:
            self.api = AsyncLiteralAPI(api_key=api_key, url=url)
        else:
            self.api = LiteralAPI(api_key=api_key, url=url)

        self.disabled = disabled

        self.event_processor = EventProcessor(
            api=LiteralAPI(api_key=api_key, url=url),
            batch_size=batch_size,
            disabled=self.disabled,
        )

    def to_sync(self) -> "LiteralClient":
        if isinstance(self.api, AsyncLiteralAPI):
            return LiteralClient(
                batch_size=self.event_processor.batch_size,
                api_key=self.api.api_key,
                url=self.api.url,
                disabled=self.disabled,
            )
        else:
            return self  # type: ignore

    def instrument_openai(self):
        instrument_openai(self.to_sync())

    def instrument_mistralai(self):
        instrument_mistralai(self.to_sync())

    def langchain_callback(
        self,
        to_ignore: Optional[List[str]] = None,
        to_keep: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        LangchainTracer = get_langchain_callback()
        return LangchainTracer(
            self.to_sync(),
            to_ignore=to_ignore,
            to_keep=to_keep,
            **kwargs,
        )

    def llama_index_callback(
        self,
        **kwargs: Any,
    ):
        LlamaIndexTracer = get_llama_index_callback()
        return LlamaIndexTracer(
            self.to_sync(),
            **kwargs,
        )

    def thread(
        self,
        original_function=None,
        *,
        thread_id: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        if original_function:
            return thread_decorator(
                self, func=original_function, thread_id=thread_id, name=name, **kwargs
            )
        else:
            return ThreadContextManager(self, thread_id=thread_id, name=name, **kwargs)

    def step(
        self,
        original_function=None,
        *,
        name: str = "",
        type: TrueStepType = "undefined",
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs,
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
                **kwargs,
            )
        else:
            return StepContextManager(
                self,
                name=name,
                type=type,
                id=id,
                parent_id=parent_id,
                thread_id=thread_id,
                **kwargs,
            )

    def run(
        self,
        original_function=None,
        *,
        name: str = "",
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        return self.step(
            original_function=original_function,
            name=name,
            type="run",
            id=id,
            parent_id=parent_id,
            thread_id=thread_id,
        )

    def message(
        self,
        content: str = "",
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
        type: Optional[MessageStepType] = "assistant_message",
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        attachments: List[Attachment] = [],
        tags: Optional[List[str]] = None,
        metadata: Dict = {},
    ):
        step = Message(
            name=name,
            id=id,
            parent_id=parent_id,
            thread_id=thread_id,
            type=type,
            content=content,
            attachments=attachments,
            tags=tags,
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
        **kwargs,
    ):
        step = Step(
            name=name,
            type=type,
            id=id,
            parent_id=parent_id,
            thread_id=thread_id,
            processor=self.event_processor,
            **kwargs,
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

    def flush_and_stop(self):
        self.event_processor.flush_and_stop()


class LiteralClient(BaseLiteralClient):
    api: LiteralAPI

    def __init__(
        self,
        batch_size: int = 5,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        disabled: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            is_async=False,
            api_key=api_key,
            url=url,
            disabled=disabled,
        )

    def flush(self):
        self.event_processor.flush()


class AsyncLiteralClient(BaseLiteralClient):
    api: AsyncLiteralAPI

    def __init__(
        self,
        batch_size: int = 5,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        disabled: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            is_async=True,
            api_key=api_key,
            url=url,
            disabled=disabled,
        )

    async def flush(self):
        await self.event_processor.aflush()
