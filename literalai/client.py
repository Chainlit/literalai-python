import os
from typing import Any, Dict, List, Optional, Union

from literalai.api import AsyncLiteralAPI, LiteralAPI
from literalai.callback.langchain_callback import get_langchain_callback
from literalai.callback.llama_index_callback import get_llama_index_callback
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


class BaseLiteralClient:
    """
    Base class for clients interacting with the Literal AI API.

    Attributes:
        api (Union[LiteralAPI, AsyncLiteralAPI]): Instance of the Literal API.
        disabled (bool): Flag indicating if the client is disabled.
        event_processor (EventProcessor): Processor for handling events.
    
    Args:
        batch_size (int, optional): Batch size for event processing. Defaults to 5.
        is_async (bool, optional): Flag indicating if the API should be asynchronous. Defaults to False.
        api_key (Optional[str], optional): API key for authentication. If not provided, it will be fetched from the environment variable 'LITERAL_API_KEY'.
        url (Optional[str], optional): URL of the Literal API. Defaults to 'https://cloud.getliteral.ai'.
        disabled (bool, optional): Flag to disable the client. Defaults to False.
    
    Raises:
        Exception: If 'LITERAL_API_KEY' is not provided and not found in environment variables.
    """

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
        """
        Convert the client to synchronous mode.

        Returns:
            LiteralClient: Synchronous LiteralClient instance.
        """
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
        """
        Instruments the OpenAI API calls with the current client.
        """
        instrument_openai(self.to_sync())

    def langchain_callback(
        self,
        to_ignore: Optional[List[str]] = None,
        to_keep: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Get a Langchain callback tracer configured with the client.

        Args:
            to_ignore (Optional[List[str]], optional): List of event types to ignore. Defaults to None.
            to_keep (Optional[List[str]], optional): List of event types to keep. Defaults to None.
            **kwargs: Additional arguments for the tracer.

        Returns:
            LangchainTracer: Configured Langchain tracer.
        """
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
        """
        Get a Llama Index callback tracer configured with the client.

        Args:
            **kwargs: Additional arguments for the tracer.

        Returns:
            LlamaIndexTracer: Configured Llama Index tracer.
        """
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
        """
        Decorator or context manager for thread-level event grouping.

        Args:
            original_function (callable, optional): Function to decorate. Defaults to None.
            thread_id (Optional[str], optional): ID of the thread. Defaults to None.
            name (Optional[str], optional): Name of the thread. Defaults to None.
            **kwargs: Additional arguments for the thread context manager or decorator.

        Returns:
            Union[ThreadContextManager, callable]: Thread context manager or decorated function.
        """
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
        """
        Decorator or context manager for step-level event grouping.

        Args:
            original_function (callable, optional): Function to decorate. Defaults to None.
            name (str, optional): Name of the step. Defaults to "".
            type (TrueStepType, optional): Type of the step. Defaults to "undefined".
            id (Optional[str], optional): ID of the step. Defaults to None.
            parent_id (Optional[str], optional): Parent ID of the step. Defaults to None.
            thread_id (Optional[str], optional): Thread ID of the step. Defaults to None.
            **kwargs: Additional arguments for the step context manager or decorator.

        Returns:
            Union[StepContextManager, callable]: Step context manager or decorated function.
        """
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
        """
        Decorator or context manager for run-level event grouping.

        Args:
            original_function (callable, optional): Function to decorate. Defaults to None.
            name (str, optional): Name of the run. Defaults to "".
            id (Optional[str], optional): ID of the run. Defaults to None.
            parent_id (Optional[str], optional): Parent ID of the run. Defaults to None.
            thread_id (Optional[str], optional): Thread ID of the run. Defaults to None.

        Returns:
            Union[StepContextManager, callable]: Run context manager or decorated function.
        """
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
        type: Optional[MessageStepType] = None,
        name: Optional[str] = None,
        thread_id: Optional[str] = None,
        attachments: List[Attachment] = [],
        tags: Optional[List[str]] = None,
        metadata: Dict = {},
    ):
        """
        Creates and ends a message step.

        Args:
            content (str, optional): Content of the message. Defaults to "".
            id (Optional[str], optional): ID of the message. Defaults to None.
            parent_id (Optional[str], optional): Parent ID of the message. Defaults to None.
            type (Optional[MessageStepType], optional): Type of the message. Defaults to None.
            name (Optional[str], optional): Name of the message. Defaults to None.
            thread_id (Optional[str], optional): Thread ID of the message. Defaults to None.
            attachments (List[Attachment], optional): List of attachments. Defaults to [].
            tags (Optional[List[str]], optional): List of tags. Defaults to None.
            metadata (Dict, optional): Metadata for the message. Defaults to {}.

        Returns:
            Message: Created message step.
        """
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
        """
        Starts a new step.

        Args:
            name (str, optional): Name of the step. Defaults to "".
            type (Optional[TrueStepType], optional): Type of the step. Defaults to None.
            id (Optional[str], optional): ID of the step. Defaults to None.
            parent_id (Optional[str], optional): Parent ID of the step. Defaults to None.
            thread_id (Optional[str], optional): Thread ID of the step. Defaults to None.
            **kwargs: Additional arguments for the step.

        Returns:
            Step: Started step.
        """
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
        """
        Retrieves the current active step.

        Returns:
            Optional[Step]: The current active step or None if no steps are active.
        """
        active_steps = active_steps_var.get()
        if active_steps and len(active_steps) > 0:
            return active_steps[-1]
        else:
            return None

    def get_current_thread(self):
        """
        Retrieves the current active thread.

        Returns:
            Optional[ThreadContextManager]: The current active thread or None if no threads are active.
        """
        return active_thread_var.get()

    def reset_context(self):
        """
        Resets the active steps and thread context.
        """
        active_steps_var.set([])
        active_thread_var.set(None)

    def flush_and_stop(self):
        """
        Flushes the event processor and stops it.
        """
        self.event_processor.flush_and_stop()


class LiteralClient(BaseLiteralClient):
    """
    Synchronous LiteralClient for interacting with the Literal API.

    Args:
        batch_size (int, optional): Batch size for event processing. Defaults to 5.
        api_key (Optional[str], optional): API key for authentication. If not provided, it will be fetched from the environment variable 'LITERAL_API_KEY'.
        url (Optional[str], optional): URL of the Literal API. Defaults to 'https://cloud.getliteral.ai'.
        disabled (bool, optional): Flag to disable the client. Defaults to False.
    """

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
        """
        Flushes the event processor.
        """
        self.event_processor.flush()


class AsyncLiteralClient(BaseLiteralClient):
    """
    Asynchronous LiteralClient for interacting with the Literal API.

    Attributes:
        api (AsyncLiteralAPI): Instance of the asynchronous Literal API.

    Args:
        batch_size (int, optional): Batch size for event processing. Defaults to 5.
        api_key (Optional[str], optional): API key for authentication. If not provided, it will be fetched from the environment variable 'LITERAL_API_KEY'.
        url (Optional[str], optional): URL of the Literal API. Defaults to 'https://cloud.getliteral.ai'.
        disabled (bool, optional): Flag to disable the client. Defaults to False.
    """

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
        """
        Asynchronously flushes the event processor.

        Returns:
            None
        """
        await self.event_processor.aflush()