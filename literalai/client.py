import os
from typing import Any, Dict, List, Optional, Union

from literalai.api import AsyncLiteralAPI, LiteralAPI
from literalai.callback.langchain_callback import get_langchain_callback
from literalai.context import active_steps_var, active_thread_var, active_root_run_var
from literalai.environment import EnvContextManager, env_decorator
from literalai.event_processor import EventProcessor
from literalai.evaluation.experiment_item_run import (
    ExperimentItemRunContextManager,
    experiment_item_run_decorator,
)
from literalai.instrumentation.mistralai import instrument_mistralai
from literalai.instrumentation.openai import instrument_openai
from literalai.observability.message import Message
from literalai.my_types import Environment
from literalai.observability.step import (
    MessageStepType,
    Step,
    StepContextManager,
    TrueStepType,
    step_decorator,
    Attachment,
)
from literalai.observability.thread import ThreadContextManager, thread_decorator

from literalai.requirements import check_all_requirements


class BaseLiteralClient:
    """
    Base class for LiteralClient and AsyncLiteralClient.
    Example:
    ```python
    from literalai import LiteralClient, AsyncLiteralClient

    # Initialize the client
    client = LiteralClient(api_key="your_api_key_here")
    async_client = AsyncLiteralClient(api_key="your_api_key_here")
    ```
    Attributes:
        api (Union[LiteralAPI, AsyncLiteralAPI]): The API client used for communication with Literal AI.
        disabled (bool): Flag indicating whether the client is disabled.
        event_processor (EventProcessor): Processor for handling events.

    """

    api: Union[LiteralAPI, AsyncLiteralAPI]

    def __init__(
        self,
        batch_size: int = 5,
        is_async: bool = False,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        environment: Optional[Environment] = None,
        disabled: bool = False,
    ):
        if not api_key:
            api_key = os.getenv("LITERAL_API_KEY", None)
            if not api_key:
                raise Exception("LITERAL_API_KEY not provided")
        if not url:
            url = os.getenv("LITERAL_API_URL", "https://cloud.getliteral.ai")
        if is_async:
            self.api = AsyncLiteralAPI(
                api_key=api_key, url=url, environment=environment
            )
        else:
            self.api = LiteralAPI(api_key=api_key, url=url, environment=environment)

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
        """
        Instruments the OpenAI SDK so that all LLM calls are logged to Literal AI.
        """
        instrument_openai(self.to_sync())

    def instrument_mistralai(self):
        """
        Instruments the Mistral AI SDK so that all LLM calls are logged to Literal AI.
        """
        instrument_mistralai(self.to_sync())

    def instrument_llamaindex(self):
        """
        Instruments the Llama Index framework so that all RAG & LLM calls are logged to Literal AI.
        """

        LLAMA_INDEX_REQUIREMENT = ["llama-index>=0.10.58"]

        if not check_all_requirements(LLAMA_INDEX_REQUIREMENT):
            raise Exception(
                f"LlamaIndex instrumentation requirements not satisfied: {LLAMA_INDEX_REQUIREMENT}"
            )
        from literalai.instrumentation.llamaindex import instrument_llamaindex

        instrument_llamaindex(self.to_sync())

    def langchain_callback(
        self,
        to_ignore: Optional[List[str]] = None,
        to_keep: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Creates a Callback for Langchain that logs all LLM calls to Literal AI.

        Args:
            to_ignore (Optional[List[str]]): Runs to ignore to declutter logging.
            to_keep (Optional[List[str]]): Runs to keep within ignored runs.

        Returns:
            LangchainTracer: The callback to use in Langchain's invoke methods.
        """
        _LangchainTracer = get_langchain_callback()
        return _LangchainTracer(
            self.to_sync(),
            to_ignore=to_ignore,
            to_keep=to_keep,
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
        root_run_id: Optional[str] = None,
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
                root_run_id=root_run_id,
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
                root_run_id=root_run_id,
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
        root_run_id: Optional[str] = None,
    ):
        return self.step(
            original_function=original_function,
            name=name,
            type="run",
            id=id,
            parent_id=parent_id,
            thread_id=thread_id,
            root_run_id=root_run_id,
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
        root_run_id: Optional[str] = None,
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
            root_run_id=root_run_id,
        )
        step.end()

        return step

    def environment(
        self,
        original_function=None,
        env: Environment = "prod",
        **kwargs,
    ):
        if original_function:
            return env_decorator(
                self,
                func=original_function,
                env=env,
                **kwargs,
            )
        else:
            return EnvContextManager(
                self,
                env=env,
                **kwargs,
            )

    def experiment_item_run(
        self,
        original_function=None,
        **kwargs,
    ):
        if original_function:
            return experiment_item_run_decorator(
                self,
                func=original_function,
                **kwargs,
            )
        else:
            return ExperimentItemRunContextManager(
                self,
                **kwargs,
            )

    def start_step(
        self,
        name: str = "",
        type: Optional[TrueStepType] = None,
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        root_run_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Creates a step and starts it in the current context. To log it on Literal AI use `.end()`.
        This is used to create Agent steps. For conversational messages use `message` instead.

        Args:
            name (Optional[str]): The name of the step to create.
            type (TrueStepType): The type of the step. Must be one of the following :
                                 "run", "tool", "llm", "embedding", "retrieval","rerank", "undefined".
            id (Optional[str]): The id of the step to create.
            parent_id (Optional[str]): The id of the parent step.
            thread_id (Optional[str]): The id of the parent thread.
            root_run_id (Optional[str]): The id of the root run.

        Returns:
            Step: the created step.
        """
        step = Step(
            name=name,
            type=type,
            id=id,
            parent_id=parent_id,
            thread_id=thread_id,
            processor=self.event_processor,
            root_run_id=root_run_id,
            **kwargs,
        )
        step.start()
        return step

    def get_current_step(self):
        """
        Gets the current step from the context.
        """
        active_steps = active_steps_var.get()
        if active_steps and len(active_steps) > 0:
            return active_steps[-1]
        else:
            return None

    def get_current_thread(self):
        """
        Gets the current thread from the context.
        """
        return active_thread_var.get()

    def get_current_root_run(self):
        """
        Gets the current root run from the context.
        """
        return active_root_run_var.get()

    def reset_context(self):
        """
        Resets the context, forgetting active steps & setting current thread to None.
        """
        active_steps_var.set([])
        active_thread_var.set(None)
        active_root_run_var.set(None)

    def flush_and_stop(self):
        """
        Sends all threads and steps to the Literal AI API. Waits synchronously for all API calls to be done.
        """
        self.event_processor.flush_and_stop()


class LiteralClient(BaseLiteralClient):
    """
    Synchronous client for interacting with the Literal AI API.
    Example:
    ```python
    from literalai import LiteralClient
    # Initialize the client
    client = LiteralClient(api_key="your_api_key_here")
    ```
    """

    api: LiteralAPI

    def __init__(
        self,
        batch_size: int = 5,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        environment: Optional[Environment] = None,
        disabled: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            is_async=False,
            api_key=api_key,
            url=url,
            disabled=disabled,
            environment=environment,
        )

    def flush(self):
        self.event_processor.flush()


class AsyncLiteralClient(BaseLiteralClient):
    """
    Asynchronous client for interacting with the Literal AI API.
    Example:
    ```python
    from literalai import AsyncLiteralClient
    # Initialize the client
    async_client = AsyncLiteralClient(api_key="your_api_key_here")
    ```
    """

    api: AsyncLiteralAPI

    def __init__(
        self,
        batch_size: int = 5,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        environment: Optional[Environment] = None,
        disabled: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            is_async=True,
            api_key=api_key,
            url=url,
            disabled=disabled,
            environment=environment,
        )

    async def flush(self):
        await self.event_processor.aflush()
