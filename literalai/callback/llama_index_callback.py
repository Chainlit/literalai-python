import uuid
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from literalai.context import active_steps_var
from literalai.helper import utc_now
from literalai.my_types import (
    Attachment,
    ChatGeneration,
    CompletionGeneration,
    GenerationMessage,
)
from literalai.step import Step, TrueStepType

if TYPE_CHECKING:
    from literalai.client import LiteralClient


def get_llama_index_callback():
    try:
        version("llama_index")
    except Exception:
        raise Exception(
            "Please install llama-index to use the llama index callback. "
            "You can install it with `pip install llama-index`"
        )

    from llama_index.core.callbacks import TokenCountingHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
    from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse

    DEFAULT_IGNORE = [
        CBEventType.CHUNKING,
        CBEventType.SYNTHESIZE,
        CBEventType.EMBEDDING,
        CBEventType.NODE_PARSING,
        CBEventType.TREE,
    ]

    class LlamaIndexTracer(TokenCountingHandler):
        """Base callback handler that can be used to track event starts and ends."""

        client: "LiteralClient"
        steps: Dict[str, Step]

        def __init__(
            self,
            client: "LiteralClient",
            event_starts_to_ignore: List[CBEventType] = DEFAULT_IGNORE,
            event_ends_to_ignore: List[CBEventType] = DEFAULT_IGNORE,
        ) -> None:
            """Initialize the base callback handler."""
            super().__init__(
                event_starts_to_ignore=event_starts_to_ignore,
                event_ends_to_ignore=event_ends_to_ignore,
            )
            self.client = client
            self.is_pristine = True

            self.steps = {}

        def _get_parent_id(
            self, event_parent_id: Optional[str] = None
        ) -> Optional[str]:
            active_steps = active_steps_var.get()
            if event_parent_id and event_parent_id in self.steps:
                return event_parent_id
            elif active_steps:
                return active_steps[-1].id
            else:
                return None

        def on_event_start(
            self,
            event_type: CBEventType,
            payload: Optional[Dict[str, Any]] = None,
            event_id: str = "",
            parent_id: str = "",
            **kwargs: Any,
        ) -> str:
            """Run when an event starts and return id of event."""

            step_type: TrueStepType = "undefined"

            if event_type == CBEventType.RETRIEVE:
                step_type = "retrieval"
            elif event_type == CBEventType.QUERY:
                step_type = "retrieval"
            elif event_type == CBEventType.LLM:
                step_type = "llm"
            else:
                return event_id

            step_type = (
                "run" if self.is_pristine and step_type != "llm" else "undefined"
            )

            self.is_pristine = False

            step = self.client.start_step(
                name=event_type.value,
                type=step_type,
                parent_id=self._get_parent_id(parent_id),
                id=event_id,
            )
            if event_type == CBEventType.EXCEPTION:
                step.error = (
                    payload.get(EventPayload.EXCEPTION, {}).get("message", "")
                    if payload
                    else None
                )
                step.end()
                return event_id
            self.steps[event_id] = step
            step.start_time = utc_now()
            step.input = payload or {}

            return event_id

        def on_event_end(
            self,
            event_type: CBEventType,
            payload: Optional[Dict[str, Any]] = None,
            event_id: str = "",
            **kwargs: Any,
        ) -> None:
            """Run when an event ends."""
            step = self.steps.get(event_id, None)

            if payload is None or step is None:
                return

            step.end_time = utc_now()

            if event_type == CBEventType.QUERY:
                response = payload.get(EventPayload.RESPONSE)
                source_nodes = getattr(response, "source_nodes", None)
                if source_nodes:
                    step.attachments = []
                    for idx, source in enumerate(source_nodes):
                        uploaded = self.client.api.upload_file(
                            content=source.text or "Empty node",
                            thread_id=step.thread_id,
                            mime="text/plain",
                        )
                        object_key = uploaded["object_key"]
                        if object_key:
                            step.attachments.append(
                                Attachment.from_dict(
                                    {
                                        "id": str(uuid.uuid4()),
                                        "name": f"Node {idx}",
                                        "mime": "text/plain",
                                        "objectKey": object_key,
                                    }
                                )
                            )

                    step.output = {}
                    step.end()

            elif event_type == CBEventType.RETRIEVE:
                sources = payload.get(EventPayload.NODES)
                if sources:
                    step.attachments = []
                    for idx, source in enumerate(sources):
                        uploaded = self.client.api.upload_file(
                            content=source.text or "Empty node",
                            thread_id=step.thread_id,
                            mime="text/plain",
                        )
                        object_key = uploaded["object_key"]
                        if object_key:
                            step.attachments.append(
                                Attachment.from_dict(
                                    {
                                        "id": str(uuid.uuid4()),
                                        "name": f"Node {idx}",
                                        "mime": "text/plain",
                                        "objectKey": object_key,
                                    }
                                )
                            )
                    step.output = {}
                step.end()

            elif event_type == CBEventType.LLM:
                formatted_messages = payload.get(
                    EventPayload.MESSAGES
                )  # type: Optional[List[ChatMessage]]
                formatted_prompt = payload.get(EventPayload.PROMPT)
                response = payload.get(EventPayload.RESPONSE)

                if formatted_messages:
                    messages = [
                        GenerationMessage(
                            role=m.role.value, content=m.content or ""  # type: ignore
                        )
                        for m in formatted_messages
                    ]
                else:
                    messages = None

                if isinstance(response, ChatResponse):
                    content = response.message.content or ""
                elif isinstance(response, CompletionResponse):
                    content = response.text
                else:
                    content = ""

                step.output = {"content": content}

                token_count = self.total_llm_token_count or None
                raw_response = response.raw if response else None
                model = raw_response.get("model", None) if raw_response else None

                if messages and isinstance(response, ChatResponse):
                    msg: ChatMessage = response.message
                    step.generation = ChatGeneration(
                        model=model,
                        messages=messages,
                        message_completion=GenerationMessage(
                            role=msg.role.value,  # type: ignore
                            content=content,
                        ),
                        token_count=token_count,
                    )
                elif formatted_prompt:
                    step.generation = CompletionGeneration(
                        model=model,
                        prompt=formatted_prompt,
                        completion=content,
                        token_count=token_count,
                    )

                step.end()

            else:
                step.output = payload
                step.end()

            self.steps.pop(event_id, None)

        def _noop(self, *args, **kwargs):
            pass

        start_trace = _noop
        end_trace = _noop

    return LlamaIndexTracer
