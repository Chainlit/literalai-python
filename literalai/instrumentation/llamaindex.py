import os
from typing import Dict, Optional, Any, TypedDict, List
import uuid
import logging
from dotenv import load_dotenv
from llama_index.core.base.response.schema import StreamingResponse, Response
from pydantic import Field

from openai.types import CompletionUsage

from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.span import SimpleSpan
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from llama_index.core.instrumentation.events.synthesis import GetResponseStartEvent, SynthesizeEndEvent
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingStartEvent,
    EmbeddingEndEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatStartEvent,
    LLMChatEndEvent,
)
from llama_index.core.instrumentation.events.query import (
    QueryStartEvent,
    QueryEndEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalStartEvent,
    RetrievalEndEvent,
)

from literalai.my_types import ChatGeneration
from literalai.step import StepType, Step

literalai_uuid_namespace = uuid.UUID("05f6b2b5-a912-47bd-958f-98a9c4496322")

load_dotenv()


def extract_query_from_bundle(str_or_query_bundle: str | QueryBundle):
    if isinstance(str_or_query_bundle, QueryBundle):
        return str_or_query_bundle.query_str

    return str_or_query_bundle


def extract_document_info(nodes: List[NodeWithScore]):
    if not nodes:
        return []

    return [
        {
            'id_': node['node']['id_'],
            'metadata': node['node']['metadata'],
            'text': node['node']['text'],
            'mimetype': node['node']['mimetype'],
            'start_char_idx': node['node']['start_char_idx'],
            'end_char_idx': node['node']['end_char_idx'],
            'char_length': node['node']['end_char_idx'] - node['node']['start_char_idx'],
            'score': node['score']
        }
        for node in nodes
    ]


def create_generation(event: LLMChatStartEvent):
    model_dict = event.dict().get("model_dict")

    return ChatGeneration(
        provider=model_dict.get("class_name"),
        model=model_dict.get("model"),
        settings={
            "model": model_dict.get("model"),
            "temperature": model_dict.get("temperature"),
            "max_tokens": model_dict.get("max_tokens"),
            "logprobs": model_dict.get("logprobs"),
            "top_logprobs": model_dict.get("top_logprobs"),
        },
        messages=[
            {
                "role": message["role"].value,
                "content": message["content"]
            }
            for message in event.dict().get("messages")
        ]
    )


class LiteralEventHandler(BaseEventHandler):
    """ This class handles events coming from LlamaIndex. """

    _client: "LiteralClient" = Field(...)
    _span_handler: "LiteralSpanHandler" = Field(...)
    runs: Dict[str, List[Step]] = {}
    streaming_run_ids: List[str] = []

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, literal_client: "LiteralClient", llama_index_span_handler: "LiteralSpanHandler"):
        super().__init__()
        object.__setattr__(self, '_client', literal_client)
        object.__setattr__(self, '_span_handler', llama_index_span_handler)

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "LiteralEventHandler"

    def handle(self, event: BaseEvent, undefined="undefined", **kwargs) -> None:
        """Logic for handling event."""
        try:
            thread_id = self._span_handler.get_thread_id(event.span_id)
            run_id = self._span_handler.get_run_id(event.span_id)

            """The events are presented here roughly in chronological order"""
            if isinstance(event, QueryStartEvent):
                self._client.api.upsert_thread(id=thread_id)

                self._client.message(
                    name="User query",
                    id=str(event.id_),
                    type="user_message",
                    thread_id=thread_id,
                    content=event.dict().get("query")
                )

            if isinstance(event, RetrievalStartEvent):
                run = self._client.start_step(
                    name="RAG run",
                    type="run",
                    id=run_id,
                    thread_id=thread_id,
                )

                self.store_step(run_id=run_id, step=run)

                retrieval_step = self._client.start_step(
                    type="retrieval",
                    name="Document retrieval",
                    parent_id=run_id,
                    thread_id=thread_id,
                )

                self.store_step(run_id=run_id, step=retrieval_step)

            if isinstance(event, EmbeddingStartEvent):
                retrieval_step = self.get_first_step_of_type(run_id=run_id, step_type="retrieval")

                if run_id and retrieval_step:
                    embedding_step = self._client.start_step(
                        type="embedding",
                        name="Query embedding",
                        parent_id=retrieval_step.id,
                        thread_id=thread_id,
                    )
                    embedding_step.metadata = event.dict().get("model_dict")
                    self.store_step(run_id=run_id, step=embedding_step)

            if isinstance(event, EmbeddingEndEvent):
                embedding_step = self.get_first_step_of_type(run_id=run_id, step_type="embedding")

                if run_id and embedding_step:
                    embedding_step.input = {"chunks": event.dict().get("chunks")}
                    embedding_step.output = {"embeddings": event.dict().get("embeddings")}
                    embedding_step.end()

            if isinstance(event, RetrievalEndEvent):
                retrieval_step = self.get_first_step_of_type(run_id=run_id, step_type="retrieval")

                if run_id and retrieval_step:
                    retrieved_documents = extract_document_info(event.dict().get("nodes"))
                    query = extract_query_from_bundle(event.dict().get("str_or_query_bundle"))

                    retrieval_step.input = {"query": query}
                    retrieval_step.output = {"retrieved_documents": retrieved_documents}
                    retrieval_step.end()

            if isinstance(event, GetResponseStartEvent):
                if run_id:
                    self._client.step()
                    llm_step = self._client.start_step(
                        type="llm",
                        parent_id=run_id,
                        thread_id=thread_id,
                    )
                    self.store_step(run_id=run_id, step=llm_step)

            if isinstance(event, LLMChatStartEvent):
                llm_step = self.get_first_step_of_type(run_id=run_id, step_type="llm")

                if run_id and llm_step:
                    generation = create_generation(event=event)
                    llm_step.generation = generation
                    llm_step.name = event.dict().get("model_dict").get("model")

            if isinstance(event, LLMChatEndEvent):
                llm_step = self.get_first_step_of_type(run_id=run_id, step_type="llm")

                if run_id and llm_step:
                    response = event.dict().get("response")

                    usage = response["raw"]["usage"]

                    if isinstance(usage, CompletionUsage):
                        llm_step.generation.input_token_count = usage.prompt_tokens
                        llm_step.generation.output_token_count = usage.completion_tokens

            if isinstance(event, SynthesizeEndEvent):
                llm_step = self.get_first_step_of_type(run_id=run_id, step_type="llm")
                run = self.get_first_step_of_type(run_id=run_id, step_type="run")

                if run_id and llm_step and run:
                    response = event.dict().get("response")
                    text_response = ""

                    if isinstance(response, StreamingResponse):
                        text_response = str(response.get_response())
                    if isinstance(response, Response):
                        text_response = str(response)

                    llm_step.generation.message_completion = {
                        "role": "assistant",
                        "content": text_response
                    }

                    llm_step.end()
                    run.end()

                    self._client.message(
                        type="assistant_message",
                        thread_id=thread_id,
                        content=text_response
                    )

            if isinstance(event, QueryEndEvent):
                if run_id in self.runs:
                    self.runs[run_id].pop()

        except Exception as e:
            logging.error("[Literal] Error in Llamaindex instrumentation : %s", str(e), exc_info=True)

    def store_step(self, run_id: str, step: Step):
        if run_id not in self.runs:
            self.runs[run_id] = []

        self.runs[run_id].append(step)

    def get_first_step_of_type(self, run_id: Optional[str], step_type: StepType) -> Optional[Step]:
        if not run_id:
            return None

        if run_id not in self.runs:
            return None

        for step in self.runs[run_id]:
            if step.type == step_type:
                return step

        return None


class SpanEntry(TypedDict):
    id: str
    parent_id: Optional[str]
    root_id: Optional[str]
    is_run_root: bool


class LiteralSpanHandler(BaseSpanHandler[SimpleSpan]):
    """ This class handles spans coming from LlamaIndex. """

    spans: Dict[str, SpanEntry] = {}

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "ExampleSpanHandler"

    def new_span(
            self,
            id_: str,
            bound_args: Any,
            instance: Optional[Any] = None,
            parent_span_id: Optional[str] = None,
            **kwargs: Any,
    ):
        self.spans[id_] = {
            "id": id_,
            "parent_id": parent_span_id,
            "root_id": None,
            "is_run_root": self.is_run_root(instance, parent_span_id)
        }

        if parent_span_id is not None:
            self.spans[id_]["root_id"] = self.get_root_span_id(parent_span_id)
        else:
            self.spans[id_]["root_id"] = id_

    def is_run_root(self, instance: Optional[Any], parent_span_id: Optional[str]) -> bool :
        """Returns True if the span is of type RetrieverQueryEngine, and it has no run root in its parent chain"""
        if not isinstance(instance, RetrieverQueryEngine):
            return False

        # Span is of correct type, we check that it doesn't have a run root in its parent chain
        while parent_span_id:
            parent_span = self.spans.get(parent_span_id)

            if not parent_span:
                parent_span_id = None
                continue

            if parent_span["is_run_root"]:
                return False

            parent_span_id = parent_span["parent_id"]

        return True

    def get_root_span_id(self, span_id: Optional[str]):
        """Finds the root span and returns its ID"""
        current_span = self.spans.get(span_id)

        if current_span is None:
            return None

        while current_span["parent_id"] is not None:
            current_span = self.spans.get(current_span["parent_id"])
            if current_span is None:
                return None

        return current_span["id"]

    def get_run_id(self, span_id: str):
        """Go up the span chain to find a run_root, return its ID (or None)"""
        current_span = self.spans.get(span_id)

        if current_span is None:
            return None

        while current_span:
            if current_span["is_run_root"]:
                return str(uuid.uuid5(literalai_uuid_namespace, current_span["id"]))

            current_span = self.spans.get(current_span["parent_id"])

        return None

    def get_thread_id(self, span_id: Optional[str]):
        """Returns the root span ID as a thread ID"""
        if span_id is None:
            return None

        current_span = self.spans.get(span_id)

        if current_span is None:
            return None

        root_span = self.spans.get(current_span["root_id"])

        if root_span is None:
            # span is already the root, uuid its own id
            return str(uuid.uuid5(literalai_uuid_namespace, span_id))
        else:
            # uuid the id of the root span
            return str(uuid.uuid5(literalai_uuid_namespace, root_span["id"]))

    def prepare_to_exit_span(
            self,
            id_: str,
            bound_args: Any,
            instance: Optional[Any] = None,
            result: Optional[Any] = None,
            **kwargs: Any,
    ):
        """Logic for preparing to exit a span."""
        if id in self.spans:
            self.spans[id].pop()

    def prepare_to_drop_span(
            self,
            id_: str,
            bound_args: Any,
            instance: Optional[Any] = None,
            err: Optional[BaseException] = None,
            **kwargs: Any,
    ):
        """Logic for preparing to drop a span."""
        if id in self.spans:
            self.spans[id].pop()


def instrument_llamaindex(client: "LiteralClient"):
    root_dispatcher = get_dispatcher()
    span_handler = LiteralSpanHandler()
    event_handler = LiteralEventHandler(literal_client=client, llama_index_span_handler=span_handler)
    root_dispatcher.add_event_handler(event_handler)
    root_dispatcher.add_span_handler(span_handler)

