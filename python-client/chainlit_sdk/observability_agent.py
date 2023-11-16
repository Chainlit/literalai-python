import json
import time
import uuid
from contextvars import ContextVar
from typing import Any

from .event_processor import EventProcessor
from .wrappers import async_wrapper, sync_wrapper

active_spans_var = ContextVar("active_spans", default=[])


class Span:
    def __init__(
        self,
        name="",
        type=None,
        thread_id=None,
        processor: EventProcessor = None,
    ):
        if processor is None:
            raise Exception("Span must be initialized with a processor.")

        self.id = uuid.uuid4()
        self.name = name
        self.params = {}
        self.parent = None
        self.start = time.time()
        self.end = None
        self.duration = None
        self.type = type
        self.processor = processor

        active_spans = active_spans_var.get()
        if active_spans:
            parent_span = active_spans[-1]
            self.parent = parent_span.id
            self.thread_id = parent_span.thread_id
        else:
            self.thread_id = thread_id
        active_spans.append(self)
        active_spans_var.set(active_spans)

    def set_parameter(self, key, value):
        self.params[key] = value

    def finalize(self):
        end_time = time.time()
        self.end = end_time
        self.duration = end_time - self.start
        active_spans = active_spans_var.get()
        active_spans.pop()
        self.processor.add_event(self.to_dict())
        active_spans_var.set(active_spans)

    def to_dict(self):
        return {
            "id": str(self.id) if self.id else None,
            "name": self.name,
            "params": self.params,
            "parent": str(self.parent) if self.parent else None,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "type": self.type,
            "thread_id": str(self.thread_id),
        }


class SpanContextManager:
    def __init__(self, agent, name="", type=None, thread_id=None):
        self.agent = agent
        self.span_name = name
        self.span_type = type
        self.span = None
        self.thread_id = thread_id

    def __enter__(self):
        self.span = self.agent.create_span(
            name=self.span_name, type=self.span_type, thread_id=self.thread_id
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.span.finalize()


class ObservabilityAgent:
    _instance = None
    processor: EventProcessor = None

    def __new__(cls, processor: EventProcessor = None):
        if not cls._instance:
            cls._instance = super(ObservabilityAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self, processor: EventProcessor = None):
        if self.processor is None:
            self.processor = processor
        elif processor is not None:
            raise Exception("ObservabilityAgent already initialized.")

    def before_wrapper(self, type=None):
        def before(context, *args, **kwargs):
            context["span"] = self.create_span(context["original_func"].__name__, type)

        return before

    def after_wrapper(self):
        def after(result, context, *args, **kwargs):
            context["span"].finalize()

        return after

    def run(self, func):
        return sync_wrapper(
            before_func=self.before_wrapper(type="run"),
            after_func=self.after_wrapper(),
        )(func)

    def message(self, func):
        return sync_wrapper(
            before_func=self.before_wrapper(type="message"),
            after_func=self.after_wrapper(),
        )(func)

    def llm(self, func):
        return sync_wrapper(
            before_func=self.before_wrapper(type="llm"),
            after_func=self.after_wrapper(),
        )(func)

    def a_run(self, func):
        return async_wrapper(
            before_func=self.before_wrapper(type="run"),
            after_func=self.after_wrapper(),
        )(func)

    def a_message(self, func):
        return async_wrapper(
            before_func=self.before_wrapper(type="message"),
            after_func=self.after_wrapper(),
        )(func)

    def a_llm(self, func):
        return async_wrapper(
            before_func=self.before_wrapper(type="llm"),
            after_func=self.after_wrapper(),
        )(func)

    def create_span(self, name="", type=None, thread_id=None):
        if self.processor is None:
            raise Exception("ObservabilityAgent not initialized.")
        span = Span(name=name, type=type, thread_id=thread_id, processor=self.processor)
        return span

    def span(self, name="", type=None, thread_id=None):
        return SpanContextManager(self, name=name, type=type, thread_id=thread_id)

    def set_span_parameter(self, key, value):
        active_spans = active_spans_var.get()
        if active_spans:
            active_spans[-1].set_parameter(key, value)
        else:
            raise Exception("No active spans to set parameter on.")
