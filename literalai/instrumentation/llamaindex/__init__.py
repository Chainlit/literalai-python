from literalai.client import LiteralClient
from llama_index.core.instrumentation import get_dispatcher

from literalai.instrumentation.llamaindex.event_handler import LiteralEventHandler
from literalai.instrumentation.llamaindex.span_handler import LiteralSpanHandler


def instrument_llamaindex(client: "LiteralClient"):
    root_dispatcher = get_dispatcher()

    span_handler = LiteralSpanHandler()
    root_dispatcher.add_span_handler(span_handler)

    event_handler = LiteralEventHandler(
        literal_client=client, llama_index_span_handler=span_handler
    )
    root_dispatcher.add_event_handler(event_handler)
