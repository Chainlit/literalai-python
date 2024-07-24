from literalai import LiteralClient
from llama_index.core import Document, VectorStoreIndex

client = LiteralClient()
client.instrument_llamaindex()

print("Vectorizing documents")
index = VectorStoreIndex.from_documents([Document.example()])

query_engine = index.as_query_engine()
# query_engine = index.as_query_engine(streaming=True)

print()
print("####################")
print("Sending query")
print("####################")
print()

query_engine = index.as_query_engine()
query_engine.query("Tell me about LLMs?")
query_engine = index.as_query_engine(streaming=True)
streaming_response = query_engine.query("Stream : Tell me about LLMs?")
streaming_response.print_response_stream()

client.flush_and_stop()
