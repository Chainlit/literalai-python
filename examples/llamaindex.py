from literalai import LiteralClient
from llama_index.core import Document, VectorStoreIndex

client = LiteralClient()
client.instrument_llamaindex()

print("Vectorizing documents")
index = VectorStoreIndex.from_documents([Document.example()])

print()
print("Sending query")
print()

query_engine = index.as_query_engine()
query_engine.query("Tell me about LLMs?")
query_engine = index.as_query_engine(streaming=True)
query_engine.query("Stream : Tell me about LLMs?")

client.flush_and_stop()
