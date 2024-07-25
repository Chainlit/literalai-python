from literalai import LiteralClient
from llama_index.core import Document, VectorStoreIndex

client = LiteralClient()
client.instrument_llamaindex()

print("Vectorizing documents")
index = VectorStoreIndex.from_documents([Document.example()])
query_engine = index.as_query_engine()

questions = [
    "Tell me about LLMs",
    "How do you fine-tune a neural network ?",
    "What is RAG ?"
]

# No context, create a Thread
for question in questions:
    print(f"> \033[92m{question}\033[0m")
    response = query_engine.query(question)
    print(response)

# One thread per question
for question in questions:
    with client.thread(name=question) as thread:
        print(f"> \033[92m{question}\033[0m")
        response = query_engine.query(question)
        print(response)

# One thread for all the questions
with client.thread(name="Llamaindex questions") as thread:
    for question in questions:
        print(f"> \033[92m{question}\033[0m")
        response = query_engine.query(question)
        print(response)

client.flush_and_stop()
