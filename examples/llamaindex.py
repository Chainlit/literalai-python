from literalai import LiteralClient
from llama_index.core import Document, VectorStoreIndex
from dotenv import load_dotenv

load_dotenv()

client = LiteralClient()
client.instrument_llamaindex()

client.api.get_step()

print("Vectorizing documents")
index = VectorStoreIndex.from_documents([Document.example()])
query_engine = index.as_query_engine()

questions = [
    "Tell me about LLMs",
    "How do you fine-tune a neural network ?",
    "What is RAG ?"
]

# No context, create a Thread (it will be named after the first user query)
print(f"> \033[92m{questions[0]}\033[0m")
response = query_engine.query(questions[0])
print(response)

# Wrap in a thread (because it has no name it will also be named after the first user query)
with client.thread() as thread:
    print(f"> \033[92m{questions[0]}\033[0m")
    response = query_engine.query(questions[0])
    print(response)

# Wrap in a thread (the name is conserved)
with client.thread(name=f"User question : {questions[0]}") as thread:
    print(f"> \033[92m{questions[0]}\033[0m")
    response = query_engine.query(questions[0])
    print(response)

# One thread for all the questions
with client.thread(name="Llamaindex questions") as thread:
    for question in questions:
        print(f"> \033[92m{question}\033[0m")
        response = query_engine.query(question)
        print(response)

client.flush_and_stop()
