from literalai import AsyncLiteralClient

literalai_client = AsyncLiteralClient()
literalai_client.instrument_llamaindex()
literalai_client.step(type="run")