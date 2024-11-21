from literalai.prompt_engineering.prompt import Prompt
from literalai.cache.shared_cache import SharedCache


def put_prompt(cache: SharedCache, prompt: Prompt):
    cache.put(prompt.id, prompt)
    cache.put(prompt.name, prompt)
    cache.put(f"{prompt.name}-{prompt.version}", prompt)
