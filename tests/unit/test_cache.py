import pytest
from threading import Thread
import time
import random

from literalai.prompt_engineering.prompt import Prompt
from literalai.api import SharedPromptCache

def default_prompt(id: str = "1", name: str = "test", version: int = 1) -> Prompt:
    return Prompt(
        api=None,
        id=id,
        name=name, 
        version=version,
        created_at="",
        updated_at="",
        type="chat",
        url="",
        version_desc=None,
        template_messages=[],
        tools=None,
        provider="",
        settings={},
        variables=[],
        variables_default_values=None
    )

def test_singleton_instance():
    """Test that SharedPromptCache maintains singleton pattern"""
    cache1 = SharedPromptCache()
    cache2 = SharedPromptCache()
    assert cache1 is cache2
    
def test_get_empty_cache():
    """Test getting from empty cache returns None"""
    cache = SharedPromptCache()
    cache.clear()  # Ensure clean state
    
    assert cache._prompts == {}
    assert cache._name_index == {}
    assert cache._name_version_index == {}

def test_put_and_get_by_id():
    """Test storing and retrieving prompt by ID"""
    cache = SharedPromptCache()
    cache.clear()
    
    prompt = default_prompt()
    cache.put(prompt)
    
    retrieved = cache.get(id="1")
    assert retrieved is prompt
    assert retrieved.id == "1"
    assert retrieved.name == "test"
    assert retrieved.version == 1

def test_put_and_get_by_name():
    """Test storing and retrieving prompt by name"""
    cache = SharedPromptCache()
    cache.clear()
    
    prompt = default_prompt()
    cache.put(prompt)
    
    retrieved = cache.get(name="test")
    assert retrieved is prompt
    assert retrieved.name == "test"

def test_put_and_get_by_name_version():
    """Test storing and retrieving prompt by name and version"""
    cache = SharedPromptCache()
    cache.clear()
    
    prompt = default_prompt()
    cache.put(prompt)
    
    retrieved = cache.get(name="test", version=1)
    assert retrieved is prompt
    assert retrieved.name == "test"
    assert retrieved.version == 1

def test_multiple_versions():
    """Test handling multiple versions of the same prompt"""
    cache = SharedPromptCache()
    cache.clear()
    
    prompt1 = default_prompt()
    prompt2 = default_prompt(id="2", version=2)
    
    cache.put(prompt1)
    cache.put(prompt2)
    
    # Get specific versions
    assert cache.get(name="test", version=1) is prompt1
    assert cache.get(name="test", version=2) is prompt2
    
    # Get by name should return latest version
    assert cache.get(name="test") is prompt2  # Returns the last indexed version

def test_clear_cache():
    """Test clearing the cache"""
    cache = SharedPromptCache()
    prompt = default_prompt()
    cache.put(prompt)
    
    cache.clear()
    assert cache._prompts == {}
    assert cache._name_index == {}
    assert cache._name_version_index == {}

def test_update_existing_prompt():
    """Test updating an existing prompt"""
    cache = SharedPromptCache()
    cache.clear()
    
    prompt1 = default_prompt()
    prompt2 = default_prompt(id="1", version=2) # Same ID, different version
    
    cache.put(prompt1)
    cache.put(prompt2)
    
    retrieved = cache.get(id="1")
    assert retrieved is prompt2
    assert retrieved.version == 2

def test_lookup_priority():
    """Test that lookup priority is id > name-version > name"""
    cache = SharedPromptCache()
    cache.clear()
    
    prompt1 = default_prompt()
    prompt2 = default_prompt(id="2", name="test", version=2)
    
    cache.put(prompt1)
    cache.put(prompt2)
    
    # ID should take precedence
    assert cache.get(id="1", name="test", version=2) is prompt1
    
    # Name-version should take precedence over name
    assert cache.get(name="test", version=2) is prompt2

def test_thread_safety():
    """Test thread safety of the cache"""
    cache = SharedPromptCache()
    cache.clear()
    
    def worker(worker_id: int):
        for i in range(100):
            prompt = default_prompt(
                id=f"{worker_id}-{i}",
                name=f"test-{worker_id}",
                version=i
            )
            cache.put(prompt)
            time.sleep(random.uniform(0, 0.001))
            
            retrieved = cache.get(id=prompt.id)
            assert retrieved is prompt
    
    threads = [Thread(target=worker, args=(i,)) for i in range(10)]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    for worker_id in range(10):
        for i in range(100):
            prompt_id = f"{worker_id}-{i}"
            assert cache.get(id=prompt_id) is not None

def test_error_handling():
    """Test error handling for invalid inputs"""
    cache = SharedPromptCache()
    cache.clear()
    
    assert cache.get() is None
    assert cache.get(id=None, name=None, version=None) is None
    
    with pytest.raises(TypeError):
        cache.get(version="invalid")  # type: ignore
        
    with pytest.raises(TypeError):
        cache.put("not a prompt")  # type: ignore