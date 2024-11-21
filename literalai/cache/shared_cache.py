from typing import Any, Optional


class SharedCache:
    """
    Singleton cache for storing data.
    Only one instance will exist regardless of how many times it's instantiated.
    """
    _instance = None
    _cache: dict[str, Any]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    def get_cache(self) -> dict[str, Any]:
        return self._cache

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves a value from the cache using the provided key.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        return self._cache.get(key)

    def put(self, key: str, value: Any):
        """
        Stores a value in the cache.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        self._cache[key] = value

    def clear(self) -> None:
        """
        Clears all cached values.
        """
        self._cache.clear()

