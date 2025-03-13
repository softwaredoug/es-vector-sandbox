"""LRU cache, but backed to disk."""

import functools
import hashlib
import logging
import os
import pickle
from collections import OrderedDict

logger = logging.getLogger(__name__)


class StoredLRUCache:
    """LRU cache decorator with persistence to disk."""

    def __init__(self, maxsize=128000, cache_file="~/.wands/cache.pkl"):
        self.maxsize = maxsize
        self.cache_file = os.path.expanduser(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self):
        """Load cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                    logger.info(f"Loaded StoredLRUCache from {self.cache_file} with {len(cache)} items")
                    return cache
            except (EOFError, pickle.PickleError, FileNotFoundError):
                return OrderedDict()
        return OrderedDict()

    def _save_cache(self):
        """Save cache to disk."""
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache, f)

    def _make_key(self, args, kwargs):
        """Generate a unique key for the given arguments."""
        key = pickle.dumps((args, tuple(sorted(kwargs.items()))))
        return hashlib.md5(key).hexdigest()  # Short, unique key

    def __call__(self, func):
        """Decorator to apply LRU caching with persistence."""
        cache = self.cache

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = self._make_key(args, kwargs)
            if key in cache:
                # Move accessed item to end (LRU behavior)
                cache.move_to_end(key)
                return cache[key]

            # Compute the function result
            result = func(*args, **kwargs)
            cache[key] = result
            cache.move_to_end(key)

            # Enforce LRU eviction policy
            if len(cache) > self.maxsize:
                cache.popitem(last=False)  # Remove oldest item

            self._save_cache()  # Save updated cache
            return result

        return wrapper
