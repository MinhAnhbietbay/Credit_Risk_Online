from __future__ import annotations

from threading import Lock

from cachetools import TTLCache, cached
from cachetools.keys import hashkey

__all__ = ["ttl_cache", "clear_ttl_caches"]

_CACHES: list[tuple[TTLCache, Lock]] = []

def ttl_cache(seconds: int, maxsize: int = 64):
    def decorator(fn):
        cache, lock = TTLCache(maxsize=maxsize, ttl=seconds), Lock()
        _CACHES.append((cache, lock))
        return cached(cache, key=lambda _session, *args, **kwargs: hashkey(*args, **kwargs),
                      lock=lock)(fn)
    return decorator

def clear_ttl_caches() -> None:
    for cache, lock in _CACHES:
        with lock:
            cache.clear()
