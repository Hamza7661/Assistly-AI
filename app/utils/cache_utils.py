"""In-memory caching for AI service to reduce response latency."""
import time
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("assistly.cache")


class InMemoryCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self):
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._max_size = 1000  # Prevent unlimited growth
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None
            
        value, expires_at = self._cache[key]
        if time.time() > expires_at:
            # Expired, remove it
            del self._cache[key]
            return None
            
        logger.debug(f"Cache HIT: {key}")
        return value
        
    def set(self, key: str, value: Any, ttl_seconds: int = 300):
        """Set value in cache with TTL (default 5 minutes)."""
        # Simple size management: remove oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
            
        expires_at = time.time() + ttl_seconds
        self._cache[key] = (value, expires_at)
        logger.debug(f"Cache SET: {key} (TTL: {ttl_seconds}s)")
        
    def delete(self, key: str):
        """Delete a key from cache."""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Cache DELETE: {key}")
            
    def clear(self):
        """Clear entire cache."""
        self._cache.clear()
        logger.info("Cache cleared")
        
    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern (simple substring match)."""
        keys_to_delete = [k for k in self._cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self._cache[key]
        logger.info(f"Invalidated {len(keys_to_delete)} cache entries matching '{pattern}'")


# Global cache instance
_cache = InMemoryCache()


def get_translation_cache_key(texts: List[str], target_language: str, app_id: Optional[str] = None) -> str:
    """Generate cache key for translated texts."""
    # Create a stable hash of the input texts
    text_hash = hashlib.md5(json.dumps(texts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    app_prefix = f"app:{app_id}:" if app_id else ""
    return f"translation:{app_prefix}{target_language}:{text_hash}"


def get_greeting_cache_key(app_id: Optional[str], lang_code: str) -> str:
    """Generate cache key for greeting."""
    app_prefix = app_id if app_id else "default"
    return f"greeting:{app_prefix}:{lang_code}"


def get_cached_translation(texts: List[str], target_language: str, app_id: Optional[str] = None) -> Optional[List[str]]:
    """Get cached translation if available."""
    key = get_translation_cache_key(texts, target_language, app_id)
    return _cache.get(key)


def cache_translation(texts: List[str], target_language: str, translated: List[str], app_id: Optional[str] = None, ttl_seconds: int = 3600):
    """Cache translated texts (default 1 hour)."""
    key = get_translation_cache_key(texts, target_language, app_id)
    _cache.set(key, translated, ttl_seconds)


def get_cached_greeting(app_id: Optional[str], lang_code: str) -> Optional[str]:
    """Get cached greeting if available."""
    key = get_greeting_cache_key(app_id, lang_code)
    return _cache.get(key)


def cache_greeting(app_id: Optional[str], lang_code: str, greeting: str, ttl_seconds: int = 600):
    """Cache greeting (default 10 minutes)."""
    key = get_greeting_cache_key(app_id, lang_code)
    _cache.set(key, greeting, ttl_seconds)


def invalidate_app_cache(app_id: str):
    """Invalidate all cached data for an app."""
    _cache.invalidate_pattern(f"app:{app_id}")
    _cache.invalidate_pattern(f"greeting:{app_id}")


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    return {
        "total_entries": len(_cache._cache),
        "max_size": _cache._max_size
    }
