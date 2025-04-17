"""
Caching functionality for SKAI-NotiAssistance.

This module provides caching mechanisms to improve performance and reduce costs.
"""

import os
import json
import time
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Callable, List, Tuple

from .logger import get_logger

logger = get_logger(__name__)


class Cache(ABC):
    """Abstract base class for all cache implementations."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all values from the cache.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def hash_key(self, data: Any) -> str:
        """
        Generate a hash key for any data type.
        
        Args:
            data: Data to hash
            
        Returns:
            Hash string
        """
        if isinstance(data, str):
            serialized = data.encode('utf-8')
        else:
            # Convert to JSON string and encode
            try:
                serialized = json.dumps(data, sort_keys=True).encode('utf-8')
            except TypeError:
                # Fallback for non-serializable objects
                serialized = str(data).encode('utf-8')
        
        return hashlib.md5(serialized).hexdigest()


class SimpleCache(Cache):
    """Simple in-memory cache implementation with optional expiration."""
    
    def __init__(self, ttl: Optional[int] = None):
        """
        Initialize the cache.
        
        Args:
            ttl: Default time-to-live in seconds for cached items
        """
        self.cache = {}  # type: Dict[str, Tuple[Any, Optional[float]]]
        self.default_ttl = ttl
        self.logger = get_logger(f"{__name__}.SimpleCache")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            return None
        
        value, expiry = self.cache[key]
        
        # Check if expired
        if expiry is not None and time.time() > expiry:
            self.delete(key)
            return None
        
        self.logger.debug(f"Cache hit for key: {key}")
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds (overrides default)
            
        Returns:
            True
        """
        # Calculate expiry time if TTL provided
        expiry = None
        if ttl is not None:
            expiry = time.time() + ttl
        elif self.default_ttl is not None:
            expiry = time.time() + self.default_ttl
        
        self.cache[key] = (value, expiry)
        self.logger.debug(f"Cached value for key: {key}")
        return True
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed, False otherwise
        """
        if key in self.cache:
            del self.cache[key]
            self.logger.debug(f"Deleted key from cache: {key}")
            return True
        return False
    
    def clear(self) -> bool:
        """
        Clear all values from the cache.
        
        Returns:
            True
        """
        self.cache.clear()
        self.logger.debug("Cleared cache")
        return True
    
    def cleanup(self) -> int:
        """
        Remove all expired items from the cache.
        
        Returns:
            Number of items removed
        """
        now = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if expiry is not None and now > expiry
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
        
        return len(expired_keys)


class DiskCache(Cache):
    """Persistent disk-based cache implementation."""
    
    def __init__(
        self, 
        cache_dir: str = ".cache",
        ttl: Optional[int] = None,
        max_size: Optional[int] = None
    ):
        """
        Initialize the disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Default time-to-live in seconds for cached items
            max_size: Maximum size of cache in bytes (None for unlimited)
        """
        self.cache_dir = os.path.abspath(cache_dir)
        self.default_ttl = ttl
        self.max_size = max_size
        self.logger = get_logger(f"{__name__}.DiskCache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check expiry
            expiry = cache_data.get("expiry")
            if expiry is not None and time.time() > expiry:
                self.delete(key)
                return None
            
            self.logger.debug(f"Cache hit for key: {key}")
            return cache_data.get("value")
            
        except Exception as e:
            self.logger.error(f"Error reading cache file for key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds (overrides default)
            
        Returns:
            True if successful, False otherwise
        """
        cache_path = self._get_cache_path(key)
        
        # Calculate expiry time if TTL provided
        expiry = None
        if ttl is not None:
            expiry = time.time() + ttl
        elif self.default_ttl is not None:
            expiry = time.time() + self.default_ttl
        
        cache_data = {
            "value": value,
            "expiry": expiry,
            "created": time.time()
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
            
            self.logger.debug(f"Cached value for key: {key}")
            
            # Check if we need to cleanup based on max size
            if self.max_size:
                self._check_and_cleanup_size()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing cache file for key {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                self.logger.debug(f"Deleted key from cache: {key}")
                return True
            except Exception as e:
                self.logger.error(f"Error deleting cache file for key {key}: {str(e)}")
        
        return False
    
    def clear(self) -> bool:
        """
        Clear all values from the cache.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
            
            self.logger.debug("Cleared cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def cleanup(self) -> int:
        """
        Remove all expired items from the cache.
        
        Returns:
            Number of items removed
        """
        removed_count = 0
        now = time.time()
        
        try:
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                file_path = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    expiry = cache_data.get("expiry")
                    if expiry is not None and now > expiry:
                        os.remove(file_path)
                        removed_count += 1
                
                except (json.JSONDecodeError, IOError):
                    # Remove corrupt files
                    os.remove(file_path)
                    removed_count += 1
            
            if removed_count:
                self.logger.debug(f"Cleaned up {removed_count} expired cache items")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {str(e)}")
            return 0
    
    def _check_and_cleanup_size(self) -> None:
        """Check total cache size and cleanup if needed."""
        try:
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f))
                for f in os.listdir(self.cache_dir)
                if os.path.isfile(os.path.join(self.cache_dir, f))
            )
            
            if total_size > self.max_size:
                self.logger.debug(f"Cache size {total_size} exceeds limit {self.max_size}, cleaning up")
                self._cleanup_by_age()
        
        except Exception as e:
            self.logger.error(f"Error checking cache size: {str(e)}")
    
    def _cleanup_by_age(self) -> None:
        """Remove oldest cache entries until under size limit."""
        try:
            # Get list of cache files with their creation times
            cache_files = []
            for filename in os.listdir(self.cache_dir):
                if not filename.endswith('.json'):
                    continue
                
                file_path = os.path.join(self.cache_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    created = cache_data.get("created", 0)
                    size = os.path.getsize(file_path)
                    cache_files.append((file_path, created, size))
                
                except (json.JSONDecodeError, IOError):
                    # Remove corrupt files
                    os.remove(file_path)
            
            # Sort by creation time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Calculate total size
            total_size = sum(item[2] for item in cache_files)
            
            # Remove files until under size limit
            removed_count = 0
            for file_path, _, size in cache_files:
                if total_size <= self.max_size:
                    break
                
                os.remove(file_path)
                total_size -= size
                removed_count += 1
            
            if removed_count:
                self.logger.debug(f"Removed {removed_count} oldest cache items to reduce size")
        
        except Exception as e:
            self.logger.error(f"Error during cache size cleanup: {str(e)}")


def cached(
    ttl: Optional[int] = None, 
    cache_instance: Optional[Cache] = None
) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds for cached results
        cache_instance: Optional cache instance to use
        
    Returns:
        Decorated function
    """
    # Create default cache if none provided
    if cache_instance is None:
        cache_instance = SimpleCache(ttl=ttl)
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create a cache key from the function name and arguments
            key_parts = [func.__module__, func.__name__]
            
            # Add args and kwargs to key
            if args:
                key_parts.append(cache_instance.hash_key(args))
            
            if kwargs:
                # Sort kwargs by key for consistent hashing
                sorted_kwargs = {k: kwargs[k] for k in sorted(kwargs.keys())}
                key_parts.append(cache_instance.hash_key(sorted_kwargs))
            
            cache_key = cache_instance.hash_key("_".join(key_parts))
            
            # Check cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator 