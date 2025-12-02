"""Solution caching for improved performance.

Caches solutions to avoid re-solving identical problems.
Uses problem hash for deduplication and LRU eviction.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from chuk_mcp_solver.diagnostics import compute_problem_hash
from chuk_mcp_solver.models import SolveConstraintModelRequest, SolveConstraintModelResponse


@dataclass
class CacheEntry:
    """Cached solution entry."""

    response: SolveConstraintModelResponse
    timestamp: float
    hit_count: int = 0


class SolutionCache:
    """LRU cache for solver solutions."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0) -> None:
        """Initialize solution cache.

        Args:
            max_size: Maximum number of cached solutions
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    def get(self, request: SolveConstraintModelRequest) -> SolveConstraintModelResponse | None:
        """Get cached solution for a request.

        Args:
            request: The solve request

        Returns:
            Cached response if available and fresh, None otherwise
        """
        problem_hash = compute_problem_hash(request)

        with self._lock:
            entry = self._cache.get(problem_hash)

            if entry is None:
                self._misses += 1
                return None

            # Check if entry is stale
            age = time.time() - entry.timestamp
            if age > self.ttl_seconds:
                # Remove stale entry
                del self._cache[problem_hash]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(problem_hash)
            entry.hit_count += 1
            self._hits += 1

            return entry.response

    def put(
        self, request: SolveConstraintModelRequest, response: SolveConstraintModelResponse
    ) -> None:
        """Cache a solution.

        Args:
            request: The solve request
            response: The solve response to cache
        """
        problem_hash = compute_problem_hash(request)

        with self._lock:
            # Remove oldest entry if at capacity
            if len(self._cache) >= self.max_size and problem_hash not in self._cache:
                self._cache.popitem(last=False)  # Remove oldest (FIFO)

            # Add or update entry
            self._cache[problem_hash] = CacheEntry(
                response=response, timestamp=time.time(), hit_count=0
            )
            self._cache.move_to_end(problem_hash)

    def clear(self) -> None:
        """Clear all cached solutions."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def evict_stale(self) -> int:
        """Evict stale cache entries.

        Returns:
            Number of entries evicted
        """
        with self._lock:
            now = time.time()
            stale_keys = [
                key
                for key, entry in self._cache.items()
                if (now - entry.timestamp) > self.ttl_seconds
            ]

            for key in stale_keys:
                del self._cache[key]

            return len(stale_keys)

    @property
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        with self._lock:
            total = self._hits + self._misses
            return (self._hits / total * 100) if total > 0 else 0.0

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(self.hit_rate, 2),
                "ttl_seconds": self.ttl_seconds,
            }


# Global solution cache
_global_cache = SolutionCache()


def get_global_cache() -> SolutionCache:
    """Get the global solution cache."""
    return _global_cache


def clear_global_cache() -> None:
    """Clear the global solution cache."""
    _global_cache.clear()
