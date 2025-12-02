"""Tests for solution caching."""

import time

from chuk_mcp_solver.cache import SolutionCache, clear_global_cache, get_global_cache
from chuk_mcp_solver.models import (
    Constraint,
    ConstraintKind,
    LinearConstraintParams,
    LinearTerm,
    SolveConstraintModelRequest,
    SolveConstraintModelResponse,
    SolverMode,
    SolverStatus,
    Variable,
    VariableDomain,
    VariableDomainType,
)


def test_cache_miss():
    """Test cache miss returns None."""
    cache = SolutionCache()

    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    result = cache.get(request)
    assert result is None


def test_cache_hit():
    """Test cache hit returns cached response."""
    cache = SolutionCache()

    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    response = SolveConstraintModelResponse(status=SolverStatus.SATISFIED)

    # Cache the response
    cache.put(request, response)

    # Should get it back
    cached = cache.get(request)
    assert cached is not None
    assert cached.status == SolverStatus.SATISFIED


def test_cache_different_problems():
    """Test that different problems don't collide."""
    cache = SolutionCache()

    request1 = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    request2 = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(
                    type=VariableDomainType.INTEGER, lower=0, upper=20
                ),  # Different
            )
        ],
        constraints=[],
    )

    response1 = SolveConstraintModelResponse(status=SolverStatus.SATISFIED)
    response2 = SolveConstraintModelResponse(status=SolverStatus.INFEASIBLE)

    cache.put(request1, response1)
    cache.put(request2, response2)

    # Each should get its own response
    assert cache.get(request1).status == SolverStatus.SATISFIED
    assert cache.get(request2).status == SolverStatus.INFEASIBLE


def test_cache_lru_eviction():
    """Test LRU eviction when cache is full."""
    cache = SolutionCache(max_size=2)

    # Add 3 items
    for i in range(3):
        request = SolveConstraintModelRequest(
            mode=SolverMode.SATISFY,
            variables=[
                Variable(
                    id=f"x{i}",
                    domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
                )
            ],
            constraints=[],
        )
        response = SolveConstraintModelResponse(status=SolverStatus.SATISFIED)
        cache.put(request, response)

    # First one should be evicted
    request0 = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x0",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )
    assert cache.get(request0) is None

    # Last two should still be there
    assert cache.size == 2


def test_cache_ttl_expiration():
    """Test cache entries expire after TTL."""
    cache = SolutionCache(ttl_seconds=0.1)  # 100ms TTL

    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    response = SolveConstraintModelResponse(status=SolverStatus.SATISFIED)
    cache.put(request, response)

    # Should be available immediately
    assert cache.get(request) is not None

    # Wait for expiration
    time.sleep(0.15)

    # Should be expired now
    assert cache.get(request) is None


def test_cache_clear():
    """Test clearing cache."""
    cache = SolutionCache()

    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    response = SolveConstraintModelResponse(status=SolverStatus.SATISFIED)
    cache.put(request, response)

    assert cache.size == 1

    cache.clear()

    assert cache.size == 0
    assert cache.get(request) is None


def test_cache_hit_rate():
    """Test cache hit rate calculation."""
    cache = SolutionCache()

    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    response = SolveConstraintModelResponse(status=SolverStatus.SATISFIED)

    # Miss
    cache.get(request)
    assert cache.hit_rate == 0.0

    # Put and hit
    cache.put(request, response)
    cache.get(request)

    # 1 hit out of 2 total = 50%
    assert cache.hit_rate == 50.0


def test_cache_stats():
    """Test cache statistics."""
    cache = SolutionCache(max_size=100, ttl_seconds=3600)

    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    response = SolveConstraintModelResponse(status=SolverStatus.SATISFIED)

    cache.get(request)  # Miss
    cache.put(request, response)
    cache.get(request)  # Hit

    stats = cache.stats()

    assert stats["size"] == 1
    assert stats["max_size"] == 100
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate_pct"] == 50.0
    assert stats["ttl_seconds"] == 3600


def test_cache_evict_stale():
    """Test manual eviction of stale entries."""
    cache = SolutionCache(ttl_seconds=0.1)

    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    response = SolveConstraintModelResponse(status=SolverStatus.SATISFIED)
    cache.put(request, response)

    assert cache.size == 1

    # Wait for staleness
    time.sleep(0.15)

    # Manually evict
    evicted = cache.evict_stale()

    assert evicted == 1
    assert cache.size == 0


def test_global_cache():
    """Test global cache instance."""
    clear_global_cache()

    cache = get_global_cache()
    assert cache.size == 0

    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[],
    )

    response = SolveConstraintModelResponse(status=SolverStatus.SATISFIED)
    cache.put(request, response)

    # Should be in global cache
    cache2 = get_global_cache()
    assert cache2.get(request) is not None

    clear_global_cache()
    assert cache.size == 0


def test_cache_identical_problems():
    """Test that identical problems get same hash."""
    cache = SolutionCache()

    # Create two identical requests
    request1 = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            ),
            Variable(
                id="y",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            ),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[
                        LinearTerm(var="x", coef=1),
                        LinearTerm(var="y", coef=1),
                    ],
                    sense="<=",
                    rhs=15,
                ),
            )
        ],
    )

    request2 = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            ),
            Variable(
                id="y",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            ),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[
                        LinearTerm(var="x", coef=1),
                        LinearTerm(var="y", coef=1),
                    ],
                    sense="<=",
                    rhs=15,
                ),
            )
        ],
    )

    response = SolveConstraintModelResponse(status=SolverStatus.SATISFIED)
    cache.put(request1, response)

    # request2 should hit the cache
    cached = cache.get(request2)
    assert cached is not None
    assert cached.status == SolverStatus.SATISFIED
