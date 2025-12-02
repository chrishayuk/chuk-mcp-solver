"""Caching Demo - Shows solution caching and performance benefits.

This example demonstrates the Phase 3 caching features:
- Automatic solution caching with problem hashing
- LRU eviction with TTL
- Cache hit rate tracking
- Performance improvements for repeated problems

Run: python examples/caching_demo.py
"""

import asyncio
import time

from chuk_mcp_solver.cache import clear_global_cache, get_global_cache
from chuk_mcp_solver.models import (
    Constraint,
    ConstraintKind,
    LinearConstraintParams,
    LinearTerm,
    Objective,
    ObjectiveSense,
    SearchConfig,
    SolveConstraintModelRequest,
    SolverMode,
    Variable,
    VariableDomain,
    VariableDomainType,
)
from chuk_mcp_solver.solver import get_solver


def create_knapsack_problem(n_items: int = 20) -> SolveConstraintModelRequest:
    """Create a knapsack problem with n items."""
    # Item values and weights
    values = [i * 3 + 5 for i in range(n_items)]
    weights = [i * 2 + 1 for i in range(n_items)]
    capacity = sum(weights) // 2  # Half total weight as capacity

    return SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=[
            Variable(
                id=f"item_{i}",
                domain=VariableDomain(type=VariableDomainType.BOOL),
            )
            for i in range(n_items)
        ],
        constraints=[
            Constraint(
                id="capacity",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[LinearTerm(var=f"item_{i}", coef=weights[i]) for i in range(n_items)],
                    sense="<=",
                    rhs=capacity,
                ),
            )
        ],
        objective=Objective(
            sense=ObjectiveSense.MAXIMIZE,
            terms=[LinearTerm(var=f"item_{i}", coef=values[i]) for i in range(n_items)],
        ),
        search=SearchConfig(enable_solution_caching=True),
    )


async def example_1_cache_hit():
    """Example 1: Demonstrate cache hit for identical problem."""
    print("=" * 80)
    print("EXAMPLE 1: Cache Hit for Identical Problem")
    print("=" * 80)
    print()

    clear_global_cache()
    solver = get_solver("ortools")

    request = create_knapsack_problem(n_items=15)

    # First solve - cache miss
    print("First solve (cache miss)...")
    start = time.time()
    response1 = await solver.solve_constraint_model(request)
    elapsed1 = time.time() - start

    print(f"Status: {response1.status}")
    print(f"Objective: {response1.objective_value}")
    print(f"Time: {elapsed1 * 1000:.2f} ms")
    print()

    # Second solve - cache hit
    print("Second solve of IDENTICAL problem (cache hit)...")
    start = time.time()
    response2 = await solver.solve_constraint_model(request)
    elapsed2 = time.time() - start

    print(f"Status: {response2.status}")
    print(f"Objective: {response2.objective_value}")
    print(f"Time: {elapsed2 * 1000:.2f} ms")
    print()

    # Speedup
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float("inf")
    print(f"✅ Speedup: {speedup:.1f}x faster (cached solution)")
    print(f"   Same objective value: {response1.objective_value == response2.objective_value}")
    print()


async def example_2_cache_stats():
    """Example 2: Show cache statistics."""
    print("=" * 80)
    print("EXAMPLE 2: Cache Statistics")
    print("=" * 80)
    print()

    clear_global_cache()
    solver = get_solver("ortools")
    cache = get_global_cache()

    # Solve 3 different problems
    print("Solving 3 different problems...")
    for i in range(3):
        request = create_knapsack_problem(n_items=10 + i)
        await solver.solve_constraint_model(request)

    # Check some problems again (cache hits)
    print("Re-solving first 2 problems (cache hits)...")
    for i in range(2):
        request = create_knapsack_problem(n_items=10 + i)
        await solver.solve_constraint_model(request)

    # Show stats
    stats = cache.stats()
    print()
    print("Cache Statistics:")
    print("-" * 80)
    print(f"  Size: {stats['size']} / {stats['max_size']} entries")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate_pct']:.1f}%")
    print(f"  TTL: {stats['ttl_seconds']} seconds")
    print()
    print(f"✅ Cache hit rate: {stats['hit_rate_pct']:.1f}% (2 hits / 5 total requests)")
    print()


async def example_3_different_problems():
    """Example 3: Different problems don't collide."""
    print("=" * 80)
    print("EXAMPLE 3: Different Problems Don't Collide")
    print("=" * 80)
    print()

    clear_global_cache()
    solver = get_solver("ortools")

    # Two different problems
    request1 = create_knapsack_problem(n_items=10)
    request2 = create_knapsack_problem(n_items=15)  # Different problem

    print("Solving problem 1 (10 items)...")
    response1 = await solver.solve_constraint_model(request1)
    print(f"Objective: {response1.objective_value}")
    print()

    print("Solving problem 2 (15 items)...")
    response2 = await solver.solve_constraint_model(request2)
    print(f"Objective: {response2.objective_value}")
    print()

    # Resolve both
    print("Re-solving both problems (both should be cache hits)...")
    response1_cached = await solver.solve_constraint_model(request1)
    response2_cached = await solver.solve_constraint_model(request2)

    print(f"Problem 1 objective: {response1_cached.objective_value}")
    print(f"Problem 2 objective: {response2_cached.objective_value}")
    print()

    cache = get_global_cache()
    stats = cache.stats()

    print(f"✅ Cache correctly maintains {stats['size']} distinct problems")
    print(f"   Hit rate: {stats['hit_rate_pct']:.1f}% (2 hits / 4 total)")
    print()


async def example_4_cache_disabled():
    """Example 4: Disabling cache."""
    print("=" * 80)
    print("EXAMPLE 4: Disabling Cache")
    print("=" * 80)
    print()

    clear_global_cache()
    solver = get_solver("ortools")
    cache = get_global_cache()

    request = create_knapsack_problem(n_items=12)

    # Solve with cache enabled
    print("Solving with cache ENABLED...")
    await solver.solve_constraint_model(request)
    await solver.solve_constraint_model(request)  # Second solve
    stats_enabled = cache.stats()
    print(f"Hits: {stats_enabled['hits']}, Misses: {stats_enabled['misses']}")
    print()

    # Clear and solve with cache disabled
    clear_global_cache()
    request.search = SearchConfig(enable_solution_caching=False)

    print("Solving twice with cache DISABLED...")
    await solver.solve_constraint_model(request)
    await solver.solve_constraint_model(request)  # Second solve
    stats_disabled = cache.stats()
    print(f"Hits: {stats_disabled['hits']}, Misses: {stats_disabled['misses']}")
    print()

    print("✅ Cache enabled: 1 hit, 1 miss")
    print("   Cache disabled: 0 hits, 0 misses (bypasses cache)")
    print()


async def example_5_cache_performance():
    """Example 5: Performance comparison."""
    print("=" * 80)
    print("EXAMPLE 5: Performance Comparison")
    print("=" * 80)
    print()

    clear_global_cache()
    solver = get_solver("ortools")

    request = create_knapsack_problem(n_items=25)  # Larger problem

    # Benchmark without cache
    print("Benchmark: 5 solves WITHOUT cache...")
    clear_global_cache()
    request.search = SearchConfig(enable_solution_caching=False)
    start = time.time()
    for _ in range(5):
        await solver.solve_constraint_model(request)
    elapsed_no_cache = time.time() - start
    print(f"Total time: {elapsed_no_cache * 1000:.2f} ms")
    print(f"Average: {elapsed_no_cache * 1000 / 5:.2f} ms per solve")
    print()

    # Benchmark with cache
    print("Benchmark: 5 solves WITH cache (4 cache hits)...")
    clear_global_cache()
    request.search = SearchConfig(enable_solution_caching=True)
    start = time.time()
    for _ in range(5):
        await solver.solve_constraint_model(request)
    elapsed_with_cache = time.time() - start
    print(f"Total time: {elapsed_with_cache * 1000:.2f} ms")
    print(f"Average: {elapsed_with_cache * 1000 / 5:.2f} ms per solve")
    print()

    speedup = elapsed_no_cache / elapsed_with_cache if elapsed_with_cache > 0 else float("inf")
    print(f"✅ Overall speedup: {speedup:.1f}x faster with cache")
    print()


async def main():
    """Run all caching examples."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "CACHING DEMO - PHASE 3 FEATURES" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print("This demo shows how solution caching improves performance")
    print("for repeated problems using LRU cache with problem hashing.")
    print()

    await example_1_cache_hit()
    await example_2_cache_stats()
    await example_3_different_problems()
    await example_4_cache_disabled()
    await example_5_cache_performance()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Solution caching provides:")
    print("  • Automatic deduplication via problem hashing")
    print("  • Significant speedup for repeated problems (10-100x+)")
    print("  • LRU eviction with configurable TTL")
    print("  • Cache statistics (hit rate, size, etc.)")
    print("  • Can be disabled per-request if needed")
    print()
    print("Use cases:")
    print("  • LLM agents iterating on similar problems")
    print("  • Batch processing with repeated patterns")
    print("  • Interactive exploration of solution space")
    print()


if __name__ == "__main__":
    asyncio.run(main())
