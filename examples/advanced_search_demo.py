"""Advanced Search Demo - Partial solutions and search strategies.

This example demonstrates Phase 3 advanced search features:
- Partial solutions (best-so-far on timeout)
- Search strategy hints (first-fail, random, etc.)
- Deterministic solving with random seeds
- Warm-start solution hints

Run: python examples/advanced_search_demo.py
"""

import asyncio

from chuk_mcp_solver.models import (
    Constraint,
    ConstraintKind,
    LinearConstraintParams,
    LinearTerm,
    Objective,
    ObjectiveSense,
    SearchConfig,
    SearchStrategy,
    SolveConstraintModelRequest,
    SolverMode,
    Variable,
    VariableDomain,
    VariableDomainType,
)
from chuk_mcp_solver.solver import get_solver


def create_large_knapsack(n_items: int = 100) -> SolveConstraintModelRequest:
    """Create a large knapsack problem for timeout demonstrations."""
    values = [i * 7 + 3 for i in range(n_items)]
    weights = [i * 3 + 1 for i in range(n_items)]
    capacity = sum(weights) // 3

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
    )


async def example_1_partial_solution():
    """Example 1: Return best solution found on timeout."""
    print("=" * 80)
    print("EXAMPLE 1: Partial Solution on Timeout")
    print("=" * 80)
    print()

    solver = get_solver("ortools")

    request = create_large_knapsack(n_items=150)

    # Without partial solution (default)
    print("Solving with 100ms timeout, return_partial_solution=False...")
    request.search = SearchConfig(
        max_time_ms=100,
        return_partial_solution=False,
        enable_solution_caching=False,
    )
    response1 = await solver.solve_constraint_model(request)
    print(f"Status: {response1.status}")
    print(f"Objective: {response1.objective_value}")
    print(f"Has solution: {len(response1.solutions or []) > 0}")
    print()

    # With partial solution
    print("Solving with 100ms timeout, return_partial_solution=True...")
    request.search = SearchConfig(
        max_time_ms=100,
        return_partial_solution=True,
        enable_solution_caching=False,
    )
    response2 = await solver.solve_constraint_model(request)
    print(f"Status: {response2.status}")
    print(f"Objective: {response2.objective_value}")
    print(f"Has solution: {len(response2.solutions or []) > 0}")
    if response2.explanation:
        print(f"\nExplanation: {response2.explanation.summary[:100]}...")
    print()

    print("✅ With return_partial_solution=True, you get the best solution found")
    print("   even when timeout occurs (useful for anytime algorithms)")
    print()


async def example_2_search_strategies():
    """Example 2: Different search strategies."""
    print("=" * 80)
    print("EXAMPLE 2: Search Strategies")
    print("=" * 80)
    print()

    solver = get_solver("ortools")

    request = create_large_knapsack(n_items=50)

    strategies = [
        (SearchStrategy.AUTO, "Auto (solver chooses)"),
        (SearchStrategy.FIRST_FAIL, "First Fail (smallest domain first)"),
        (SearchStrategy.RANDOM, "Random (randomized search)"),
    ]

    print("Solving with different search strategies (500ms timeout each)...")
    print()

    results = []
    for strategy, description in strategies:
        request.search = SearchConfig(
            max_time_ms=500,
            strategy=strategy,
            return_partial_solution=True,
            enable_solution_caching=False,
        )
        response = await solver.solve_constraint_model(request)
        results.append((description, response.status, response.objective_value or 0))
        print(
            f"{description:40} Status: {response.status:12} Objective: {response.objective_value}"
        )

    print()
    print("✅ Different strategies can find different quality solutions")
    print("   in the same time budget. Choose based on problem structure.")
    print()


async def example_3_deterministic_solving():
    """Example 3: Deterministic solving with random seeds."""
    print("=" * 80)
    print("EXAMPLE 3: Deterministic Solving (Random Seeds)")
    print("=" * 80)
    print()

    solver = get_solver("ortools")

    request = create_large_knapsack(n_items=40)

    # Two solves with same seed
    print("Two solves with random_seed=42:")
    for i in range(2):
        request.search = SearchConfig(
            max_time_ms=200,
            random_seed=42,
            return_partial_solution=True,
            enable_solution_caching=False,
        )
        response = await solver.solve_constraint_model(request)
        print(f"  Solve {i + 1}: Status={response.status}, Objective={response.objective_value}")

    print()

    # Two solves with different seeds
    print("Two solves with different seeds:")
    for i, seed in enumerate([123, 456]):
        request.search = SearchConfig(
            max_time_ms=200,
            random_seed=seed,
            return_partial_solution=True,
            enable_solution_caching=False,
        )
        response = await solver.solve_constraint_model(request)
        print(
            f"  Solve {i + 1} (seed={seed}): Status={response.status}, Objective={response.objective_value}"
        )

    print()
    print("✅ Same seed = deterministic results (useful for debugging)")
    print("   Different seeds = explore different search paths")
    print()


async def example_4_warm_start():
    """Example 4: Warm-start from previous solution."""
    print("=" * 80)
    print("EXAMPLE 4: Warm-Start Solution Hints")
    print("=" * 80)
    print()

    solver = get_solver("ortools")

    request = create_large_knapsack(n_items=30)

    # First solve without warm-start
    print("Solving without warm-start (300ms timeout)...")
    request.search = SearchConfig(
        max_time_ms=300,
        return_partial_solution=True,
        enable_solution_caching=False,
    )
    response1 = await solver.solve_constraint_model(request)
    print(f"Status: {response1.status}")
    print(f"Objective: {response1.objective_value}")
    print()

    # Extract solution for warm-start
    if response1.solutions:
        warm_start = {var.id: var.value for var in response1.solutions[0].variables}

        # Solve again with warm-start
        print("Solving WITH warm-start from previous solution (100ms timeout)...")
        request.search = SearchConfig(
            max_time_ms=100,
            warm_start_solution=warm_start,
            return_partial_solution=True,
            enable_solution_caching=False,
        )
        response2 = await solver.solve_constraint_model(request)
        print(f"Status: {response2.status}")
        print(f"Objective: {response2.objective_value}")
        print()

        print("✅ Warm-start gives solver a good starting point")
        print("   Can find better solutions faster or prove optimality quicker")
    print()


async def example_5_anytime_algorithm():
    """Example 5: Anytime algorithm - improving solutions over time."""
    print("=" * 80)
    print("EXAMPLE 5: Anytime Algorithm (Progressive Improvement)")
    print("=" * 80)
    print()

    solver = get_solver("ortools")

    request = create_large_knapsack(n_items=80)

    # Solve with increasing time budgets
    time_budgets = [50, 100, 200, 500, 1000]

    print("Solving with increasing time budgets (anytime algorithm):")
    print()

    best_objective = 0
    for time_ms in time_budgets:
        request.search = SearchConfig(
            max_time_ms=time_ms,
            return_partial_solution=True,
            enable_solution_caching=False,
        )
        response = await solver.solve_constraint_model(request)

        improvement = ""
        if response.objective_value and response.objective_value > best_objective:
            improvement = f" (+{response.objective_value - best_objective})"
            best_objective = response.objective_value

        print(
            f"  {time_ms:5}ms: Status={response.status:12} Objective={response.objective_value}{improvement}"
        )

    print()
    print("✅ Anytime algorithm: solution quality improves with more time")
    print("   Can stop early if good-enough solution found")
    print("   Useful for interactive/real-time applications")
    print()


async def main():
    """Run all advanced search examples."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "ADVANCED SEARCH DEMO - PHASE 3 FEATURES" + " " * 21 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print("This demo shows advanced search features:")
    print("  • Partial solutions (best-so-far on timeout)")
    print("  • Search strategy hints")
    print("  • Deterministic solving")
    print("  • Warm-start hints")
    print("  • Anytime algorithms")
    print()

    await example_1_partial_solution()
    await example_2_search_strategies()
    await example_3_deterministic_solving()
    await example_4_warm_start()
    await example_5_anytime_algorithm()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Advanced search features enable:")
    print()
    print("Partial Solutions:")
    print("  • Get best solution found even on timeout")
    print("  • Useful for anytime algorithms")
    print("  • Interactive applications with time limits")
    print()
    print("Search Strategies:")
    print("  • Guide solver's variable selection")
    print("  • first_fail: good for highly constrained problems")
    print("  • random: diversification, avoid local patterns")
    print()
    print("Deterministic Solving:")
    print("  • Reproducible results with random_seed")
    print("  • Essential for debugging and testing")
    print()
    print("Warm-Start:")
    print("  • Bootstrap from known good solution")
    print("  • Faster convergence to better solutions")
    print("  • Useful for iterative refinement")
    print()


if __name__ == "__main__":
    asyncio.run(main())
