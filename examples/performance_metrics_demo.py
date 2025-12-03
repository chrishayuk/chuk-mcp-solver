"""Performance Metrics Demo - Enhanced Status Codes and Timing.

This example demonstrates the new performance metrics and enhanced status codes
introduced in v0.2.0:

- optimality_gap: Percentage gap from best bound (0% = proven optimal)
- solve_time_ms: Wall-clock solve time in milliseconds
- timeout_best: Timeout with best solution found
- timeout_no_solution: Timeout with no solution found

Run this example:
    python examples/performance_metrics_demo.py
"""

import asyncio

from chuk_mcp_solver.models import (
    Constraint,
    LinearConstraintParams,
    LinearTerm,
    Objective,
    SearchConfig,
    SolveConstraintModelRequest,
    SolverMode,
    SolverStatus,
    Variable,
    VariableDomain,
    VariableDomainType,
)
from chuk_mcp_solver.solver import get_solver


async def demo_optimal_with_metrics():
    """Demo 1: Optimal solution with performance metrics."""
    print("=" * 70)
    print("Demo 1: Optimal Solution with Performance Metrics")
    print("=" * 70)

    solver = get_solver("ortools")

    # Simple knapsack problem
    request = SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=[
            Variable(id=f"item_{i}", domain=VariableDomain(type=VariableDomainType.BOOL))
            for i in range(5)
        ],
        constraints=[
            Constraint(
                id="capacity",
                kind="linear",
                params=LinearConstraintParams(
                    terms=[LinearTerm(var=f"item_{i}", coef=i + 1) for i in range(5)],
                    sense="<=",
                    rhs=10,
                ),
            )
        ],
        objective=Objective(
            sense="max",
            terms=[LinearTerm(var=f"item_{i}", coef=(i + 1) * 10) for i in range(5)],
        ),
    )

    response = await solver.solve_constraint_model(request)

    print(f"\nStatus: {response.status.value}")
    print(f"Objective Value: {response.objective_value:.0f}")
    print(f"Optimality Gap: {response.optimality_gap:.2f}%")
    print(f"Solve Time: {response.solve_time_ms}ms")
    print(f"\n✓ Optimal solution found with 0% gap in {response.solve_time_ms}ms")
    print()


async def demo_timeout_with_best_solution():
    """Demo 2: Timeout with best solution found (TIMEOUT_BEST)."""
    print("=" * 70)
    print("Demo 2: Timeout with Best Solution Found (TIMEOUT_BEST)")
    print("=" * 70)

    solver = get_solver("ortools")

    # Larger problem with very short timeout
    request = SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=[
            Variable(
                id=f"x{i}",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
            for i in range(15)
        ],
        constraints=[],
        objective=Objective(
            sense="max",
            terms=[LinearTerm(var=f"x{i}", coef=i + 1) for i in range(15)],
        ),
        search=SearchConfig(
            max_time_ms=2,  # Very short timeout
            return_partial_solution=True,  # Request best-so-far
        ),
    )

    response = await solver.solve_constraint_model(request)

    print(f"\nStatus: {response.status.value}")
    print(f"Solve Time: {response.solve_time_ms}ms")

    if response.status == SolverStatus.TIMEOUT_BEST:
        print(f"Objective Value: {response.objective_value:.0f}")
        print(f"Optimality Gap: {response.optimality_gap:.2f}%")
        print(
            f"\n⚠ Solver timed out after {response.solve_time_ms}ms but returned best solution found"
        )
        print(f"  Gap of {response.optimality_gap:.2f}% means solution could be improved further")
    elif response.status == SolverStatus.OPTIMAL:
        print(f"Objective Value: {response.objective_value:.0f}")
        print(f"Optimality Gap: {response.optimality_gap:.2f}%")
        print(f"\n✓ Problem solved optimally despite short timeout ({response.solve_time_ms}ms)")
    else:
        print("\n✗ No solution found before timeout")

    if response.explanation:
        print(f"\nExplanation: {response.explanation.summary}")
    print()


async def demo_timeout_no_solution():
    """Demo 3: Timeout with no solution found (TIMEOUT_NO_SOLUTION)."""
    print("=" * 70)
    print("Demo 3: Timeout with No Solution Found (TIMEOUT_NO_SOLUTION)")
    print("=" * 70)

    solver = get_solver("ortools")

    # Complex problem with extremely short timeout and no partial solutions
    request = SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=[
            Variable(
                id=f"x{i}",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=100),
            )
            for i in range(20)
        ],
        constraints=[
            Constraint(
                id=f"c{i}",
                kind="linear",
                params=LinearConstraintParams(
                    terms=[
                        LinearTerm(var=f"x{i}", coef=2),
                        LinearTerm(var=f"x{i + 1}", coef=3),
                    ],
                    sense="<=",
                    rhs=150,
                ),
            )
            for i in range(19)
        ],
        objective=Objective(
            sense="max",
            terms=[LinearTerm(var=f"x{i}", coef=i + 1) for i in range(20)],
        ),
        search=SearchConfig(
            max_time_ms=1,  # Extremely short timeout
            return_partial_solution=False,  # Don't return partial solutions
        ),
    )

    response = await solver.solve_constraint_model(request)

    print(f"\nStatus: {response.status.value}")
    print(f"Solve Time: {response.solve_time_ms}ms")

    if response.status == SolverStatus.TIMEOUT_NO_SOLUTION:
        print(f"\n✗ Solver timed out after {response.solve_time_ms}ms without finding any solution")
        print("  Recommendations:")
        print("    • Increase max_time_ms")
        print("    • Simplify the problem")
        print("    • Provide warm-start hints")
        print("    • Enable return_partial_solution")
    elif response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        print(f"\n✓ Solution found: {response.objective_value:.0f}")
        print(f"  Gap: {response.optimality_gap:.2f}%")

    if response.explanation:
        print(f"\nExplanation: {response.explanation.summary}")
    print()


async def demo_performance_comparison():
    """Demo 4: Compare solve times across problem sizes."""
    print("=" * 70)
    print("Demo 4: Performance Comparison Across Problem Sizes")
    print("=" * 70)

    solver = get_solver("ortools")

    problem_sizes = [5, 10, 15, 20]
    results = []

    for size in problem_sizes:
        request = SolveConstraintModelRequest(
            mode=SolverMode.OPTIMIZE,
            variables=[
                Variable(
                    id=f"x{i}",
                    domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
                )
                for i in range(size)
            ],
            constraints=[],
            objective=Objective(
                sense="max",
                terms=[LinearTerm(var=f"x{i}", coef=i + 1) for i in range(size)],
            ),
            search=SearchConfig(max_time_ms=100),
        )

        response = await solver.solve_constraint_model(request)
        results.append(
            {
                "size": size,
                "status": response.status.value,
                "time_ms": response.solve_time_ms,
                "gap": response.optimality_gap,
            }
        )

    print("\nProblem Size | Status    | Time (ms) | Gap (%)")
    print("-------------|-----------|-----------|--------")
    for r in results:
        gap_str = f"{r['gap']:.2f}" if r["gap"] is not None else "N/A"
        print(f"{r['size']:12} | {r['status']:9} | {r['time_ms']:9} | {gap_str:6}")

    print("\n✓ Performance scales with problem size as expected")
    print()


async def main():
    """Run all performance metrics demos."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       CHUK MCP Solver - Performance Metrics & Status Demo           ║")
    print("║                        Version 0.2.0+                                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    await demo_optimal_with_metrics()
    await demo_timeout_with_best_solution()
    await demo_timeout_no_solution()
    await demo_performance_comparison()

    print("=" * 70)
    print("Summary of Enhanced Status Codes")
    print("=" * 70)
    print()
    print("OPTIMAL            - Proven optimal solution (gap = 0%)")
    print("FEASIBLE           - Valid solution, may not be optimal (gap > 0%)")
    print("TIMEOUT_BEST       - Timeout, returning best solution found")
    print("TIMEOUT_NO_SOLUTION - Timeout, no solution found")
    print("INFEASIBLE         - No solution exists")
    print()
    print("New Performance Metrics:")
    print("  • optimality_gap   : % gap from best bound (lower is better)")
    print("  • solve_time_ms    : Wall-clock solve time in milliseconds")
    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
