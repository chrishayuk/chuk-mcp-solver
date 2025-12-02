"""Validation Demo - Shows how the solver helps LLMs self-correct.

This example demonstrates the Phase 2 validation features:
- Pre-solve model validation
- Actionable error messages
- Smart typo detection with suggestions
- Three-level severity (ERROR, WARNING, INFO)

Run: python examples/validation_demo.py
"""

import asyncio

from chuk_mcp_solver.models import (
    Constraint,
    ConstraintKind,
    LinearConstraintParams,
    LinearTerm,
    Objective,
    ObjectiveSense,
    SolveConstraintModelRequest,
    SolverMode,
    Variable,
    VariableDomain,
    VariableDomainType,
)
from chuk_mcp_solver.solver import get_solver


async def example_1_undefined_variable():
    """Example 1: Undefined variable with typo detection."""
    print("=" * 80)
    print("EXAMPLE 1: Undefined Variable with Typo Detection")
    print("=" * 80)
    print()

    solver = get_solver("ortools")

    # Model with typo: references 'y' but variable is named 'x'
    request = SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            )
        ],
        constraints=[
            Constraint(
                id="c1",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[LinearTerm(var="y", coef=1)],  # Typo: 'y' instead of 'x'
                    sense="<=",
                    rhs=5,
                ),
            )
        ],
        objective=Objective(sense=ObjectiveSense.MAXIMIZE, terms=[LinearTerm(var="x", coef=1)]),
    )

    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}")
    print()
    print("Validation Error Message:")
    print("-" * 80)
    if response.explanation:
        print(response.explanation.summary)
    print()
    print("✅ Notice how the error message suggests 'Did you mean x?'")
    print()


async def example_2_invalid_bounds():
    """Example 2: Invalid domain bounds."""
    print("=" * 80)
    print("EXAMPLE 2: Invalid Domain Bounds")
    print("=" * 80)
    print()

    solver = get_solver("ortools")

    # Model with invalid bounds: lower > upper
    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(
                    type=VariableDomainType.INTEGER,
                    lower=10,
                    upper=5,  # Invalid!
                ),
            )
        ],
        constraints=[
            Constraint(
                id="c1",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[LinearTerm(var="x", coef=1)], sense="<=", rhs=20
                ),
            )
        ],
    )

    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}")
    print()
    print("Validation Error Message:")
    print("-" * 80)
    if response.explanation:
        print(response.explanation.summary)
    print()
    print("✅ Clear error message explaining the problem")
    print()


async def example_3_duplicate_ids():
    """Example 3: Duplicate variable IDs."""
    print("=" * 80)
    print("EXAMPLE 3: Duplicate Variable IDs")
    print("=" * 80)
    print()

    solver = get_solver("ortools")

    # Model with duplicate variable IDs
    request = SolveConstraintModelRequest(
        mode=SolverMode.SATISFY,
        variables=[
            Variable(
                id="x",
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=10),
            ),
            Variable(
                id="x",  # Duplicate ID!
                domain=VariableDomain(type=VariableDomainType.INTEGER, lower=0, upper=20),
            ),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[LinearTerm(var="x", coef=1)], sense="<=", rhs=5
                ),
            )
        ],
    )

    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}")
    print()
    print("Validation Error Message:")
    print("-" * 80)
    if response.explanation:
        print(response.explanation.summary)
    print()
    print("✅ Detects duplicate IDs across variables")
    print()


async def example_4_valid_model():
    """Example 4: Valid model passes validation."""
    print("=" * 80)
    print("EXAMPLE 4: Valid Model (No Validation Errors)")
    print("=" * 80)
    print()

    solver = get_solver("ortools")

    # Valid model
    request = SolveConstraintModelRequest(
        mode=SolverMode.OPTIMIZE,
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
                id="budget",
                kind=ConstraintKind.LINEAR,
                params=LinearConstraintParams(
                    terms=[LinearTerm(var="x", coef=2), LinearTerm(var="y", coef=3)],
                    sense="<=",
                    rhs=20,
                ),
            )
        ],
        objective=Objective(
            sense=ObjectiveSense.MAXIMIZE,
            terms=[LinearTerm(var="x", coef=5), LinearTerm(var="y", coef=4)],
        ),
    )

    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}")
    print(f"Objective Value: {response.objective_value}")
    if response.solutions:
        print("\nSolution:")
        for var in response.solutions[0].variables:
            print(f"  {var.id} = {var.value}")
    print()
    print("✅ Valid model passes validation and solves successfully")
    print()


async def main():
    """Run all validation examples."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "VALIDATION DEMO - PHASE 2 FEATURES" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print("This demo shows how the solver validates models and provides")
    print("actionable error messages to help LLMs self-correct.")
    print()

    await example_1_undefined_variable()
    await example_2_invalid_bounds()
    await example_3_duplicate_ids()
    await example_4_valid_model()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("The validation framework helps LLMs by:")
    print("  • Detecting errors before expensive solve operations")
    print("  • Providing specific locations of errors")
    print("  • Suggesting corrections (e.g., 'Did you mean x?')")
    print("  • Using three severity levels (ERROR, WARNING, INFO)")
    print()
    print("This enables LLMs to self-correct and iterate toward valid models.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
