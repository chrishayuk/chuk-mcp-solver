"""Example: 0/1 Knapsack problem.

Demonstrates binary variables and classic optimization.
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_knapsack_model(items: list[dict], capacity: int) -> dict:
    """Build a 0/1 knapsack model.

    Args:
        items: List of items with id, weight, and value.
        capacity: Maximum weight capacity.

    Returns:
        Model dictionary.
    """
    variables = []
    constraints = []

    # Binary variable for each item (take it or not)
    for item in items:
        variables.append(
            {
                "id": f"take_{item['id']}",
                "domain": {"type": "bool"},
                "metadata": {
                    "item_id": item["id"],
                    "weight": item["weight"],
                    "value": item["value"],
                },
            }
        )

    # Capacity constraint: total weight <= capacity
    constraints.append(
        {
            "id": "capacity",
            "kind": "linear",
            "params": {
                "terms": [{"var": f"take_{item['id']}", "coef": item["weight"]} for item in items],
                "sense": "<=",
                "rhs": capacity,
            },
            "metadata": {"description": f"Total weight cannot exceed {capacity}"},
        }
    )

    # Objective: maximize total value
    objective = {
        "sense": "max",
        "terms": [{"var": f"take_{item['id']}", "coef": item["value"]} for item in items],
        "metadata": {"description": "Maximize total value"},
    }

    return {
        "mode": "optimize",
        "variables": variables,
        "constraints": constraints,
        "objective": objective,
    }


def display_solution(solution_vars: list, items: list[dict], capacity: int) -> None:
    """Display the knapsack solution.

    Args:
        solution_vars: List of SolutionVariable objects.
        items: Original item definitions.
        capacity: Capacity limit.
    """
    # Extract selected items
    selected = []
    total_weight = 0
    total_value = 0

    for var in solution_vars:
        if var.value == 1:  # Item is selected
            item_id = var.metadata["item_id"]
            weight = var.metadata["weight"]
            value = var.metadata["value"]
            selected.append({"id": item_id, "weight": weight, "value": value})
            total_weight += weight
            total_value += value

    print("\nOptimal Selection:")
    print(f"Capacity: {capacity}")
    print(f"Used Weight: {total_weight}/{capacity}")
    print(f"Total Value: {total_value}\n")

    if selected:
        print("Selected Items:")
        print("Item | Weight | Value")
        print("-----|--------|-------")
        for item in selected:
            print(f"{item['id']:4} | {item['weight']:6} | {item['value']:5}")
    else:
        print("No items selected.")


async def main() -> None:
    """Run the knapsack optimizer example."""
    print("=== 0/1 Knapsack Optimizer Example ===\n")

    # Define items
    items = [
        {"id": "A", "weight": 3, "value": 10},
        {"id": "B", "weight": 5, "value": 15},
        {"id": "C", "weight": 2, "value": 8},
        {"id": "D", "weight": 4, "value": 12},
        {"id": "E", "weight": 1, "value": 5},
    ]

    capacity = 7

    print("Items:")
    print("Item | Weight | Value")
    print("-----|--------|-------")
    for item in items:
        print(f"{item['id']:4} | {item['weight']:6} | {item['value']:5}")

    print(f"\nKnapsack Capacity: {capacity}")

    # Build model
    model_dict = build_knapsack_model(items, capacity)
    request = SolveConstraintModelRequest(**model_dict)

    # Solve
    print("\nOptimizing...")
    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    # Display results
    print(f"\nStatus: {response.status}")
    if response.objective_value is not None:
        print(f"Objective Value: {response.objective_value:.0f}")

    if response.solutions:
        display_solution(response.solutions[0].variables, items, capacity)

    if response.explanation:
        print(f"\n{response.explanation.summary}")
        if response.explanation.binding_constraints:
            print("\nBinding Constraint:")
            bc = response.explanation.binding_constraints[0]
            print(
                f"  {bc.metadata.get('description', bc.id)} (used: {bc.lhs_value:.0f}/{bc.rhs:.0f})"
            )


if __name__ == "__main__":
    asyncio.run(main())
