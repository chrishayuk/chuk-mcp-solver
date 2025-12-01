"""Example: Delivery route optimization using circuit constraints.

Demonstrates circuit constraints for solving Traveling Salesman Problem (TSP)
and vehicle routing problems.
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_tsp_model(locations: list[str], distances: dict[tuple[int, int], int]) -> dict:
    """Build a Traveling Salesman Problem model using circuit constraints.

    Args:
        locations: List of location names
        distances: Dict mapping (from_idx, to_idx) -> distance

    Returns:
        Model dictionary ready for SolveConstraintModelRequest.
    """
    variables = []
    constraints = []
    n = len(locations)

    # Create boolean variables for each possible arc (edge in the tour)
    arc_vars = []
    distance_terms = []

    for i in range(n):
        for j in range(n):
            if i != j:  # No self-loops
                arc_id = f"arc_{i}_{j}"
                variables.append(
                    {
                        "id": arc_id,
                        "domain": {"type": "bool"},
                        "metadata": {
                            "from": locations[i],
                            "to": locations[j],
                            "distance": distances.get((i, j), 999),
                        },
                    }
                )
                arc_vars.append((i, j, arc_id))

                # Add to objective (minimize total distance)
                distance = distances.get((i, j), 999)
                distance_terms.append({"var": arc_id, "coef": distance})

    # Add circuit constraint - ensures a valid Hamiltonian circuit
    constraints.append(
        {
            "id": "hamilton_circuit",
            "kind": "circuit",
            "params": {"arcs": arc_vars},
            "metadata": {"description": "Must form a complete tour visiting each city once"},
        }
    )

    return {
        "mode": "optimize",
        "variables": variables,
        "constraints": constraints,
        "objective": {
            "sense": "min",
            "terms": distance_terms,
        },
    }


async def main():
    """Run delivery routing example."""
    print("=== Delivery Route Optimization (TSP with Circuit Constraints) ===\n")

    # Define locations
    locations = ["Warehouse", "Customer_A", "Customer_B", "Customer_C", "Customer_D"]

    # Define distances between locations (in km)
    # This is a symmetric distance matrix
    distance_data = {
        (0, 1): 10,
        (1, 0): 10,  # Warehouse <-> Customer_A
        (0, 2): 15,
        (2, 0): 15,  # Warehouse <-> Customer_B
        (0, 3): 20,
        (3, 0): 20,  # Warehouse <-> Customer_C
        (0, 4): 12,
        (4, 0): 12,  # Warehouse <-> Customer_D
        (1, 2): 8,
        (2, 1): 8,  # Customer_A <-> Customer_B
        (1, 3): 18,
        (3, 1): 18,  # Customer_A <-> Customer_C
        (1, 4): 14,
        (4, 1): 14,  # Customer_A <-> Customer_D
        (2, 3): 9,
        (3, 2): 9,  # Customer_B <-> Customer_C
        (2, 4): 11,
        (4, 2): 11,  # Customer_B <-> Customer_D
        (3, 4): 7,
        (4, 3): 7,  # Customer_C <-> Customer_D
    }

    print("Locations:")
    for idx, loc in enumerate(locations):
        print(f"  {idx}: {loc}")

    print("\nDistance Matrix (km):")
    print("     ", end="")
    for j in range(len(locations)):
        print(f"{j:4}", end="")
    print()

    for i in range(len(locations)):
        print(f"{i:4} ", end="")
        for j in range(len(locations)):
            if i == j:
                print("   -", end="")
            else:
                dist = distance_data.get((i, j), 999)
                print(f"{dist:4}", end="")
        print()

    print("\nFinding optimal delivery route...\n")

    # Build and solve
    model = build_tsp_model(locations, distance_data)
    request = SolveConstraintModelRequest(**model)

    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}")

    if response.status.value in ["optimal", "feasible"]:
        # Extract solution
        total_distance = response.objective_value

        print(f"Minimum Total Distance: {int(total_distance)} km")
        print(f"Objective Value: {response.objective_value}\n")

        # Reconstruct the tour
        # Find which arcs are selected (value = 1)
        arcs = {}
        for var in response.solutions[0].variables:
            if var.value == 1 and var.id.startswith("arc_"):
                parts = var.id.split("_")
                from_idx = int(parts[1])
                to_idx = int(parts[2])
                arcs[from_idx] = to_idx

        # Build tour starting from warehouse (index 0)
        tour = [0]
        current = 0
        while len(tour) < len(locations):
            next_city = arcs[current]
            tour.append(next_city)
            current = next_city

        print("Optimal Delivery Route:")
        print("Step | Location        | Distance to Next")
        print("-----|-----------------|------------------")

        for i, city_idx in enumerate(tour):
            location = locations[city_idx]
            if i < len(tour) - 1:
                next_idx = tour[i + 1]
                distance = distance_data.get((city_idx, next_idx), 0)
                print(f"{i + 1:4} | {location:15} | {distance:4} km")
            else:
                # Return to start
                next_idx = tour[0]
                distance = distance_data.get((city_idx, next_idx), 0)
                print(f"{i + 1:4} | {location:15} | {distance:4} km (return)")

        print(f"\nRoute: {' → '.join(locations[i] for i in tour)} → {locations[tour[0]]}")

        # Show explanation
        if response.explanation:
            print(f"\n{response.explanation.summary}")

    else:
        print(f"Could not find solution: {response.status}")
        if response.explanation:
            print(f"\n{response.explanation.summary}")


if __name__ == "__main__":
    asyncio.run(main())
