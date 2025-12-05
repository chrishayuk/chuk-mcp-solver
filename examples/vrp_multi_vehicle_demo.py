"""Multi-vehicle VRP example demonstrating Phase 1 functionality.

This example shows how to solve vehicle routing problems with multiple vehicles,
capacity constraints, and different optimization objectives.
"""

import asyncio

from chuk_mcp_solver.server import solve_routing_problem


async def example_1_basic_vrp():
    """Example 1: Basic multi-vehicle routing with capacity constraints."""

    print("\n" + "=" * 70)
    print("Example 1: Basic Multi-Vehicle VRP")
    print("=" * 70)
    print("\nProblem: Deliver to 5 customers using 2 trucks with capacity limits")

    locations = [
        {"id": "warehouse", "coordinates": (0, 0), "demand": 0},
        {"id": "customer_1", "coordinates": (10, 5), "demand": 15, "service_time": 10},
        {"id": "customer_2", "coordinates": (5, 10), "demand": 20, "service_time": 15},
        {"id": "customer_3", "coordinates": (15, 15), "demand": 25, "service_time": 10},
        {"id": "customer_4", "coordinates": (20, 10), "demand": 10, "service_time": 5},
        {"id": "customer_5", "coordinates": (12, 3), "demand": 18, "service_time": 12},
    ]

    vehicles = [
        {
            "id": "truck_1",
            "capacity": 50,
            "start_location": "warehouse",
            "cost_per_distance": 1.5,
            "fixed_cost": 50.0,
        },
        {
            "id": "truck_2",
            "capacity": 40,
            "start_location": "warehouse",
            "cost_per_distance": 1.2,
            "fixed_cost": 40.0,
        },
    ]

    response = await solve_routing_problem(
        locations=locations,
        vehicles=vehicles,
        objective="minimize_distance",
        max_time_ms=20000,
    )

    _print_solution(response)


async def example_2_minimize_vehicles():
    """Example 2: Minimize number of vehicles used."""

    print("\n" + "=" * 70)
    print("Example 2: Minimize Fleet Size")
    print("=" * 70)
    print("\nProblem: Use as few vehicles as possible to serve all customers")

    locations = [
        {"id": "depot", "coordinates": (0, 0), "demand": 0},
        {"id": "c1", "coordinates": (5, 0), "demand": 10},
        {"id": "c2", "coordinates": (10, 0), "demand": 10},
        {"id": "c3", "coordinates": (15, 0), "demand": 15},
        {"id": "c4", "coordinates": (20, 0), "demand": 12},
    ]

    vehicles = [
        {"id": "truck_A", "capacity": 30, "start_location": "depot", "fixed_cost": 100.0},
        {"id": "truck_B", "capacity": 30, "start_location": "depot", "fixed_cost": 100.0},
        {"id": "truck_C", "capacity": 30, "start_location": "depot", "fixed_cost": 100.0},
    ]

    response = await solve_routing_problem(
        locations=locations,
        vehicles=vehicles,
        objective="minimize_vehicles",
        max_time_ms=15000,
    )

    _print_solution(response)


async def example_3_minimize_cost():
    """Example 3: Minimize total cost (fixed + variable)."""

    print("\n" + "=" * 70)
    print("Example 3: Minimize Total Cost")
    print("=" * 70)
    print("\nProblem: Balance fixed vehicle costs with distance costs")

    locations = [
        {"id": "warehouse", "coordinates": (0, 0), "demand": 0},
        {"id": "store_1", "coordinates": (10, 0), "demand": 20},
        {"id": "store_2", "coordinates": (20, 0), "demand": 25},
        {"id": "store_3", "coordinates": (30, 0), "demand": 15},
    ]

    vehicles = [
        {
            "id": "small_van",
            "capacity": 30,
            "start_location": "warehouse",
            "cost_per_distance": 0.8,
            "fixed_cost": 30.0,
        },
        {
            "id": "large_truck",
            "capacity": 70,
            "start_location": "warehouse",
            "cost_per_distance": 1.5,
            "fixed_cost": 100.0,
        },
    ]

    response = await solve_routing_problem(
        locations=locations,
        vehicles=vehicles,
        objective="minimize_cost",
        max_time_ms=15000,
    )

    _print_solution(response)


async def example_4_capacity_infeasible():
    """Example 4: Demonstrate infeasible problem (demand exceeds capacity)."""

    print("\n" + "=" * 70)
    print("Example 4: Infeasible Problem (Capacity Exceeded)")
    print("=" * 70)
    print("\nProblem: Customer demand exceeds all vehicle capacities")

    locations = [
        {"id": "depot", "coordinates": (0, 0), "demand": 0},
        {"id": "bulk_order", "coordinates": (10, 0), "demand": 100},  # Too large!
    ]

    vehicles = [
        {"id": "small_truck", "capacity": 50, "start_location": "depot"},
        {"id": "medium_truck", "capacity": 60, "start_location": "depot"},
    ]

    response = await solve_routing_problem(
        locations=locations,
        vehicles=vehicles,
        objective="minimize_distance",
        max_time_ms=5000,
    )

    _print_solution(response)


def _print_solution(response):
    """Pretty-print routing solution."""
    print(f"\n{'Status:':<20} {response.status}")

    if response.status in ("optimal", "feasible", "timeout_best"):
        print(f"{'Vehicles Used:':<20} {response.vehicles_used}")
        print(f"{'Total Distance:':<20} {response.total_distance}")
        print(f"{'Total Time:':<20} {response.total_time}")
        print(f"{'Total Cost:':<20} ${response.total_cost:.2f}")
        print(f"{'Solve Time:':<20} {response.solve_time_ms}ms")

        if response.optimality_gap:
            print(f"{'Optimality Gap:':<20} {response.optimality_gap:.2f}%")

        print("\nRoutes:")
        for i, route in enumerate(response.routes, 1):
            print(f"\n  Route {i} - {route.vehicle_id}:")
            print(f"    Sequence: {' → '.join(route.sequence)}")
            print(f"    Distance: {route.total_distance}")
            print(f"    Time: {route.total_time}")
            print(f"    Cost: ${route.total_cost:.2f}")

            if route.load_timeline:
                print("    Load Timeline:")
                for loc, load in route.load_timeline:
                    print(f"      {loc}: {load} units")

        if response.explanation:
            print(f"\nExplanation: {response.explanation.summary}")

    else:
        # Infeasible, error, or timeout
        print(
            f"\n{'Explanation:':<20} {response.explanation.summary if response.explanation else 'No solution found'}"
        )

        if response.explanation and response.explanation.recommendations:
            print("\nRecommendations:")
            for rec in response.explanation.recommendations:
                print(f"  • {rec}")


async def main():
    """Run all VRP examples."""
    print("\n" + "=" * 70)
    print("MULTI-VEHICLE VRP EXAMPLES")
    print("=" * 70)

    await example_1_basic_vrp()
    await example_2_minimize_vehicles()
    await example_3_minimize_cost()
    await example_4_capacity_infeasible()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
