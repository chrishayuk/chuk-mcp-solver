"""Routing problem demonstration.

This example shows how to use the high-level solve_routing_problem tool
to solve TSP (Traveling Salesman Problem) and VRP (Vehicle Routing Problem)
without directly working with CP-SAT circuit constraints.
"""

import asyncio

from chuk_mcp_solver.server import solve_routing_problem


async def demo_simple_tsp():
    """Demo 1: Simple TSP with coordinates."""
    print("=" * 70)
    print("Demo 1: Simple TSP (Traveling Salesman Problem)")
    print("=" * 70)

    response = await solve_routing_problem(
        locations=[
            {"id": "warehouse", "coordinates": (0, 0)},
            {"id": "customer_A", "coordinates": (10, 5)},
            {"id": "customer_B", "coordinates": (5, 10)},
            {"id": "customer_C", "coordinates": (15, 15)},
            {"id": "customer_D", "coordinates": (20, 5)},
        ],
        objective="minimize_distance",
    )

    print(f"Status: {response.status}")
    print(f"Total Distance: {response.total_distance} units")
    print(f"Total Time: {response.total_time} units")
    print(f"Solve time: {response.solve_time_ms}ms")
    print("\nOptimal Route:")
    route = response.routes[0]
    print(f"  {' → '.join(route.sequence)} → {route.sequence[0]}")
    print(f"\n{response.explanation.summary}")
    print()


async def demo_tsp_with_distance_matrix():
    """Demo 2: TSP with explicit distance matrix."""
    print("=" * 70)
    print("Demo 2: TSP with Distance Matrix")
    print("=" * 70)

    # Asymmetric distances (e.g., one-way streets, traffic patterns)
    distance_matrix = [
        [0, 10, 15, 20],  # From A to others
        [12, 0, 35, 25],  # From B to others
        [15, 30, 0, 30],  # From C to others
        [18, 22, 28, 0],  # From D to others
    ]

    response = await solve_routing_problem(
        locations=[
            {"id": "A"},
            {"id": "B"},
            {"id": "C"},
            {"id": "D"},
        ],
        distance_matrix=distance_matrix,
        objective="minimize_distance",
    )

    print(f"Status: {response.status}")
    print(f"Total Distance: {response.total_distance} units")
    print("\nDistance Matrix:")
    print("     A   B   C   D")
    for i, row in enumerate(distance_matrix):
        loc = ["A", "B", "C", "D"][i]
        print(f"  {loc}  {row[0]:3} {row[1]:3} {row[2]:3} {row[3]:3}")

    print("\nOptimal Route:")
    route = response.routes[0]
    sequence = route.sequence
    for i in range(len(sequence)):
        from_loc = sequence[i]
        to_loc = sequence[(i + 1) % len(sequence)]
        from_idx = ["A", "B", "C", "D"].index(from_loc)
        to_idx = ["A", "B", "C", "D"].index(to_loc)
        dist = distance_matrix[from_idx][to_idx]
        print(f"  {from_loc} → {to_loc}: {dist} units")
    print()


async def demo_tsp_with_service_times():
    """Demo 3: TSP with service times at locations."""
    print("=" * 70)
    print("Demo 3: TSP with Service Times")
    print("=" * 70)

    response = await solve_routing_problem(
        locations=[
            {"id": "depot", "coordinates": (0, 0), "service_time": 0},
            {"id": "store_A", "coordinates": (10, 0), "service_time": 15},  # 15 min unloading
            {"id": "store_B", "coordinates": (10, 10), "service_time": 20},  # 20 min unloading
            {"id": "store_C", "coordinates": (0, 10), "service_time": 10},  # 10 min unloading
        ],
        objective="minimize_time",
    )

    print(f"Status: {response.status}")
    print(f"Total Travel Distance: {response.total_distance} units")
    print(f"Total Time (travel + service): {response.total_time} units")
    print(f"  Service time: {response.total_time - response.total_distance} units")
    print("\nRoute with Service Times:")
    route = response.routes[0]
    print(f"  {' → '.join(route.sequence)} → {route.sequence[0]}")
    print()


async def demo_tsp_with_costs():
    """Demo 4: TSP with vehicle costs."""
    print("=" * 70)
    print("Demo 4: TSP with Cost Optimization")
    print("=" * 70)

    response = await solve_routing_problem(
        locations=[
            {"id": "warehouse", "coordinates": (0, 0)},
            {"id": "delivery_1", "coordinates": (8, 6)},
            {"id": "delivery_2", "coordinates": (12, 3)},
            {"id": "delivery_3", "coordinates": (5, 9)},
        ],
        vehicles=[
            {
                "id": "truck_1",
                "start_location": "warehouse",
                "cost_per_distance": 2.5,  # $2.50 per unit distance
                "fixed_cost": 50.0,  # $50 fixed cost for using truck
            }
        ],
        objective="minimize_cost",
    )

    print(f"Status: {response.status}")
    print(f"Vehicle: {response.routes[0].vehicle_id}")
    print(f"Total Distance: {response.total_distance} units")
    print("Cost Breakdown:")
    print("  Fixed cost: $50.00")
    print(
        f"  Variable cost: ${response.total_distance} units × $2.50/unit = ${response.total_distance * 2.5:.2f}"
    )
    print(f"  Total cost: ${response.total_cost:.2f}")
    print(f"\nRoute: {' → '.join(response.routes[0].sequence)} → {response.routes[0].sequence[0]}")
    print()


async def demo_delivery_route_optimization():
    """Demo 5: Real-world delivery route example."""
    print("=" * 70)
    print("Demo 5: Delivery Route Optimization")
    print("=" * 70)

    # Real delivery scenario: warehouse + 6 customers
    response = await solve_routing_problem(
        locations=[
            {"id": "Warehouse", "coordinates": (0, 0), "service_time": 0},
            {"id": "Restaurant_A", "coordinates": (2, 3), "service_time": 5},
            {"id": "Office_B", "coordinates": (5, 1), "service_time": 3},
            {"id": "Home_C", "coordinates": (7, 4), "service_time": 2},
            {"id": "School_D", "coordinates": (3, 6), "service_time": 4},
            {"id": "Hospital_E", "coordinates": (8, 8), "service_time": 6},
            {"id": "Mall_F", "coordinates": (1, 7), "service_time": 5},
        ],
        vehicles=[
            {
                "id": "DeliveryVan_1",
                "start_location": "Warehouse",
                "cost_per_distance": 1.5,
                "fixed_cost": 25.0,
            }
        ],
        objective="minimize_cost",
    )

    print(f"Status: {response.status}")
    print("Total Locations: 7 (warehouse + 6 deliveries)")
    print(f"Total Distance: {response.total_distance} km")
    print(f"Total Time: {response.total_time} minutes")
    print(f"Total Cost: ${response.total_cost:.2f}")
    print("\nOptimized Delivery Sequence:")

    route = response.routes[0]
    for i, location in enumerate(route.sequence):
        if i == 0:
            print(f"  {i + 1}. Start at {location}")
        else:
            print(f"  {i + 1}. Deliver to {location}")

    print(f"  {len(route.sequence) + 1}. Return to {route.sequence[0]}")
    print(f"\n{response.explanation.summary}")
    print()


async def demo_small_tsp():
    """Demo 6: Smallest TSP (3 cities)."""
    print("=" * 70)
    print("Demo 6: Small TSP (3 Cities)")
    print("=" * 70)

    response = await solve_routing_problem(
        locations=[
            {"id": "CityA", "coordinates": (0, 0)},
            {"id": "CityB", "coordinates": (4, 3)},
            {"id": "CityC", "coordinates": (3, 4)},
        ],
        objective="minimize_distance",
    )

    print(f"Status: {response.status}")
    print(f"Total Distance: {response.total_distance} units")

    # Show all location pairs
    print("\nLocation Coordinates:")
    print("  CityA: (0, 0)")
    print("  CityB: (4, 3)")
    print("  CityC: (3, 4)")

    print("\nOptimal Tour:")
    route = response.routes[0]
    print(f"  {' → '.join(route.sequence)} → {route.sequence[0]}")

    # Calculate and show each leg
    coords = {"CityA": (0, 0), "CityB": (4, 3), "CityC": (3, 4)}
    print("\nRoute Breakdown:")
    sequence = route.sequence
    total_manual = 0
    for i in range(len(sequence)):
        from_city = sequence[i]
        to_city = sequence[(i + 1) % len(sequence)]
        from_coords = coords[from_city]
        to_coords = coords[to_city]
        dist = int(
            ((from_coords[0] - to_coords[0]) ** 2 + (from_coords[1] - to_coords[1]) ** 2) ** 0.5
        )
        total_manual += dist
        print(f"  {from_city} → {to_city}: {dist} units")
    print(f"  Total: {total_manual} units")
    print()


async def main():
    """Run all routing demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "ROUTING PROBLEM EXAMPLES" + " " * 26 + "║")
    print("║" + " " * 68 + "║")
    print("║  High-level routing API for TSP and VRP optimization" + " " * 15 + "║")
    print("║  with coordinates, distance matrices, and costs." + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")

    await demo_simple_tsp()
    await demo_tsp_with_distance_matrix()
    await demo_tsp_with_service_times()
    await demo_tsp_with_costs()
    await demo_delivery_route_optimization()
    await demo_small_tsp()

    print("=" * 70)
    print("All routing demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
