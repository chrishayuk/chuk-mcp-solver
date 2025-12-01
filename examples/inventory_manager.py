"""Example: Inventory management using reservoir constraints.

Demonstrates reservoir constraints for managing stock levels with
production and consumption events over time.
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_inventory_model(
    events: list[dict], initial_stock: int, min_stock: int, max_stock: int, horizon: int
) -> dict:
    """Build an inventory management model using reservoir constraints.

    Args:
        events: List of production/consumption events
        initial_stock: Starting inventory level
        min_stock: Minimum allowed inventory (safety stock)
        max_stock: Maximum storage capacity
        horizon: Time horizon for planning

    Returns:
        Model dictionary ready for SolveConstraintModelRequest.
    """
    variables = []
    constraints = []

    # Create time variables for each event
    time_vars = []
    level_changes = []

    for event in events:
        event_id = event["id"]
        change = event["change"]  # positive for production, negative for consumption
        earliest = event.get("earliest", 0)
        latest = event.get("latest", horizon)

        # Time when event occurs
        variables.append(
            {
                "id": f"time_{event_id}",
                "domain": {"type": "integer", "lower": earliest, "upper": latest},
                "metadata": {
                    "event": event_id,
                    "type": event["type"],
                    "change": change,
                    "description": event.get("description", ""),
                },
            }
        )
        time_vars.append(f"time_{event_id}")
        level_changes.append(change)

    # Add reservoir constraint - manages stock levels
    constraints.append(
        {
            "id": "inventory_levels",
            "kind": "reservoir",
            "params": {
                "time_vars": time_vars,
                "level_changes": level_changes,
                "min_level": min_stock,
                "max_level": max_stock,
            },
            "metadata": {
                "description": f"Inventory must stay between {min_stock} and {max_stock} units",
                "initial_stock": initial_stock,
            },
        }
    )

    # Precedence constraints (some events must happen before others)
    for event in events:
        if "must_precede" in event:
            for successor in event["must_precede"]:
                constraints.append(
                    {
                        "id": f"precedence_{event['id']}_{successor}",
                        "kind": "linear",
                        "params": {
                            "terms": [
                                {"var": f"time_{successor}", "coef": 1},
                                {"var": f"time_{event['id']}", "coef": -1},
                            ],
                            "sense": ">=",
                            "rhs": 1,  # At least 1 time unit apart
                        },
                        "metadata": {
                            "description": f"{event['id']} must complete before {successor}"
                        },
                    }
                )

    # Create makespan variable (total planning horizon used)
    variables.append(
        {
            "id": "makespan",
            "domain": {"type": "integer", "lower": 0, "upper": horizon},
            "metadata": {"type": "objective"},
        }
    )

    # Makespan must be >= all event times
    for event in events:
        constraints.append(
            {
                "id": f"makespan_{event['id']}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": "makespan", "coef": 1},
                        {"var": f"time_{event['id']}", "coef": -1},
                    ],
                    "sense": ">=",
                    "rhs": 0,
                },
            }
        )

    return {
        "mode": "optimize",
        "variables": variables,
        "constraints": constraints,
        "objective": {
            "sense": "min",
            "terms": [{"var": "makespan", "coef": 1}],
        },
    }


async def main():
    """Run inventory management example."""
    print("=== Inventory Management with Reservoir Constraints ===\n")

    # Define production and consumption events
    events = [
        {
            "id": "delivery_1",
            "type": "production",
            "change": 50,  # Receive 50 units
            "earliest": 0,
            "latest": 10,
            "description": "First supplier delivery",
        },
        {
            "id": "order_1",
            "type": "consumption",
            "change": -30,  # Ship 30 units
            "earliest": 2,
            "latest": 15,
            "description": "Customer order A",
            "must_precede": [],  # Can add dependencies here
        },
        {
            "id": "delivery_2",
            "type": "production",
            "change": 40,
            "earliest": 5,
            "latest": 20,
            "description": "Second supplier delivery",
        },
        {
            "id": "order_2",
            "type": "consumption",
            "change": -25,
            "earliest": 8,
            "latest": 25,
            "description": "Customer order B",
        },
        {
            "id": "order_3",
            "type": "consumption",
            "change": -35,
            "earliest": 12,
            "latest": 30,
            "description": "Customer order C",
        },
    ]

    initial_stock = 20  # Starting inventory
    min_stock = 10  # Safety stock (never go below)
    max_stock = 80  # Storage capacity

    print("Inventory Constraints:")
    print(f"  Initial Stock: {initial_stock} units")
    print(f"  Safety Stock: {min_stock} units (minimum)")
    print(f"  Storage Capacity: {max_stock} units (maximum)")

    print("\nEvents to Schedule:")
    print("Event       | Type        | Change | Window    | Description")
    print("------------|-------------|--------|-----------|---------------------------")
    for event in events:
        change_str = f"{event['change']:+3}"
        window = f"[{event['earliest']}, {event['latest']}]"
        print(
            f"{event['id']:11} | {event['type']:11} | {change_str:6} | {window:9} | {event['description']}"
        )

    print("\nOptimizing schedule to maintain inventory levels...\n")

    # Build and solve
    model = build_inventory_model(events, initial_stock, min_stock, max_stock, 35)
    request = SolveConstraintModelRequest(**model)

    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}")

    if response.status.value in ["optimal", "feasible"]:
        # Extract solution
        var_map = {var.id: var.value for var in response.solutions[0].variables}
        makespan = var_map["makespan"]

        print(f"Planning Horizon: {int(makespan)} time units")
        print(f"Objective Value: {response.objective_value}\n")

        # Build schedule
        schedule = []
        for event in events:
            time = int(var_map[f"time_{event['id']}"])
            schedule.append(
                {
                    "time": time,
                    "event": event["id"],
                    "type": event["type"],
                    "change": event["change"],
                    "description": event["description"],
                }
            )

        # Sort by time
        schedule.sort(key=lambda x: x["time"])

        print("Optimal Schedule:")
        print("Time | Event       | Type        | Change | Stock After | Description")
        print(
            "-----|-------------|-------------|--------|-------------|---------------------------"
        )

        stock = initial_stock
        for item in schedule:
            stock += item["change"]
            change_str = f"{item['change']:+3}"
            print(
                f"{item['time']:4} | {item['event']:11} | {item['type']:11} | "
                f"{change_str:6} | {stock:11} | {item['description']}"
            )

        # Show inventory timeline
        print("\nInventory Level Over Time:")
        print("Time | Stock | Status")
        print("-----|-------|----------------------------------")

        # Track stock level at each time point
        stock_timeline = {}
        stock_timeline[0] = initial_stock

        for item in schedule:
            t = item["time"]
            if t not in stock_timeline:
                # Find stock level just before this event
                prev_stock = initial_stock
                for prev_t in sorted(stock_timeline.keys()):
                    if prev_t < t:
                        prev_stock = stock_timeline[prev_t]
                stock_timeline[t] = prev_stock

            stock_timeline[t] += item["change"]

        for t in sorted(stock_timeline.keys()):
            stock = stock_timeline[t]
            bar = "█" * (stock // 5)

            # Check status
            if stock < min_stock:
                status = "⚠️  BELOW SAFETY STOCK"
            elif stock > max_stock:
                status = "⚠️  OVER CAPACITY"
            elif stock == min_stock:
                status = "At safety stock level"
            elif stock == max_stock:
                status = "At capacity"
            else:
                status = "OK"

            print(f"{t:4} | {stock:5} | {bar:20} {status}")

        # Show explanation
        if response.explanation:
            print(f"\n{response.explanation.summary}")

    else:
        print(f"Could not find solution: {response.status}")
        if response.explanation:
            print(f"\n{response.explanation.summary}")


if __name__ == "__main__":
    asyncio.run(main())
