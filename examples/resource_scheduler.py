"""Example: Resource scheduling with cumulative constraints.

Demonstrates cumulative constraints for resource capacity management,
useful for CPU/memory allocation, worker assignment, or any resource-limited scheduling.
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_resource_scheduling_model(
    tasks: list[dict], resource_capacity: int, horizon: int
) -> dict:
    """Build a resource scheduling model with cumulative constraints.

    Args:
        tasks: List of tasks with duration and resource demand
        resource_capacity: Maximum resource units available
        horizon: Time horizon for scheduling

    Returns:
        Model dictionary ready for SolveConstraintModelRequest.
    """
    variables = []
    constraints = []

    # Create start time variables for each task
    start_vars = []
    duration_vars = []
    demand_vars = []

    for task in tasks:
        task_id = task["id"]
        duration = task["duration"]
        demand = task["demand"]

        # Start time variable
        variables.append(
            {
                "id": f"start_{task_id}",
                "domain": {"type": "integer", "lower": 0, "upper": horizon - duration},
                "metadata": {"task": task_id, "type": "start_time"},
            }
        )
        start_vars.append(f"start_{task_id}")

        # Duration and demand (fixed values, but we represent as constants)
        duration_vars.append(duration)
        demand_vars.append(demand)

        # End time variable (for visibility in solution)
        variables.append(
            {
                "id": f"end_{task_id}",
                "domain": {"type": "integer", "lower": duration, "upper": horizon},
                "metadata": {"task": task_id, "type": "end_time"},
            }
        )

        # Link end = start + duration
        constraints.append(
            {
                "id": f"end_time_{task_id}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": f"end_{task_id}", "coef": 1},
                        {"var": f"start_{task_id}", "coef": -1},
                    ],
                    "sense": "==",
                    "rhs": duration,
                },
                "metadata": {"description": f"End time for {task_id}"},
            }
        )

    # Add cumulative constraint - resource capacity must not be exceeded
    constraints.append(
        {
            "id": "resource_capacity",
            "kind": "cumulative",
            "params": {
                "start_vars": start_vars,
                "duration_vars": duration_vars,
                "demand_vars": demand_vars,
                "capacity": resource_capacity,
            },
            "metadata": {"description": f"Resource capacity limit: {resource_capacity} units"},
        }
    )

    # Create makespan variable (project completion time)
    variables.append(
        {
            "id": "makespan",
            "domain": {"type": "integer", "lower": 0, "upper": horizon},
            "metadata": {"type": "objective"},
        }
    )

    # Makespan must be >= all end times
    for task in tasks:
        task_id = task["id"]
        constraints.append(
            {
                "id": f"makespan_end_{task_id}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": "makespan", "coef": 1},
                        {"var": f"end_{task_id}", "coef": -1},
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
    """Run resource scheduling example."""
    print("=== Resource Scheduling with Cumulative Constraints ===\n")

    # Define tasks: each task has duration and resource demand
    tasks = [
        {"id": "task_A", "duration": 3, "demand": 2, "name": "Database Migration"},
        {"id": "task_B", "duration": 4, "demand": 3, "name": "API Development"},
        {"id": "task_C", "duration": 2, "demand": 1, "name": "Unit Testing"},
        {"id": "task_D", "duration": 3, "demand": 2, "name": "Documentation"},
        {"id": "task_E", "duration": 2, "demand": 2, "name": "Code Review"},
    ]

    resource_capacity = 4  # Maximum 4 resource units available
    horizon = 15  # Maximum project duration

    print("Tasks:")
    print("Task   | Duration | Demand | Name")
    print("-------|----------|--------|--------------------")
    for task in tasks:
        print(f"{task['id']:6} | {task['duration']:8} | {task['demand']:6} | {task['name']}")

    print(f"\nResource Capacity: {resource_capacity} units")
    print(f"Time Horizon: {horizon} time units\n")

    # Build and solve
    model = build_resource_scheduling_model(tasks, resource_capacity, horizon)
    request = SolveConstraintModelRequest(**model)

    print("Optimizing schedule...\n")

    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}")

    if response.status.value in ["optimal", "feasible"]:
        # Extract solution
        var_map = {var.id: var.value for var in response.solutions[0].variables}
        makespan = var_map["makespan"]

        print(f"Minimum Project Duration: {int(makespan)} time units")
        print(f"Objective Value: {response.objective_value}\n")

        # Display schedule
        print("Optimal Schedule:")
        print("Task   | Start | End | Duration | Demand | Name")
        print("-------|-------|-----|----------|--------|--------------------")

        schedule = []
        for task in tasks:
            task_id = task["id"]
            start = int(var_map[f"start_{task_id}"])
            end = int(var_map[f"end_{task_id}"])
            schedule.append(
                {
                    "id": task_id,
                    "start": start,
                    "end": end,
                    "duration": task["duration"],
                    "demand": task["demand"],
                    "name": task["name"],
                }
            )

        # Sort by start time
        schedule.sort(key=lambda x: x["start"])

        for item in schedule:
            print(
                f"{item['id']:6} | {item['start']:5} | {item['end']:3} | "
                f"{item['duration']:8} | {item['demand']:6} | {item['name']}"
            )

        # Show resource utilization over time
        print("\nResource Utilization Timeline:")
        print("Time | Utilization | Running Tasks")
        print("-----|-------------|------------------")

        for t in range(int(makespan) + 1):
            active_tasks = []
            utilization = 0
            for item in schedule:
                if item["start"] <= t < item["end"]:
                    active_tasks.append(item["id"])
                    utilization += item["demand"]

            task_list = ", ".join(active_tasks) if active_tasks else "-"
            bar = "█" * utilization
            print(f"{t:4} | {utilization}/{resource_capacity} {bar:10} | {task_list}")

        # Show explanation
        if response.explanation:
            print(f"\n{response.explanation.summary}")
            if response.explanation.binding_constraints:
                print(f"\nBinding Constraints: {len(response.explanation.binding_constraints)}")
                for bc in response.explanation.binding_constraints[:3]:
                    if bc.metadata:
                        print(f"  - {bc.metadata.get('description', bc.id)}")
    else:
        print(f"Could not find solution: {response.status}")
        if response.explanation:
            print(f"\n{response.explanation.summary}")


if __name__ == "__main__":
    asyncio.run(main())
