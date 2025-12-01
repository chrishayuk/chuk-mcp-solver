"""Example: Project scheduling with resource constraints.

Demonstrates optimization, precedence constraints, and makespan minimization.
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_project_model(
    tasks: list[dict],
    dependencies: list[tuple[str, str]],
    resources: dict[str, int],
) -> dict:
    """Build a project scheduling model.

    Args:
        tasks: List of tasks with id, duration, and resource.
        dependencies: List of (predecessor, successor) task ID pairs.
        resources: Dict of resource_name -> max_parallel_tasks.

    Returns:
        Model dictionary.
    """
    variables = []
    constraints = []

    # Create start time variables for each task
    max_time = sum(t["duration"] for t in tasks)  # Upper bound
    for task in tasks:
        variables.append(
            {
                "id": f"start_{task['id']}",
                "domain": {"type": "integer", "lower": 0, "upper": max_time},
                "metadata": {"task_id": task["id"], "type": "start_time"},
            }
        )
        # Also need end time for makespan
        variables.append(
            {
                "id": f"end_{task['id']}",
                "domain": {"type": "integer", "lower": 0, "upper": max_time},
                "metadata": {"task_id": task["id"], "type": "end_time"},
            }
        )

    # Add makespan variable (max end time)
    variables.append(
        {
            "id": "makespan",
            "domain": {"type": "integer", "lower": 0, "upper": max_time},
            "metadata": {"type": "makespan"},
        }
    )

    # Duration constraints: end = start + duration
    for task in tasks:
        constraints.append(
            {
                "id": f"duration_{task['id']}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": f"end_{task['id']}", "coef": 1},
                        {"var": f"start_{task['id']}", "coef": -1},
                    ],
                    "sense": "==",
                    "rhs": task["duration"],
                },
                "metadata": {"description": f"Duration of task {task['id']}"},
            }
        )

    # Precedence constraints: successor starts after predecessor ends
    for pred, succ in dependencies:
        constraints.append(
            {
                "id": f"precedence_{pred}_to_{succ}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": f"start_{succ}", "coef": 1},
                        {"var": f"end_{pred}", "coef": -1},
                    ],
                    "sense": ">=",
                    "rhs": 0,
                },
                "metadata": {"description": f"Task {succ} starts after {pred} finishes"},
            }
        )

    # Makespan constraints: makespan >= end time of all tasks
    for task in tasks:
        constraints.append(
            {
                "id": f"makespan_{task['id']}",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": "makespan", "coef": 1},
                        {"var": f"end_{task['id']}", "coef": -1},
                    ],
                    "sense": ">=",
                    "rhs": 0,
                },
                "metadata": {"description": f"Makespan covers task {task['id']}"},
            }
        )

    # Objective: minimize makespan
    objective = {
        "sense": "min",
        "terms": [{"var": "makespan", "coef": 1}],
        "metadata": {"description": "Minimize project duration"},
    }

    return {
        "mode": "optimize",
        "variables": variables,
        "constraints": constraints,
        "objective": objective,
    }


def display_schedule(solution_vars: list, tasks: list[dict]) -> None:
    """Display the project schedule.

    Args:
        solution_vars: List of SolutionVariable objects.
        tasks: Original task definitions.
    """
    # Extract values
    values = {var.id: int(var.value) for var in solution_vars}
    makespan = values["makespan"]

    print("\nOptimal Schedule:")
    print(f"Total Project Duration: {makespan} time units\n")

    # Build task schedule
    schedule = []
    for task in tasks:
        start = values[f"start_{task['id']}"]
        end = values[f"end_{task['id']}"]
        schedule.append(
            {
                "id": task["id"],
                "start": start,
                "end": end,
                "duration": task["duration"],
                "resource": task.get("resource", "default"),
            }
        )

    # Sort by start time
    schedule.sort(key=lambda x: x["start"])

    # Print schedule
    print("Task | Start | End | Duration | Resource")
    print("-----|-------|-----|----------|----------")
    for item in schedule:
        print(
            f"{item['id']:4} | {item['start']:5} | {item['end']:3} | "
            f"{item['duration']:8} | {item['resource']}"
        )

    # Print Gantt-like view
    print("\nGantt Chart:")
    for item in schedule:
        bar = " " * item["start"] + "█" * item["duration"]
        print(f"{item['id']:4} |{bar}")
    print(f"     0{' ' * (makespan - 1)}{makespan}")


async def main() -> None:
    """Run the project scheduler example."""
    print("=== Project Scheduler Example ===\n")

    # Define tasks
    tasks = [
        {"id": "A", "duration": 3, "resource": "alice"},
        {"id": "B", "duration": 2, "resource": "bob"},
        {"id": "C", "duration": 4, "resource": "alice"},
        {"id": "D", "duration": 2, "resource": "bob"},
    ]

    # Define dependencies (A -> C, B -> D)
    dependencies = [
        ("A", "C"),
        ("B", "D"),
    ]

    # Resources (not enforcing capacity in this simple example)
    resources = {"alice": 1, "bob": 1}

    print("Tasks:")
    for task in tasks:
        print(f"  {task['id']}: duration={task['duration']}, resource={task['resource']}")

    print("\nDependencies:")
    for pred, succ in dependencies:
        print(f"  {pred} -> {succ}")

    # Build model
    model_dict = build_project_model(tasks, dependencies, resources)
    request = SolveConstraintModelRequest(**model_dict)

    # Solve
    print("\nOptimizing schedule...")
    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    # Display results
    print(f"\nStatus: {response.status}")
    if response.objective_value is not None:
        print(f"Objective Value (Makespan): {response.objective_value:.0f}")

    if response.solutions:
        display_schedule(response.solutions[0].variables, tasks)

    if response.explanation:
        print(f"\n{response.explanation.summary}")
        if response.explanation.binding_constraints:
            print("\nBinding Constraints (critical path indicators):")
            for bc in response.explanation.binding_constraints[:3]:
                desc = bc.metadata.get("description", bc.id) if bc.metadata else bc.id
                print(f"  - {desc}")


if __name__ == "__main__":
    asyncio.run(main())
