"""Example: MCP tool/model selection under constraints.

Demonstrates implication constraints and practical MCP orchestration use case.
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_tool_selection_model(
    tasks: list[dict],
    tools: list[dict],
    budget: float,
    max_latency: float,
) -> dict:
    """Build a tool selection model.

    Args:
        tasks: List of tasks with id and required capability.
        tools: List of tools with id, cost, latency, and capabilities.
        budget: Maximum total cost.
        max_latency: Maximum acceptable average latency.

    Returns:
        Model dictionary.
    """
    variables = []
    constraints = []

    # Binary variable for each (task, tool) assignment
    for task in tasks:
        for tool in tools:
            variables.append(
                {
                    "id": f"assign_{task['id']}_to_{tool['id']}",
                    "domain": {"type": "bool"},
                    "metadata": {
                        "task": task["id"],
                        "tool": tool["id"],
                        "cost": tool["cost"],
                        "latency": tool["latency"],
                    },
                }
            )

    # Binary variable for whether each tool is used
    for tool in tools:
        variables.append(
            {
                "id": f"use_{tool['id']}",
                "domain": {"type": "bool"},
                "metadata": {"tool": tool["id"]},
            }
        )

    # Each task must be assigned to exactly one tool
    for task in tasks:
        constraints.append(
            {
                "id": f"task_{task['id']}_assigned",
                "kind": "linear",
                "params": {
                    "terms": [
                        {"var": f"assign_{task['id']}_to_{tool['id']}", "coef": 1}
                        for tool in tools
                        if task["capability"] in tool["capabilities"]
                    ],
                    "sense": "==",
                    "rhs": 1,
                },
                "metadata": {"description": f"Task {task['id']} must be assigned"},
            }
        )

    # Link assignment to tool usage (if task assigned to tool, tool must be used)
    for tool in tools:
        for task in tasks:
            if task["capability"] in tool["capabilities"]:
                constraints.append(
                    {
                        "id": f"link_{task['id']}_to_{tool['id']}",
                        "kind": "implication",
                        "params": {
                            "if_var": f"assign_{task['id']}_to_{tool['id']}",
                            "then": {
                                "id": f"use_{tool['id']}_implied",
                                "kind": "linear",
                                "params": {
                                    "terms": [{"var": f"use_{tool['id']}", "coef": 1}],
                                    "sense": "==",
                                    "rhs": 1,
                                },
                            },
                        },
                    }
                )

    # Budget constraint: total cost of used tools <= budget
    constraints.append(
        {
            "id": "budget",
            "kind": "linear",
            "params": {
                "terms": [{"var": f"use_{tool['id']}", "coef": tool["cost"]} for tool in tools],
                "sense": "<=",
                "rhs": budget,
            },
            "metadata": {"description": f"Total cost cannot exceed {budget}"},
        }
    )

    # Objective: minimize total latency (sum of latencies for assigned tasks)
    objective_terms = []
    for task in tasks:
        for tool in tools:
            if task["capability"] in tool["capabilities"]:
                objective_terms.append(
                    {"var": f"assign_{task['id']}_to_{tool['id']}", "coef": tool["latency"]}
                )

    objective = {
        "sense": "min",
        "terms": objective_terms,
        "metadata": {"description": "Minimize total latency"},
    }

    return {
        "mode": "optimize",
        "variables": variables,
        "constraints": constraints,
        "objective": objective,
    }


def display_solution(solution_vars: list, tasks: list[dict], tools: list[dict]) -> None:
    """Display the tool selection solution.

    Args:
        solution_vars: List of SolutionVariable objects.
        tasks: Original task definitions.
        tools: Original tool definitions.
    """
    # Extract assignments
    assignments = {}
    used_tools = set()
    total_cost = 0
    total_latency = 0

    for var in solution_vars:
        if var.value == 1:
            if var.id.startswith("assign_"):
                # Extract task and tool from variable name
                task_id = var.metadata["task"]
                tool_id = var.metadata["tool"]
                assignments[task_id] = tool_id
                used_tools.add(tool_id)
                total_latency += var.metadata["latency"]

    # Calculate total cost
    tool_by_id = {tool["id"]: tool for tool in tools}
    for tool_id in used_tools:
        total_cost += tool_by_id[tool_id]["cost"]

    print("\nOptimal Tool Selection:")
    print(f"Total Cost: ${total_cost:.2f}")
    print(f"Total Latency: {total_latency:.0f}ms\n")

    print("Task Assignments:")
    print("Task | Tool | Cost | Latency")
    print("-----|------|------|--------")
    for task in tasks:
        tool_id = assignments.get(task["id"], "NONE")
        if tool_id != "NONE":
            tool = tool_by_id[tool_id]
            print(f"{task['id']:4} | {tool_id:4} | ${tool['cost']:4.2f} | {tool['latency']:4.0f}ms")

    print("\nTools Used:")
    for tool_id in sorted(used_tools):
        tool = tool_by_id[tool_id]
        print(f"  {tool_id}: ${tool['cost']:.2f}, {tool['latency']}ms, {tool['capabilities']}")


async def main() -> None:
    """Run the tool selector example."""
    print("=== MCP Tool/Model Selection Example ===\n")

    # Define tasks
    tasks = [
        {"id": "T1", "capability": "text"},
        {"id": "T2", "capability": "vision"},
        {"id": "T3", "capability": "text"},
        {"id": "T4", "capability": "text"},
    ]

    # Define available tools/models
    tools = [
        {"id": "GPT4", "cost": 10.0, "latency": 500, "capabilities": ["text", "vision"]},
        {"id": "GPT3", "cost": 2.0, "latency": 200, "capabilities": ["text"]},
        {"id": "CLIP", "cost": 1.0, "latency": 100, "capabilities": ["vision"]},
    ]

    budget = 15.0

    print("Tasks:")
    for task in tasks:
        print(f"  {task['id']}: requires {task['capability']}")

    print("\nAvailable Tools:")
    for tool in tools:
        print(f"  {tool['id']}: ${tool['cost']:.2f}, {tool['latency']}ms, {tool['capabilities']}")

    print(f"\nBudget: ${budget:.2f}")

    # Build model
    model_dict = build_tool_selection_model(tasks, tools, budget, max_latency=1000)
    request = SolveConstraintModelRequest(**model_dict)

    # Solve
    print("\nOptimizing tool selection...")
    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    # Display results
    print(f"\nStatus: {response.status}")
    if response.objective_value is not None:
        print(f"Objective Value (Total Latency): {response.objective_value:.0f}ms")

    if response.solutions:
        display_solution(response.solutions[0].variables, tasks, tools)

    if response.explanation:
        print(f"\n{response.explanation.summary}")


if __name__ == "__main__":
    asyncio.run(main())
