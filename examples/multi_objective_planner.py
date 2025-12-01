"""Example: Multi-objective optimization with priority-based lexicographic ordering.

Demonstrates how to optimize multiple conflicting objectives with different priorities,
useful for trade-off analysis in planning and decision-making.
"""

import asyncio

from chuk_mcp_solver.models import SolveConstraintModelRequest
from chuk_mcp_solver.solver import get_solver


def build_cloud_deployment_model(instances: list[dict], requirements: dict) -> dict:
    """Build a multi-objective cloud deployment optimization model.

    Objectives (in priority order):
    1. Minimize cost (highest priority)
    2. Minimize latency (medium priority)
    3. Maximize reliability (lowest priority)

    Args:
        instances: Available instance types with specs
        requirements: Deployment requirements (CPU, memory, storage)

    Returns:
        Model dictionary ready for SolveConstraintModelRequest.
    """
    variables = []
    constraints = []

    # Create count variables for each instance type
    count_vars = []
    for instance in instances:
        instance_id = instance["id"]
        variables.append(
            {
                "id": f"count_{instance_id}",
                "domain": {"type": "integer", "lower": 0, "upper": 10},
                "metadata": {
                    "instance_type": instance_id,
                    "cost_per_unit": instance["cost"],
                    "latency": instance["latency"],
                    "reliability": instance["reliability"],
                },
            }
        )
        count_vars.append(instance_id)

    # CPU requirement
    cpu_terms = []
    for instance in instances:
        cpu_terms.append({"var": f"count_{instance['id']}", "coef": instance["cpu"]})

    constraints.append(
        {
            "id": "cpu_requirement",
            "kind": "linear",
            "params": {
                "terms": cpu_terms,
                "sense": ">=",
                "rhs": requirements["cpu"],
            },
            "metadata": {"description": f"Must provide at least {requirements['cpu']} CPU cores"},
        }
    )

    # Memory requirement
    memory_terms = []
    for instance in instances:
        memory_terms.append({"var": f"count_{instance['id']}", "coef": instance["memory"]})

    constraints.append(
        {
            "id": "memory_requirement",
            "kind": "linear",
            "params": {
                "terms": memory_terms,
                "sense": ">=",
                "rhs": requirements["memory"],
            },
            "metadata": {
                "description": f"Must provide at least {requirements['memory']} GB memory"
            },
        }
    )

    # Storage requirement
    storage_terms = []
    for instance in instances:
        storage_terms.append({"var": f"count_{instance['id']}", "coef": instance["storage"]})

    constraints.append(
        {
            "id": "storage_requirement",
            "kind": "linear",
            "params": {
                "terms": storage_terms,
                "sense": ">=",
                "rhs": requirements["storage"],
            },
            "metadata": {
                "description": f"Must provide at least {requirements['storage']} GB storage"
            },
        }
    )

    # Build objective terms directly from instance counts
    cost_terms = []
    latency_terms = []

    for instance in instances:
        cost_terms.append({"var": f"count_{instance['id']}", "coef": instance["cost"]})
        # Weighted average latency (approximation: sum of count * latency)
        latency_terms.append({"var": f"count_{instance['id']}", "coef": instance["latency"]})

    # Multi-objective optimization with priorities
    return {
        "mode": "optimize",
        "variables": variables,
        "constraints": constraints,
        "objective": [
            {
                "sense": "min",
                "terms": cost_terms,
                "priority": 2,  # Higher priority - minimize cost first
                "weight": 1.0,
            },
            {
                "sense": "min",
                "terms": latency_terms,
                "priority": 1,  # Lower priority - minimize latency second
                "weight": 1.0,
            },
        ],
    }


async def main():
    """Run multi-objective cloud deployment example."""
    print("=== Multi-Objective Cloud Deployment Optimization ===\n")

    # Define available instance types
    instances = [
        {
            "id": "t3_small",
            "name": "t3.small",
            "cpu": 2,
            "memory": 2,
            "storage": 20,
            "cost": 20,
            "latency": 50,
            "reliability": 95,
        },
        {
            "id": "t3_medium",
            "name": "t3.medium",
            "cpu": 2,
            "memory": 4,
            "storage": 30,
            "cost": 40,
            "latency": 45,
            "reliability": 96,
        },
        {
            "id": "c5_large",
            "name": "c5.large",
            "cpu": 4,
            "memory": 4,
            "storage": 40,
            "cost": 85,
            "latency": 30,
            "reliability": 98,
        },
        {
            "id": "r5_large",
            "name": "r5.large",
            "cpu": 2,
            "memory": 8,
            "storage": 50,
            "cost": 100,
            "latency": 40,
            "reliability": 97,
        },
    ]

    # Define requirements
    requirements = {
        "cpu": 6,  # cores
        "memory": 10,  # GB
        "storage": 80,  # GB
    }

    print("Available Instance Types:")
    print("Type       | CPU | Memory | Storage | Cost/hr | Latency | Reliability")
    print("-----------|-----|--------|---------|---------|---------|-------------")
    for inst in instances:
        print(
            f"{inst['name']:10} | {inst['cpu']:3} | {inst['memory']:6} | "
            f"{inst['storage']:7} | ${inst['cost']:6} | {inst['latency']:5}ms | {inst['reliability']:10}%"
        )

    print("\nDeployment Requirements:")
    print(f"  CPU: {requirements['cpu']} cores")
    print(f"  Memory: {requirements['memory']} GB")
    print(f"  Storage: {requirements['storage']} GB")

    print("\nObjective Priorities (lexicographic ordering):")
    print("  1. Minimize cost (highest priority)")
    print("  2. Minimize latency (lower priority)")

    print("\nOptimizing deployment...\n")

    # Build and solve
    model = build_cloud_deployment_model(instances, requirements)
    request = SolveConstraintModelRequest(**model)

    solver = get_solver("ortools")
    response = await solver.solve_constraint_model(request)

    print(f"Status: {response.status}")

    if response.status.value in ["optimal", "feasible"]:
        # Extract solution
        var_map = {var.id: var.value for var in response.solutions[0].variables}

        print(f"Combined Objective Value: {response.objective_value}\n")

        print("Optimal Deployment Configuration:")
        print("Type       | Count | Total CPU | Total Mem | Total Storage | Total Cost")
        print("-----------|-------|-----------|-----------|---------------|------------")

        total_cpu = 0
        total_memory = 0
        total_storage = 0
        deployment_cost = 0

        for inst in instances:
            count = int(var_map[f"count_{inst['id']}"])
            if count > 0:
                cpu_total = count * inst["cpu"]
                mem_total = count * inst["memory"]
                storage_total = count * inst["storage"]
                cost_total = count * inst["cost"]

                total_cpu += cpu_total
                total_memory += mem_total
                total_storage += storage_total
                deployment_cost += cost_total

                print(
                    f"{inst['name']:10} | {count:5} | {cpu_total:9} | "
                    f"{mem_total:9} | {storage_total:13} | ${cost_total:10}"
                )

        print("-" * 77)
        print(
            f"{'TOTAL':10} | {' ':5} | {total_cpu:9} | "
            f"{total_memory:9} | {total_storage:13} | ${deployment_cost:10}"
        )

        print("\nResource Utilization:")
        print(
            f"  CPU: {total_cpu}/{requirements['cpu']} cores "
            f"({100 * total_cpu // requirements['cpu']}%)"
        )
        print(
            f"  Memory: {total_memory}/{requirements['memory']} GB "
            f"({100 * total_memory // requirements['memory']}%)"
        )
        print(
            f"  Storage: {total_storage}/{requirements['storage']} GB "
            f"({100 * total_storage // requirements['storage']}%)"
        )

        # Show explanation
        if response.explanation:
            print(f"\n{response.explanation.summary}")
            if response.explanation.binding_constraints:
                print(f"\nBinding Constraints: {len(response.explanation.binding_constraints)}")
                for bc in response.explanation.binding_constraints[:5]:
                    if bc.metadata:
                        print(f"  - {bc.metadata.get('description', bc.id)}")

    else:
        print(f"Could not find solution: {response.status}")
        if response.explanation:
            print(f"\n{response.explanation.summary}")


if __name__ == "__main__":
    asyncio.run(main())
